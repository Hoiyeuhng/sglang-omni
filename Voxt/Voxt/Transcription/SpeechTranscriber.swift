// SpeechTranscriber.swift
// Provides Speech Transcriber for transcription engines.

import Foundation
import Speech
import AVFoundation
import Combine
import AudioToolbox

@MainActor
class SpeechTranscriber: ObservableObject, TranscriberProtocol {
    private final class AudioSampleStore {
        private let lock = NSLock()
        private var samples: [Float] = []
        private var callbackCount = 0

        @discardableResult
        func append(_ newSamples: [Float]) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            callbackCount += 1
            samples.append(contentsOf: newSamples)
            return callbackCount == 1
        }

        func metrics() -> (callbacks: Int, samples: Int) {
            lock.lock()
            defer { lock.unlock() }
            return (callbackCount, samples.count)
        }

        func snapshot() -> [Float] {
            lock.lock()
            defer { lock.unlock() }
            return samples
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            callbackCount = 0
            samples.removeAll(keepingCapacity: false)
        }
    }

    var diagnosticSessionID: UUID?
    private let diagnostics: RecordingDiagnosticRecorder
    private(set) var recognitionGenerationID = UUID()
    @Published var isRecording = false
    @Published var audioLevel: Float = 0.0
    @Published var transcribedText = ""
    @Published var isEnhancing = false
    @Published var isFinalizingTranscription = false

    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private lazy var audioEngine = AVAudioEngine()
    private var didPrepareCapture = false
    private let sampleStore = AudioSampleStore()
    private var preferredInputDeviceID: AudioDeviceID?
    private var inputSampleRate: Double = 16000
    private var completedAudioArchiveURL: URL?

    private var finalizeTimeoutTask: Task<Void, Never>?
    private var hasDeliveredFinalResult = false
    private var latestRecognizedText = ""
    var sessionReportsPartialResultsOverride: Bool?

    var onTranscriptionFinished: ((String) -> Void)?
    private(set) var lastStartFailureMessage: String?

    init(diagnostics: RecordingDiagnosticRecorder = RecordingDiagnostics.shared, initializeRecognizer: Bool = true) {
        self.diagnostics = diagnostics
        if initializeRecognizer { refreshSpeechRecognizer(localeIdentifier: nil) }
    }

    func setPreferredInputDevice(_ deviceID: AudioDeviceID?) {
        preferredInputDeviceID = deviceID
    }

    func requestPermissions() async -> Bool {
        guard await RecordingPermissionRequest.speechRecognitionAccess() else { return false }
        return await RecordingPermissionRequest.microphoneAccess()
    }

    func consumeCompletedAudioArchiveURL() -> URL? {
        let url = completedAudioArchiveURL
        completedAudioArchiveURL = nil
        return url
    }

    func discardCompletedAudioArchive() {
        removeCompletedAudioArchiveIfNeeded()
    }

    func startRecording() {
        guard !isRecording else { return }
        lastStartFailureMessage = nil
        removeCompletedAudioArchiveIfNeeded()

        let settings = resolvedDictationSettings()
        refreshSpeechRecognizer(localeIdentifier: settings.localeIdentifier)

        guard let recognizer = speechRecognizer else {
            let message = AppLocalization.localizedString("Direct Dictation is unavailable for the current language.")
            lastStartFailureMessage = message
            VoxtLog.asrWarning("Speech transcriber start blocked: recognizer is unavailable for current locale.")
            return
        }
        if settings.prefersOnDeviceRecognition && !recognizer.supportsOnDeviceRecognition {
            let message = AppLocalization.localizedString("Direct Dictation on-device recognition is unavailable for the selected language.")
            lastStartFailureMessage = message
            VoxtLog.asrWarning(
                "Speech transcriber start blocked: on-device recognition is unavailable. locale=\(recognizer.locale.identifier)"
            )
            return
        }
        guard recognizer.isAvailable else {
            let message = AppLocalization.localizedString("Direct Dictation is temporarily unavailable. Try again in a moment.")
            lastStartFailureMessage = message
            VoxtLog.asrWarning("Speech transcriber start blocked: recognizer is not currently available.")
            return
        }

        cleanupSessionState()
        sampleStore.clear()
        transcribedText = ""
        audioLevel = 0
        hasDeliveredFinalResult = false
        latestRecognizedText = ""

        do {
            try startSpeechRecognition(recognizer: recognizer, settings: settings)
            isRecording = true
            lastStartFailureMessage = nil
        } catch {
            lastStartFailureMessage = AppLocalization.localizedString("Direct Dictation failed to start recording.")
            VoxtLog.asrError("Speech transcriber start recording failed: \(error)")
            stopAudioCapture()
            cleanupSessionState()
        }
    }

    func stopRecording() {
        guard isRecording else { return }

        stopAudioCapture()
        let metrics = sampleStore.metrics()
        diagnostics.record(.init(.captureSummary, sessionID: diagnosticSessionID, backend: .dictation, count: metrics.callbacks, sampleCount: metrics.samples))
        isRecording = false

        finalizeTimeoutTask?.cancel()
        finalizeTimeoutTask = Task { [weak self] in
            do { try await Task.sleep(for: .milliseconds(900)) } catch { return }
            await MainActor.run {
                self?.forceFinalizeIfNeeded()
            }
        }
    }

    func shutdownForApplicationTermination() {
        if !hasDeliveredFinalResult, didPrepareCapture {
            diagnostics.record(.init(.asrCompleted, sessionID: diagnosticSessionID, backend: .dictation,
                outcome: .cancelled, failure: .cancelled))
        }
        recognitionGenerationID = UUID()
        stopAudioCapture()
        cleanupSessionState()
        removeCompletedAudioArchiveIfNeeded()
        audioLevel = 0
        isEnhancing = false
        isFinalizingTranscription = false
        onTranscriptionFinished = nil
    }

    func restartCaptureForPreferredInputDevice() throws {
        guard isRecording else { return }
        let settings = resolvedDictationSettings()
        refreshSpeechRecognizer(localeIdentifier: settings.localeIdentifier)
        guard let recognizer = speechRecognizer else {
            throw NSError(
                domain: "Voxt.SpeechTranscriber",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Speech recognizer is unavailable."]
            )
        }
        if settings.prefersOnDeviceRecognition && !recognizer.supportsOnDeviceRecognition {
            throw NSError(
                domain: "Voxt.SpeechTranscriber",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "On-device recognition is unavailable for the selected language."]
            )
        }
        stopAudioCapture()
        try startSpeechRecognition(recognizer: recognizer, settings: settings)
    }

    private func cleanupSessionState() {
        recognitionGenerationID = UUID()
        finalizeTimeoutTask?.cancel()
        finalizeTimeoutTask = nil
        isRecording = false
        clearRecognitionPipeline(cancelTask: true)
        sampleStore.clear()
    }

    private func stopAudioCapture() {
        guard didPrepareCapture else { return }
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        didPrepareCapture = false
    }

    func forceFinalizeIfNeeded() {
        guard !hasDeliveredFinalResult else { return }
        finishRecognition(with: latestRecognizedText, outcome: .failed, failure: .timeout)
    }

    private func finishRecognition(
        with text: String,
        outcome: RecordingDiagnosticEvent.Outcome,
        failure: RecordingDiagnosticEvent.Failure = .none
    ) {
        guard !hasDeliveredFinalResult else { return }
        hasDeliveredFinalResult = true

        finalizeTimeoutTask?.cancel()
        finalizeTimeoutTask = nil
        diagnostics.record(.init(.asrCompleted, sessionID: diagnosticSessionID, backend: .dictation,
            outcome: outcome, failure: failure, count: text.count))
        clearRecognitionPipeline(cancelTask: true)
        stageCompletedAudioArchive()

        onTranscriptionFinished?(text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    func handleRecognitionUpdate(text: String?, isFinal: Bool, error: Error?, generationID: UUID) {
        guard generationID == recognitionGenerationID, !hasDeliveredFinalResult else { return }
        if let text { latestRecognizedText = text }
        if let error {
            let speechError = error as NSError
            if speechError.domain == "kAFAssistantErrorDomain", speechError.code == 1110 {
                finishRecognition(with: "", outcome: .empty)
            } else {
                let classified = RecordingDiagnosticEvent.Failure.classify(error)
                let failure: RecordingDiagnosticEvent.Failure = speechError.domain == "kAFAssistantErrorDomain" && speechError.code == 216
                    ? .cancelled : (classified == .unknown ? .runtime : classified)
                finishRecognition(with: failure == .cancelled ? "" : latestRecognizedText,
                    outcome: failure == .cancelled ? .cancelled : .failed, failure: failure)
            }
        } else if let text {
            if !isFinal, !text.isEmpty {
                diagnostics.record(.init(.asrFirstPartial, sessionID: diagnosticSessionID, backend: .dictation, count: text.count))
            }
            if (sessionReportsPartialResultsOverride ?? true) || isFinal { transcribedText = text }
            if isFinal { finishRecognition(with: text, outcome: text.isEmpty ? .empty : .success) }
        }
    }

    private func startSpeechRecognition(
        recognizer: SFSpeechRecognizer,
        settings: ResolvedDictationSettings
    ) throws {
        recognitionGenerationID = UUID()
        let generationID = recognitionGenerationID
        clearRecognitionPipeline(cancelTask: true)

        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        audioEngine.reset()

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = settings.reportsPartialResults
        request.taskHint = .dictation
        request.contextualStrings = settings.contextualPhrases
        request.requiresOnDeviceRecognition = settings.prefersOnDeviceRecognition
        if #available(macOS 13.0, *) {
            request.addsPunctuation = settings.addsPunctuation
        }
        recognitionRequest = request

        let inputNode = audioEngine.inputNode
        let didApplyPreferredInputDevice = applyPreferredInputDeviceIfNeeded(inputNode: inputNode)
        let activeInputDeviceID = didApplyPreferredInputDevice ? preferredInputDeviceID : AudioInputDeviceManager.defaultInputDeviceID()
        let nodeOutputFormat = inputNode.outputFormat(forBus: 0)
        let hardwareSampleRate = AudioInputDeviceManager.nominalSampleRate(for: activeInputDeviceID)
        let tapFormat = AudioInputDeviceManager.captureTapFormat(
            nodeOutputFormat: nodeOutputFormat,
            hardwareSampleRate: hardwareSampleRate
        )
        inputSampleRate = tapFormat.sampleRate

        if abs(tapFormat.sampleRate - nodeOutputFormat.sampleRate) > 1 {
            VoxtLog.warning(
                "Speech transcriber adjusted input tap format. deviceID=\(activeInputDeviceID.map(String.init(describing:)) ?? "default"), hardwareSampleRate=\(hardwareSampleRate.map { String(Int($0.rounded())) } ?? "unknown"), nodeSampleRate=\(Int(nodeOutputFormat.sampleRate.rounded())), tapSampleRate=\(Int(tapFormat.sampleRate.rounded()))"
            )
        }

        let diagnosticSessionID = diagnosticSessionID
        let diagnostics = diagnostics
        didPrepareCapture = true
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) { [weak self] buffer, _ in
            guard let self else { return }
            self.recognitionRequest?.append(buffer)

            if let samples = AudioLevelMeter.monoSamples(from: buffer), !samples.isEmpty {
                let firstAudio = self.sampleStore.append(samples)
                if firstAudio {
                    Task { @MainActor in
                        diagnostics.record(.init(.asrReceivingAudio, sessionID: diagnosticSessionID,
                            backend: .dictation, sampleCount: samples.count))
                    }
                }
            }

            guard let channelData = buffer.floatChannelData?[0] else { return }
            let frameLength = Int(buffer.frameLength)
            if frameLength == 0 { return }

            var rms: Float = 0
            for i in 0..<frameLength {
                rms += channelData[i] * channelData[i]
            }
            rms = sqrt(rms / Float(frameLength))
            let normalized = min(rms * 20, 1.0)

            Task { @MainActor [weak self] in
                self?.audioLevel = normalized
            }
        }

        audioEngine.prepare()
        try audioEngine.start()

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            let text = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            Task { @MainActor in
                self?.handleRecognitionUpdate(text: text, isFinal: isFinal, error: error, generationID: generationID)
            }
        }
    }

    private func clearRecognitionPipeline(cancelTask: Bool) {
        if cancelTask {
            recognitionTask?.cancel()
        }
        recognitionTask = nil
        recognitionRequest = nil
    }

    @discardableResult
    private func applyPreferredInputDeviceIfNeeded(inputNode: AVAudioInputNode) -> Bool {
        guard let preferredInputDeviceID,
              preferredInputDeviceID != AudioDeviceID(kAudioObjectUnknown),
              AudioInputDeviceManager.isAvailableInputDevice(preferredInputDeviceID)
        else {
            return false
        }
        guard let audioUnit = inputNode.audioUnit else { return false }
        var deviceID = preferredInputDeviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &deviceID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        if status != noErr {
            VoxtLog.asrWarning("Unable to switch input device. status=\(status)")
            return false
        }
        return true
    }

    private func resolvedDictationSettings() -> ResolvedDictationSettings {
        let defaults = UserDefaults.standard
        let settings = ASRHintSettingsStore.resolvedSettings(
            for: .dictation,
            rawValue: defaults.string(forKey: AppPreferenceKey.asrHintSettings)
        )
        let userLanguageCodes = UserMainLanguageOption.storedSelection(
            from: defaults.string(forKey: AppPreferenceKey.userMainLanguageCodes)
        )
        let resolved = ASRHintResolver.resolveDictationSettings(
            settings: settings,
            userLanguageCodes: userLanguageCodes
        )
        if let override = sessionReportsPartialResultsOverride {
            return ResolvedDictationSettings(
                localeIdentifier: resolved.localeIdentifier,
                contextualPhrases: resolved.contextualPhrases,
                prefersOnDeviceRecognition: resolved.prefersOnDeviceRecognition,
                addsPunctuation: resolved.addsPunctuation,
                reportsPartialResults: override
            )
        }
        return resolved
    }

    private func refreshSpeechRecognizer(localeIdentifier: String?) {
        let locale = localeIdentifier.map(Locale.init(identifier:)) ?? Locale.current
        speechRecognizer = SFSpeechRecognizer(locale: locale)
        if speechRecognizer == nil {
            lastStartFailureMessage = AppLocalization.localizedString("Direct Dictation is unavailable for the current language.")
            VoxtLog.asrWarning("Speech recognizer initialization failed for locale=\(locale.identifier).")
        }
    }

    private func stageCompletedAudioArchive() {
        removeCompletedAudioArchiveIfNeeded()
        let samples = sampleStore.snapshot()
        guard !samples.isEmpty else { return }
        let tempURL = HistoryAudioArchiveSupport.temporaryArchiveURL(prefix: "voxt-speech-history")
        do {
            if try HistoryAudioArchiveSupport.exportWAV(samples: samples, sampleRate: inputSampleRate, to: tempURL) {
                completedAudioArchiveURL = tempURL
            }
        } catch {
            try? FileManager.default.removeItem(at: tempURL)
            VoxtLog.asrWarning("Speech completed audio archive export failed: \(error.localizedDescription)")
        }
    }

    private func removeCompletedAudioArchiveIfNeeded() {
        guard let completedAudioArchiveURL else { return }
        try? FileManager.default.removeItem(at: completedAudioArchiveURL)
        self.completedAudioArchiveURL = nil
    }
}
