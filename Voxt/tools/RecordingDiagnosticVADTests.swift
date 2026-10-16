import Foundation

@main
struct RecordingDiagnosticVADTests {
    static func main() async throws {
        let suiteName = "Voxt.Diagnostics.VAD.Tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let output = VADTestOutput()
        let recorder = RecordingDiagnosticRecorder(defaults: defaults, sink: { output.append($0) })
        recorder.setEnabled(true)
        let sessionID = UUID()
        recorder.record(.init(.sessionStarted, sessionID: sessionID))
        let frame = ASRVoiceActivityAudioFrame(samples: [0.8, 0.8], sampleRate: 16_000,
            startSeconds: 0, endSeconds: 0.1, level: 0.8)
        let silero = RecordingVoiceActivityFrameDecider(mode: .silero, useCase: .transcription,
            energyThreshold: 0.2, diagnosticSessionID: sessionID, diagnostics: recorder)
        let fallback = await silero.decision(for: frame)!
        precondition(fallback.source == .fallbackEnergy)
        precondition(fallback.decision.isSpeech)
        _ = await silero.decision(for: frame)
        var stats = RecordingVoiceActivityDebugStats(mode: .silero, useCase: .transcription, backend: .mlxSilero)
        stats.record(frame: frame, result: fallback, resolvedSpeech: true, events: [], fallbackSource: .fallbackEnergy)
        precondition(stats.totalFrames == 1 && stats.fallbackFrames == 1 && stats.speechFrames == 1)
        await silero.releaseResources()
        let omni = RecordingVoiceActivityFrameDecider(mode: .omni, useCase: .transcription,
            energyThreshold: 0.2, diagnosticSessionID: sessionID, diagnostics: recorder)
        let success = await omni.decision(for: frame)!
        precondition(success.source == .omni && success.decision.isSpeech)
        await VADTestControl.shared.setOmniFailure(true)
        let omniFallback = await omni.decision(for: frame)!
        precondition(omniFallback.source == .fallbackEnergy)
        await omni.releaseResources()
        let events = try output.snapshot().map { try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data($0.utf8)) }
        let fallbacks = events.filter { $0.kind == .vadFallback }
        precondition(fallbacks.count == 2)
        precondition(fallbacks.allSatisfy { $0.sessionID == sessionID && $0.failure == .timeout && $0.outcome == .fallback })
        precondition(events.filter { $0.kind == .vadReleased }.count == 2)
        precondition(!output.snapshot().joined().contains("private"))
        print("PASS 9 VAD assertions using production planning/decider/statistics and injected failing backends")
    }
}

nonisolated final class VADTestOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func append(_ line: String) { lock.lock(); lines.append(line); lock.unlock() }
    func snapshot() -> [String] { lock.lock(); defer { lock.unlock() }; return lines }
}

actor VADTestControl {
    static let shared = VADTestControl()
    private(set) var failOmni = false
    func setOmniFailure(_ value: Bool) { failOmni = value }
}

actor ASRSileroStreamingVoiceActivityDetector {
    init(diagnosticSessionID: UUID? = nil, diagnostics: RecordingDiagnosticRecorder) {}
    func probability(samples: [Float], sampleRate: Double, streamID: String) async throws -> Float? {
        throw NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut,
            userInfo: [NSLocalizedDescriptionKey: "private VAD error /private/model token=secret"])
    }
    func reset() {}
    func unload() {}
}

actor OmniStreamVoiceActivityBackend {
    init(useCase: ASRVoiceActivityUseCase) {}
    func decision(for frame: ASRVoiceActivityAudioFrame, streamID: String) async throws -> ASRVoiceActivityFrameDecision? {
        if await VADTestControl.shared.failOmni { throw URLError(.timedOut) }
        return ASRVoiceActivityFrameDecision(startSeconds: frame.startSeconds, endSeconds: frame.endSeconds,
            isSpeech: true, probability: 0.9)
    }
    func reset() {}
    func releaseResources() {}
}

nonisolated enum MeetingVADModelError: Error { case modelNotDownloaded, runtimeUnavailable }
nonisolated enum TranscriptionEngine { case mlxAudio, dictation, remote }
nonisolated enum AppPreferenceKey {
    static let localVADMode = "test.localVADMode"
    static let meetingSileroVADSensitivity = "test.meetingSileroVADSensitivity"
}
nonisolated enum AppLocalization { static func localizedString(_ value: String) -> String { value } }
nonisolated enum VoxtLog { static func asrWarning(_ value: @autoclosure () -> String) {} }
extension Notification.Name { nonisolated static let voxtFeatureSettingsDidChange = Self("test.featureSettingsChanged") }

nonisolated enum VoxtLogger {
    static func recordDiagnosticLine(_ line: String) { fatalError("Tests must inject the diagnostic sink") }
}
