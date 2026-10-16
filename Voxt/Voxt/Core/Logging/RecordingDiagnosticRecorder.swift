import Foundation

nonisolated struct RecordingDiagnosticEvent: Codable, Sendable {
    enum Kind: String, Codable, Sendable {
        case sessionStarted, inputPreparing, inputReady, recordingStopped
        case sessionFailed, sessionCancelled, sessionFinished, sessionCleanup
        case captureSummary, asrInitializing, asrReceivingAudio, asrFirstPartial, asrCompleted
        case vadStarted, vadFallback, vadSummary, vadReleased
        case runtimeStarted, runtimeReady, runtimeModelLoaded, runtimeFailed, runtimeExited, runtimeReleased
        case requestStarted, requestFirstPartial, requestCompleted, requestCleanup
        case enhancementStarted, enhancementCompleted, insertionAttempted, insertionCompleted
        case applicationStarted, applicationStopping, applicationStopped, legacyDiagnostic
        case hotkeyConfigured, hotkeyMonitorStarted, hotkeyMonitorStopped, hotkeyMonitorFailed
        case hotkeyStateReset, hotkeyPressed, hotkeyReleased, hotkeyIgnored, hotkeyFallback
        case hotkeyScheduled, hotkeyCancelled, hotkeyDispatched, hotkeyBindingCaptured
        case hotkeyModifierChanged, hotkeyKeyObserved, voiceStopDetected, voiceStopCleared
        case modelLoadStarted, modelLoadCompleted, modelLoadFailed, modelLoadCancelled, modelReused, modelReleased
        case modelDownloadStarted, modelDownloadCompleted, modelDownloadPaused, modelDownloadCancelled, modelDownloadFailed
        case modelDownloadRetry, modelDownloadResumed, modelRoutingChanged
        case modelRequestStarted, modelRequestCompleted, modelRequestFailed, modelConfigured
        case modelInputReady, contextReadCompleted, contextReadFailed
        case modelRequestMetrics, modelStreamStopped, modelResultReceived, modelConfigurationChanged
        case modelAudioBufferFlushed, modelStopDeferred
        case outputEncodingIssue, hotkeyHandlingSlow
        case historyWriteFailed, noteWriteFailed, credentialReadFailed, credentialWriteFailed, credentialDeleteFailed
        case credentialMigrationFailed
    }

    enum Action: String, Codable, Sendable { case transcription, translation, rewrite, note, meeting, paste, stop }
    enum Severity: String, Codable, Sendable { case info, warning, error }
    enum FinishReason: String, Codable, Sendable { case tokenLimit, noGenerationBudget, contextLimit, eos, eot }
    enum Route: String, Codable, Sendable {
        case aliyunQwenRealtime, aliyunOmniRealtime, aliyunFunRealtime, aliyunFile, aliyunUnsupported
        case stepFunRealtime, geminiLive, providerDefault
    }

    enum Outcome: String, Codable, Sendable {
        case none, success, failed, cancelled, empty, fallback, shortcutPosted, clipboardOnly, notPosted
    }

    enum Backend: String, Codable, Sendable {
        case unspecified, appleIntelligence, mlx, dictation, remote, energy, silero, omni, gguf, off
    }

    enum Failure: String, Codable, Sendable {
        case none, cancelled, timeout, permission, unavailable, transport, runtime, input, storage, unknown

        static func classify(_ error: Error) -> Self {
            if error is CancellationError { return .cancelled }
            let value = error as NSError
            if value.domain == NSURLErrorDomain {
                if value.code == NSURLErrorCancelled { return .cancelled }
                return value.code == NSURLErrorTimedOut ? .timeout : .transport
            }
            if value.domain == NSCocoaErrorDomain {
                return .storage
            }
            return .unknown
        }
    }

    let kind: Kind
    var sessionID: UUID?
    var requestID: UUID?
    var runtimeID: UUID?
    var backend: Backend
    var outcome: Outcome
    var failure: Failure
    var category: VoxtLogCategory?
    var action: Action?
    var severity: Severity?
    var byteCount: UInt64?
    var errorCode: Int?
    var characterCount: Int?
    var tokenCount: Int?
    var durationMilliseconds: UInt64?
    var prefillMilliseconds: UInt64?
    var firstTokenMilliseconds: UInt64?
    var finishReason: FinishReason?
    var route: Route?
    var thinkingEnabled: Bool?
    var httpStatusCode: Int?
    var exitStatusCode: Int?
    var count: Int?
    var sampleCount: Int?
    var droppedChunkCount: Int?
    var droppedByteCount: Int?
    var fallbackFrameCount: Int?
    var speechFrameCount: Int?
    var elapsedMilliseconds: UInt64?
    var timestampMilliseconds: UInt64?

    init(
        _ kind: Kind, sessionID: UUID? = nil, requestID: UUID? = nil, runtimeID: UUID? = nil,
        backend: Backend = .unspecified, outcome: Outcome = .none, failure: Failure = .none,
        httpStatusCode: Int? = nil, exitStatusCode: Int? = nil,
        count: Int? = nil, sampleCount: Int? = nil, droppedChunkCount: Int? = nil, droppedByteCount: Int? = nil,
        fallbackFrameCount: Int? = nil, speechFrameCount: Int? = nil,
        action: Action? = nil, byteCount: UInt64? = nil, errorCode: Int? = nil,
        characterCount: Int? = nil, tokenCount: Int? = nil, durationMilliseconds: UInt64? = nil,
        prefillMilliseconds: UInt64? = nil, firstTokenMilliseconds: UInt64? = nil, finishReason: FinishReason? = nil,
        route: Route? = nil, thinkingEnabled: Bool? = nil
    ) {
        self.kind = kind
        self.sessionID = sessionID
        self.requestID = requestID
        self.runtimeID = runtimeID
        self.backend = backend
        self.outcome = outcome
        self.failure = failure
        self.action = action
        self.byteCount = byteCount
        self.errorCode = errorCode
        self.characterCount = characterCount.map { max(0, $0) }
        self.tokenCount = tokenCount.map { max(0, $0) }
        self.durationMilliseconds = durationMilliseconds
        self.prefillMilliseconds = prefillMilliseconds
        self.firstTokenMilliseconds = firstTokenMilliseconds
        self.finishReason = finishReason
        self.route = route
        self.thinkingEnabled = thinkingEnabled
        self.httpStatusCode = httpStatusCode
        self.exitStatusCode = exitStatusCode
        self.count = count.map { max(0, $0) }
        self.sampleCount = sampleCount.map { max(0, $0) }
        self.droppedChunkCount = droppedChunkCount.map { max(0, $0) }
        self.droppedByteCount = droppedByteCount.map { max(0, $0) }
        self.fallbackFrameCount = fallbackFrameCount.map { max(0, $0) }
        self.speechFrameCount = speechFrameCount.map { max(0, $0) }
    }

    func encodedLine() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }

    static func safeLine(_ line: String) -> String? {
        guard line.utf8.count <= 4_096,
              let event = try? JSONDecoder().decode(Self.self, from: Data(line.utf8)) else { return nil }
        return try? event.encodedLine()
    }
}

nonisolated final class RecordingDiagnosticRecorder: @unchecked Sendable {
    static let preferenceKey = "recordingDiagnosticsEnabled"
    private let lock = NSLock()
    private let defaults: UserDefaults
    private let monotonicMilliseconds: @Sendable () -> UInt64
    private let sink: @Sendable (String) throws -> Void
    private var enabled: Bool
    private var starts: [UUID: UInt64] = [:]
    private var firstPartials = Set<UUID>()
    private var completedASR = Set<UUID>()
    private var receivedAudio = Set<UUID>()
    private var sessionOutcomes: [UUID: RecordingDiagnosticEvent.Outcome] = [:]
    private var sessionFailures: [UUID: RecordingDiagnosticEvent.Failure] = [:]
    private var operationStarts: [UUID: UInt64] = [:]
    private var ended = Set<UUID>()
    private var writeFailures = 0

    init(
        defaults: UserDefaults,
        monotonicMilliseconds: @escaping @Sendable () -> UInt64 = { DispatchTime.now().uptimeNanoseconds / 1_000_000 },
        sink: @escaping @Sendable (String) throws -> Void
    ) {
        self.defaults = defaults
        self.monotonicMilliseconds = monotonicMilliseconds
        self.sink = sink
        enabled = defaults.bool(forKey: Self.preferenceKey)
    }

    var isEnabled: Bool { lock.lock(); defer { lock.unlock() }; return enabled }
    var failedWriteCount: Int { lock.lock(); defer { lock.unlock() }; return writeFailures }

    func setEnabled(_ value: Bool) {
        lock.lock()
        defer { lock.unlock() }
        enabled = value
        defaults.set(value, forKey: Self.preferenceKey)
        if !value {
            starts.removeAll()
            firstPartials.removeAll()
            completedASR.removeAll()
            receivedAudio.removeAll()
            sessionOutcomes.removeAll()
            sessionFailures.removeAll()
            operationStarts.removeAll()
            ended.removeAll()
        }
    }

    func record(_ makeEvent: @autoclosure () -> RecordingDiagnosticEvent) {
        lock.lock()
        defer { lock.unlock() }
        guard enabled else { return }
        var event = makeEvent()
        let now = monotonicMilliseconds()
        if let sessionID = event.sessionID {
            switch event.kind {
            case .sessionStarted, .requestStarted, .requestFirstPartial, .requestCompleted, .requestCleanup,
                 .runtimeStarted, .runtimeReady, .runtimeModelLoaded, .runtimeFailed, .runtimeExited, .runtimeReleased:
                break
            default:
                guard starts[sessionID] != nil else { return }
            }
            if event.kind == .sessionStarted {
                if starts.count >= 256, let oldest = starts.min(by: { $0.value < $1.value })?.key {
                    starts.removeValue(forKey: oldest)
                    firstPartials.remove(oldest)
                    completedASR.remove(oldest)
                    receivedAudio.remove(oldest)
                    sessionOutcomes.removeValue(forKey: oldest)
                    sessionFailures.removeValue(forKey: oldest)
                    ended.remove(oldest)
                }
                starts[sessionID] = now
                firstPartials.remove(sessionID)
                completedASR.remove(sessionID)
                receivedAudio.remove(sessionID)
                sessionOutcomes.removeValue(forKey: sessionID)
                sessionFailures.removeValue(forKey: sessionID)
                ended.remove(sessionID)
            }
            let hasActiveSession = starts[sessionID] != nil && !ended.contains(sessionID)
            if [.asrFirstPartial, .asrReceivingAudio, .asrCompleted, .inputReady, .asrInitializing,
                .enhancementStarted, .enhancementCompleted, .insertionAttempted, .insertionCompleted,
                .sessionFailed, .sessionCancelled, .sessionFinished].contains(event.kind), !hasActiveSession { return }
            if event.kind == .asrFirstPartial, !firstPartials.insert(sessionID).inserted { return }
            if event.kind == .asrReceivingAudio, !receivedAudio.insert(sessionID).inserted { return }
            if event.kind == .asrCompleted {
                guard completedASR.insert(sessionID).inserted else { return }
                if sessionOutcomes[sessionID] != .failed && sessionOutcomes[sessionID] != .cancelled {
                    sessionOutcomes[sessionID] = event.outcome
                }
            }
            if hasActiveSession, [.asrCompleted, .enhancementCompleted, .insertionCompleted].contains(event.kind),
               event.outcome == .failed || event.outcome == .cancelled || event.outcome == .notPosted || event.failure != .none {
                sessionOutcomes[sessionID] = event.outcome == .cancelled ? .cancelled : .failed
                sessionFailures[sessionID] = event.failure
            }
            if event.kind == .sessionFinished, let outcome = sessionOutcomes[sessionID] {
                event.outcome = outcome
                event.failure = sessionFailures[sessionID] ?? .none
            }
            if [.sessionFailed, .sessionCancelled, .sessionFinished].contains(event.kind),
               !ended.insert(sessionID).inserted { return }
            if let start = starts[sessionID] { event.elapsedMilliseconds = now >= start ? now - start : 0 }
            if event.kind == .sessionCleanup {
                starts.removeValue(forKey: sessionID)
                firstPartials.remove(sessionID)
                completedASR.remove(sessionID)
                receivedAudio.remove(sessionID)
                sessionOutcomes.removeValue(forKey: sessionID)
                sessionFailures.removeValue(forKey: sessionID)
                ended.remove(sessionID)
            }
        }
        if let operationID = event.requestID ?? event.runtimeID {
            if [.requestStarted, .runtimeStarted, .enhancementStarted, .insertionAttempted].contains(event.kind) {
                operationStarts[operationID] = now
            }
            if let start = operationStarts[operationID] { event.elapsedMilliseconds = now >= start ? now - start : 0 }
            if [.requestCleanup, .runtimeReleased, .enhancementCompleted, .insertionCompleted].contains(event.kind) {
                operationStarts.removeValue(forKey: operationID)
            }
            if operationStarts.count > 512 { operationStarts = [operationID: now] }
        }
        event.timestampMilliseconds = UInt64(max(0, Date().timeIntervalSince1970 * 1_000))
        do { try sink(event.encodedLine()) } catch { writeFailures += 1 }
    }
}

nonisolated enum RecordingDiagnosticContext {
    @TaskLocal static var sessionID: UUID?
}

nonisolated struct RecordingDiagnosticExportMetadata: Encodable {
    let schemaVersion = 1
    let macOSMajorVersion: Int
    let macOSMinorVersion: Int
    let macOSPatchVersion: Int
    let recordingDiagnosticsEnabled: Bool
    let diagnosticWriteFailed: Bool

    func encodedText() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let bytes = try? encoder.encode(self) else { return "{}" }
        return String(decoding: bytes, as: UTF8.self)
    }
}
