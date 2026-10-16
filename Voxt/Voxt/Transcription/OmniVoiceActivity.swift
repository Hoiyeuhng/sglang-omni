import Foundation

/// Silero VAD on the native runtime (`--model-kind silero_vad`): every
/// detector shares one server.
nonisolated enum OmniSileroVADRuntime {
    static let shared = OmniSharedModelRuntime(kind: .sileroVAD)

    /// Whether Silero runs on the native runtime in this process.
    static var isEnabled: Bool { OmniASRBackend.launchSettings != nil }
}

nonisolated enum OmniVoiceActivityError: LocalizedError, Equatable {
    case httpStatus(Int)
    case malformedResponse

    var errorDescription: String? {
        switch self {
        case .httpStatus(let status):
            return "The local voice activity server answered with HTTP \(status)."
        case .malformedResponse:
            return "The local voice activity server sent a malformed response."
        }
    }
}

/// One `/v1/vad/stream` socket: the server keeps this stream's Silero state.
actor OmniVoiceActivityStream {
    private let session: URLSession
    private let socket: URLSessionWebSocketTask
    private var tail: Task<Void, Never>?
    private let diagnostics: RecordingDiagnosticRecorder
    private let diagnosticSessionID: UUID?
    private let diagnosticRuntimeID: UUID?
    private let diagnosticRequestID = UUID()
    private var completedExchanges = 0
    private var processedSamples = 0
    private var pendingExchanges = 0
    private var didCompleteDiagnostics = false
    private var isClosed = false

    init(
        endpoint: OmniServerEndpoint,
        diagnosticSessionID: UUID? = nil,
        diagnostics: RecordingDiagnosticRecorder = RecordingDiagnostics.shared
    ) {
        self.diagnostics = diagnostics
        self.diagnosticSessionID = diagnosticSessionID
        diagnosticRuntimeID = endpoint.diagnosticRuntimeID
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        session = URLSession(configuration: configuration)
        var components = URLComponents()
        components.scheme = "ws"
        components.host = endpoint.host
        components.port = endpoint.port
        components.path = "/v1/vad/stream"
        socket = session.webSocketTask(with: components.url!)
        socket.resume()
        diagnostics.record(.init(.requestStarted, sessionID: diagnosticSessionID,
            requestID: diagnosticRequestID, runtimeID: diagnosticRuntimeID, backend: .silero))
    }

    /// Feeds 16 kHz samples; the probability of the last whole 512-sample
    /// chunk they completed, or nil when none completed.
    func probability(samples16k: [Float]) async throws -> Float? {
        pendingExchanges += 1
        defer { pendingExchanges -= 1 }
        let previous = tail
        let socket = socket
        // Note (Jiaxin Deng): replies arrive in order, so one exchange at a time per stream.
        let exchange = Task { () throws -> Float? in
            await previous?.value
            try await socket.send(.data(Self.float32LittleEndian(samples16k)))
            guard case .string(let text) = try await socket.receive() else {
                throw OmniVoiceActivityError.malformedResponse
            }
            return try Self.probability(fromReply: text)
        }
        tail = Task { _ = try? await exchange.value }
        do {
            let probability = try await exchange.value
            if !didCompleteDiagnostics {
                completedExchanges += 1
                processedSamples += samples16k.count
                if completedExchanges == 1 {
                    diagnostics.record(.init(.vadStreamReady, sessionID: diagnosticSessionID,
                        requestID: diagnosticRequestID, runtimeID: diagnosticRuntimeID, backend: .silero))
                }
            }
            return probability
        } catch {
            let failure = RecordingDiagnosticEvent.Failure.classify(error)
            completeDiagnostics(outcome: failure == .cancelled ? .cancelled : .failed,
                failure: failure == .unknown ? .runtime : failure)
            throw error
        }
    }

    func close() {
        guard !isClosed else { return }
        isClosed = true
        completeDiagnostics(outcome: pendingExchanges > 0 ? .cancelled : (completedExchanges > 0 ? .success : .empty),
            failure: pendingExchanges > 0 ? .cancelled : .none)
        socket.cancel(with: .normalClosure, reason: nil)
        session.invalidateAndCancel()
        diagnostics.record(.init(.requestCleanup, sessionID: diagnosticSessionID,
            requestID: diagnosticRequestID, runtimeID: diagnosticRuntimeID, backend: .silero))
    }

    private func completeDiagnostics(outcome: RecordingDiagnosticEvent.Outcome, failure: RecordingDiagnosticEvent.Failure) {
        guard !didCompleteDiagnostics else { return }
        didCompleteDiagnostics = true
        diagnostics.record(.init(.requestCompleted, sessionID: diagnosticSessionID,
            requestID: diagnosticRequestID, runtimeID: diagnosticRuntimeID, backend: .silero,
            outcome: outcome, failure: failure, count: completedExchanges, sampleCount: processedSamples))
    }

    nonisolated static func float32LittleEndian(_ samples: [Float]) -> Data {
        var data = Data(capacity: samples.count * 4)
        for sample in samples {
            withUnsafeBytes(of: sample.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    nonisolated static func probability(fromReply text: String) throws -> Float? {
        guard let object = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
              object.keys.contains("probability")
        else { throw OmniVoiceActivityError.malformedResponse }
        if object["probability"] is NSNull { return nil }
        guard let probability = object["probability"] as? NSNumber else {
            throw OmniVoiceActivityError.malformedResponse
        }
        return probability.floatValue
    }
}

nonisolated enum OmniVoiceActivityRequests {
    struct Options: Sendable, Equatable {
        var threshold: Float
        var minSpeechDurationMs: Int
        var minSilenceDurationMs: Int
        var speechPadMs: Int
    }

    /// Speech ranges in 16 kHz samples (Swift `getSpeechTimestamps`).
    static func speechTimestamps(
        samples16k: [Float],
        options: Options,
        endpoint: OmniServerEndpoint
    ) async throws -> [Range<Int>] {
        let boundary = "voxt-\(UUID().uuidString)"
        var request = URLRequest(url: endpoint.baseURL.appendingPathComponent("v1/vad/speech_timestamps"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let fields = [
            ("threshold", String(options.threshold)),
            ("min_speech_duration_ms", String(options.minSpeechDurationMs)),
            ("min_silence_duration_ms", String(options.minSilenceDurationMs)),
            ("speech_pad_ms", String(options.speechPadMs)),
        ]
        var body = Data()
        for (name, value) in fields {
            body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".utf8))
        }
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".utf8))
        body.append(OmniWAVEncoding.float32WAV(samples: samples16k, sampleRate: 16_000))
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        request.httpBody = body

        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        let (data, response) = try await session.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else { throw OmniVoiceActivityError.httpStatus(status) }
        return try ranges(fromResponse: data)
    }

    static func ranges(fromResponse data: Data) throws -> [Range<Int>] {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let timestamps = object["timestamps"] as? [[String: Any]]
        else { throw OmniVoiceActivityError.malformedResponse }
        return try timestamps.map { timestamp in
            guard let start = timestamp["start"] as? Int,
                  let end = timestamp["end"] as? Int,
                  start <= end
            else { throw OmniVoiceActivityError.malformedResponse }
            return start ..< end
        }
    }
}
