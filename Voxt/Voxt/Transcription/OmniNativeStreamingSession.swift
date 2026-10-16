import Foundation
import MLXAudioSTT

/// Presents an Omni live session through the native streaming session seam.
///
/// The adapter leases the runtime for the session's lifetime, so retiring the
/// runtime waits for the session to end instead of cutting it off.
nonisolated final class OmniNativeStreamingSession: MLXNativeStreamingSession, @unchecked Sendable {
    let events: AsyncStream<TranscriptionEvent>
    private let session: OmniRealtimeTranscriptionSession
    private let forwarding: Task<Void, Never>
    private let diagnosticSessionID: UUID?
    private let diagnosticRequestID = UUID()
    private let diagnosticRuntimeID: UUID
    private let diagnosticLock = NSLock()
    private var receivedAudio = false

    /// Qwen3-ASR preview over the realtime transcription socket.
    static func qwen(runtime: OmniASRRuntime, language: String?, sessionID: UUID? = nil) async throws -> OmniNativeStreamingSession {
        let endpoint = try await runtime.beginUse()
        let session = OmniRealtimeTranscriptionSession(endpoint: endpoint, language: language)
        return OmniNativeStreamingSession(session: session, runtime: runtime, sessionID: sessionID)
    }

    /// MOSS-Transcribe-Diarize windows over the realtime transcription socket,
    /// with the task prompt Voxt's Swift live session used.
    static func moss(runtime: OmniASRRuntime, prompt: String?, sessionID: UUID? = nil) async throws -> OmniNativeStreamingSession {
        let endpoint = try await runtime.beginUse()
        let session = OmniRealtimeTranscriptionSession(
            endpoint: endpoint,
            language: nil,
            prompt: prompt,
            joining: .lines
        )
        return OmniNativeStreamingSession(session: session, runtime: runtime, sessionID: sessionID)
    }

    /// Nemotron 3.5 ASR's cache-aware stream over the realtime transcription
    /// socket, with the live latency and prompt language Voxt's Swift session used.
    static func nemotron(
        runtime: OmniASRRuntime,
        language: String,
        chunkMilliseconds: Int,
        sessionID: UUID? = nil
    ) async throws -> OmniNativeStreamingSession {
        let endpoint = try await runtime.beginUse()
        let session = OmniRealtimeTranscriptionSession(
            endpoint: endpoint,
            language: language,
            chunkMilliseconds: chunkMilliseconds
        )
        return OmniNativeStreamingSession(session: session, runtime: runtime, sessionID: sessionID)
    }

    private init(session: OmniRealtimeTranscriptionSession, runtime: OmniASRRuntime, sessionID: UUID?) {
        self.session = session
        diagnosticSessionID = sessionID
        diagnosticRuntimeID = runtime.diagnosticRuntimeID
        let requestID = diagnosticRequestID
        let runtimeID = diagnosticRuntimeID
        RecordingDiagnostics.shared.record(.init(.requestStarted, sessionID: sessionID, requestID: requestID, runtimeID: runtimeID))
        let source = session.events
        let (events, continuation) = AsyncStream.makeStream(of: TranscriptionEvent.self)
        self.events = events
        forwarding = Task.detached {
            var sawPartial = false
            var completed = false
            for await event in source {
                switch event {
                case .display(let confirmedText, let provisionalText):
                    if !sawPartial, !confirmedText.isEmpty || !provisionalText.isEmpty {
                        sawPartial = true
                        RecordingDiagnostics.shared.record(.init(.requestFirstPartial, sessionID: sessionID, requestID: requestID, runtimeID: runtimeID))
                    }
                    continuation.yield(.displayUpdate(confirmedText: confirmedText, provisionalText: provisionalText))
                case .ended(let text, let segments):
                    completed = true
                    RecordingDiagnostics.shared.record(.init(.requestCompleted, sessionID: sessionID, requestID: requestID, runtimeID: runtimeID,
                        outcome: text.isEmpty ? .empty : .success, count: text.count))
                    continuation.yield(.ended(STTOutput(
                        text: text,
                        segments: segments.isEmpty ? nil : segments.map(\.transcriptSegment)
                    )))
                case .failed(let message):
                    completed = true
                    RecordingDiagnostics.shared.record(.init(.requestCompleted, sessionID: sessionID, requestID: requestID, runtimeID: runtimeID,
                        outcome: .failed, failure: .runtime))
                    continuation.yield(.failed(StreamingFailure(message: message)))
                }
            }
            if !completed {
                RecordingDiagnostics.shared.record(.init(.requestCompleted, sessionID: sessionID, requestID: requestID, runtimeID: runtimeID, outcome: .cancelled))
            }
            continuation.finish()
            await runtime.endUse()
            RecordingDiagnostics.shared.record(.init(.requestCleanup, sessionID: sessionID, requestID: requestID, runtimeID: runtimeID))
        }
    }

    func feedAudio(samples: [Float]) {
        diagnosticLock.lock()
        let firstAudio = !receivedAudio && !samples.isEmpty
        if firstAudio { receivedAudio = true }
        diagnosticLock.unlock()
        if firstAudio {
            RecordingDiagnostics.shared.record(.init(.asrReceivingAudio, sessionID: diagnosticSessionID,
                requestID: diagnosticRequestID, runtimeID: diagnosticRuntimeID, sampleCount: samples.count))
        }
        session.feedAudio(samples: samples)
    }

    func stop() {
        session.stop()
    }

    func cancel() {
        session.cancel()
    }
}

nonisolated extension OmniSpeakerSegment {
    /// The MLXAudio segment Voxt's MOSS handling reads; no speaker means none.
    var transcriptSegment: STTTranscriptSegment {
        STTTranscriptSegment(
            text: text,
            startTime: startSeconds,
            endTime: endSeconds,
            speakerID: speakerID.isEmpty ? nil : speakerID
        )
    }
}
