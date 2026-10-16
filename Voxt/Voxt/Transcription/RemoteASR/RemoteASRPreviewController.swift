import Foundation

/// Each polling invocation has its own deduplication state. A cancelled request
/// may finish late, but cannot publish or release a newer invocation's state.
@MainActor
final class RemoteASRPreviewController {
    private let tasks = TrackedTaskStore()
    private var generation = UUID()

    @discardableResult
    func start(
        shouldRun: @escaping @MainActor () -> Bool,
        transcribe: @escaping @MainActor () async throws -> String?,
        publish: @escaping @MainActor (String) -> Void,
        waitForNext: @escaping @MainActor () async throws -> Void = { try await Task.sleep(for: .seconds(1.4)) }
    ) -> Task<Void, Never> {
        cancel()
        let id = generation
        return tasks.start { [weak self] in
            var lastText = ""
            while !Task.isCancelled {
                do { try await waitForNext() } catch { return }
                guard !Task.isCancelled, let self, self.generation == id, shouldRun() else { return }
                let text: String?
                do { text = try await transcribe() } catch { continue }
                guard !Task.isCancelled, self.generation == id, shouldRun() else { return }
                guard let text, !text.isEmpty, text != lastText else { continue }
                lastText = text
                publish(text)
            }
        }
    }

    func cancel() {
        generation = UUID()
        tasks.cancelAll()
    }

    func waitForIdle() async { await tasks.waitForAll() }
}

@MainActor
extension RemoteASRTranscriber {
    func publishIntermediateTranscription(
        _ text: String,
        isPartial: Bool,
        generationID: UUID
    ) {
        guard isCurrentGeneration(generationID), !Task.isCancelled else { return }
        if isPartial {
            recordDiagnosticPartial(text, generationID: generationID)
        }
        guard sessionAllowsRealtimeTextDisplay else { return }
        let visibleText = RecordingSessionSupport.textAfterSuppressingPromptEcho(text)
        guard !visibleText.isEmpty else {
            VoxtLog.asrWarning("Remote ASR intermediate transcription suppressed because it matched prompt guidance.")
            return
        }
        pendingIntermediateTranscription = visibleText
        guard intermediateTranscriptionPublishTask == nil else { return }
        intermediateTranscriptionPublishTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(50))
            guard !Task.isCancelled, let self, self.isCurrentGeneration(generationID) else { return }
            let pending = self.pendingIntermediateTranscription
            self.pendingIntermediateTranscription = nil
            self.intermediateTranscriptionPublishTask = nil
            if let pending {
                self.transcribedText = pending
            }
        }
    }

    func recordDiagnosticPartial(_ text: String, generationID: UUID) {
        guard isCurrentGeneration(generationID), !Task.isCancelled, let diagnosticSessionID else { return }
        let recorder = diagnosticRecorderOverride ?? RecordingDiagnostics.shared
        guard recorder.isEnabled, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        recorder.record(.init(.asrFirstPartial, sessionID: diagnosticSessionID,
                              requestID: generationID, backend: .remote, count: text.count))
    }

    func resetIntermediateTranscriptionPublishing() {
        intermediateTranscriptionPublishTask?.cancel()
        intermediateTranscriptionPublishTask = nil
        pendingIntermediateTranscription = nil
    }
}
