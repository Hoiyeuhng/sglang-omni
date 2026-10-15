import Foundation

@main
struct RecordingDiagnosticRemoteASRTests {
    @MainActor
    static func main() async throws {
        let suiteName = "Voxt.RemotePartialContract.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let output = RemoteDiagnosticOutput()
        let recorder = RecordingDiagnosticRecorder(defaults: defaults, sink: { output.append($0) })
        recorder.setEnabled(true)
        let transcriber = RemoteASRTranscriber()
        transcriber.diagnosticRecorderOverride = recorder
        transcriber.diagnosticSessionID = UUID()
        transcriber.sessionAllowsRealtimeTextDisplay = false
        let generationID = transcriber.recordingGenerationID
        var assertions = 0
        func check(_ condition: @autoclosure () -> Bool, _ description: String) {
            precondition(condition(), description)
            assertions += 1
        }
        transcriber.publishIntermediateTranscription("synthetic private transcript https://example.invalid/?token=secret", isPartial: true, generationID: generationID)
        transcriber.publishIntermediateTranscription("second partial", isPartial: true, generationID: generationID)
        transcriber.publishIntermediateTranscription("final result", isPartial: false, generationID: generationID)
        let first = try output.events()
        check(first.count == 1, "UI-off partials must be recorded exactly once")
        check(first[0].kind == .asrFirstPartial, "the recorded event must be a partial")
        check(first[0].sessionID == transcriber.diagnosticSessionID && first[0].requestID == generationID,
              "the partial must retain its session and request generation")
        check(first[0].backend == .remote, "the backend must be remote")
        check(!output.lines.joined().contains("private") && !output.lines.joined().contains("secret"),
              "transcript, URL and credentials must not enter diagnostics")
        check(transcriber.transcribedText.isEmpty && transcriber.pendingIntermediateTranscription == nil,
              "disabled realtime display must remain disabled")

        transcriber.diagnosticSessionID = UUID()
        transcriber.recordingGenerationID = UUID()
        transcriber.sessionAllowsRealtimeTextDisplay = true
        transcriber.publishIntermediateTranscription("final-only result", isPartial: false,
            generationID: transcriber.recordingGenerationID)
        check(output.lines.count == 1, "a final-only response must not be recorded as partial")
        await transcriber.intermediateTranscriptionPublishTask?.value
        check(transcriber.transcribedText == "final-only result", "final responses must retain UI publishing behavior")
        transcriber.publishIntermediateTranscription("stale partial", isPartial: true, generationID: generationID)
        check(output.lines.count == 1 && transcriber.pendingIntermediateTranscription == nil,
              "a late request cannot write into a newer session")
        transcriber.publishIntermediateTranscription(" \n ", isPartial: true, generationID: transcriber.recordingGenerationID)
        check(output.lines.count == 1, "empty interim responses must not count as first partial")
        transcriber.publishIntermediateTranscription("current partial", isPartial: true,
            generationID: transcriber.recordingGenerationID)
        let current = try output.events().last!
        check(current.sessionID == transcriber.diagnosticSessionID && current.requestID == transcriber.recordingGenerationID,
              "a new session must retain independent first-partial state")
        transcriber.resetIntermediateTranscriptionPublishing()
        recorder.record(.init(.sessionCancelled, sessionID: transcriber.diagnosticSessionID))
        let countAfterCancellation = output.lines.count
        transcriber.publishIntermediateTranscription("late cancelled partial", isPartial: true,
            generationID: transcriber.recordingGenerationID)
        check(output.lines.count == countAfterCancellation, "ended sessions must not be revived by partials")
        transcriber.resetIntermediateTranscriptionPublishing()
        transcriber.diagnosticSessionID = UUID()
        let cancelledTask = Task { @MainActor in
            transcriber.publishIntermediateTranscription("cancelled task", isPartial: true,
                generationID: transcriber.recordingGenerationID)
        }
        cancelledTask.cancel()
        await cancelledTask.value
        check(output.lines.count == countAfterCancellation && transcriber.pendingIntermediateTranscription == nil,
              "cancelled tasks must not publish or record a partial")
        transcriber.diagnosticSessionID = UUID()
        transcriber.recordDiagnosticPartial("file socket interim", generationID: transcriber.recordingGenerationID)
        check(output.lines.count == countAfterCancellation + 1, "file socket partials must not require UI publishing")
        check(transcriber.pendingIntermediateTranscription == nil, "diagnostic-only file responses must not change UI publishing")
        let countBeforeDisable = output.lines.count
        recorder.setEnabled(false)
        transcriber.diagnosticSessionID = UUID()
        transcriber.publishIntermediateTranscription("disabled", isPartial: true, generationID: transcriber.recordingGenerationID)
        check(output.lines.count == countBeforeDisable, "disabled diagnostics must not emit partials")
        transcriber.resetIntermediateTranscriptionPublishing()

        for payload in [
            #"{"type":"transcript.text.delta","delta":"partial"}"#,
            #"{"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}"#
        ] {
            check(RemoteASRTextSupport.isPartialStreamPayload(fromLine: payload), "explicit streamed deltas must be partial")
        }
        for payload in [
            #"{"type":"transcript.text.done","text":"final"}"#,
            #"{"text":"final"}"#,
            #"{"delta":"final","is_final":true}"#,
            #"{"choices":[{"delta":{"content":"final"},"finish_reason":"stop"}]}"#,
            "plain final response",
            #"{"delta":""}"#
        ] {
            check(!RemoteASRTextSupport.isPartialStreamPayload(fromLine: payload), "final or unclassified text must not create partial diagnostics")
        }
        print("PASS \(assertions) remote ASR diagnostic assertions (production publisher and payload classifier; capture/network dependencies stubbed)")
    }
}

// Dependency stubs leave the production publisher, parser and recorder unchanged.
@MainActor
final class RemoteASRTranscriber {
    var diagnosticSessionID: UUID?
    var diagnosticRecorderOverride: RecordingDiagnosticRecorder?
    var recordingGenerationID = UUID()
    var sessionAllowsRealtimeTextDisplay = true
    var pendingIntermediateTranscription: String?
    var intermediateTranscriptionPublishTask: Task<Void, Never>?
    var transcribedText = ""
    func isCurrentGeneration(_ generationID: UUID) -> Bool { recordingGenerationID == generationID }
}

enum RecordingDiagnostics {
    static var shared: RecordingDiagnosticRecorder { fatalError("Tests must inject their recorder") }
}

enum RecordingSessionSupport {
    static func textAfterSuppressingPromptEcho(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

enum VoxtLog {
    static func asrWarning(_ text: String) {}
}

struct ResolvedASRHintPayload {
    var language: String?
    var prompt: String?
}

enum RemoteASRProvider {
    case xiaomiMiMoASR, openAIWhisper
    var suggestedModel: String { "fixture" }
}

enum RemoteASRTextSanitizer {
    static func isLikelyIdentifierText(_ text: String) -> Bool { false }
}

nonisolated final class RemoteDiagnosticOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []
    var lines: [String] { lock.lock(); defer { lock.unlock() }; return storage }
    func append(_ line: String) { lock.lock(); storage.append(line); lock.unlock() }
    func events() throws -> [RecordingDiagnosticEvent] {
        try lines.map { try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data($0.utf8)) }
    }
}
