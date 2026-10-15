import XCTest
@testable import Voxt

@MainActor
final class RemoteASRCompletionTests: XCTestCase {
    func testProviderFailureRecoversPartialWithoutReportingSuccessAsFailure() async {
        let transcriber = RemoteASRTranscriber()
        var failures = 0
        transcriber.onRuntimeFailure = { _ in failures += 1 }
        let text = await transcriber.resolveStreamingResult(
            warningMessage: "test",
            waitForFinal: { throw URLError(.networkConnectionLost) },
            fallback: { "partial result" }
        )
        XCTAssertEqual(text, "partial result")
        XCTAssertEqual(failures, 0)
    }

    func testFailureBeforeAnyTextReportsRuntimeFailure() async {
        let transcriber = RemoteASRTranscriber()
        var failures = 0
        transcriber.onRuntimeFailure = { _ in failures += 1 }
        let text = await transcriber.resolveStreamingResult(
            warningMessage: "test",
            waitForFinal: { throw URLError(.timedOut) },
            fallback: { "" }
        )
        XCTAssertTrue(text.isEmpty)
        XCTAssertEqual(failures, 1)
    }

    func testCancellationDoesNotRecoverOrDeliverPartialText() async {
        let transcriber = RemoteASRTranscriber()
        var failures = 0
        var delivered = 0
        transcriber.onRuntimeFailure = { _ in failures += 1 }
        transcriber.onTranscriptionFinished = { _ in delivered += 1 }
        let generation = transcriber.recordingGenerationID
        let task = Task {
            let text = await transcriber.resolveStreamingResult(
                warningMessage: "test",
                waitForFinal: { try Task.checkCancellation(); return "unexpected" },
                fallback: { "must not be delivered" }
            )
            transcriber.finish(with: text, generationID: generation)
            return text
        }
        task.cancel()
        let result = await task.value
        XCTAssertTrue(result.isEmpty)
        XCTAssertEqual(failures, 0)
        XCTAssertEqual(delivered, 0)
    }

    func testLateFailureCannotReportIntoNewGeneration() {
        let transcriber = RemoteASRTranscriber()
        let oldGeneration = transcriber.recordingGenerationID
        transcriber.discardPendingSessionOutput()
        var failures = 0
        transcriber.onRuntimeFailure = { _ in failures += 1 }
        transcriber.notifyRuntimeFailure(URLError(.timedOut), generationID: oldGeneration)
        XCTAssertEqual(failures, 0)
        transcriber.notifyRuntimeFailure(URLError(.timedOut), generationID: transcriber.recordingGenerationID)
        XCTAssertEqual(failures, 1)
    }

    func testLateCompletionCannotDeliverIntoNewGeneration() {
        // No capture is prepared here. Completing a cold instance must not
        // create an input node or initialize microphone hardware during cleanup.
        let transcriber = RemoteASRTranscriber()
        let oldGeneration = transcriber.recordingGenerationID
        transcriber.discardPendingSessionOutput()
        var delivered: [String] = []
        transcriber.onTranscriptionFinished = { delivered.append($0) }
        transcriber.finish(with: "stale", generationID: oldGeneration)
        XCTAssertTrue(delivered.isEmpty)
        transcriber.finish(with: "current", generationID: transcriber.recordingGenerationID)
        XCTAssertEqual(delivered, ["current"])
    }

    func testStepFunProviderDistinguishesDeltaFromFinalWithDisplayDisabled() async throws {
        let suiteName = "RemoteASR.StepFunDiagnostics.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let output = RemotePartialDiagnosticOutput()
        let diagnostics = RecordingDiagnosticRecorder(defaults: defaults, sink: { output.append($0) })
        diagnostics.setEnabled(true)
        let transcriber = RemoteASRTranscriber()
        transcriber.diagnosticRecorderOverride = diagnostics
        let session = URLSession(configuration: .ephemeral)
        defer { session.invalidateAndCancel() }
        let socket = session.webSocketTask(with: URL(string: "wss://example.invalid/asr")!)
        let finalContext = StepFunStreamingContext(session: session, ws: socket,
            responseState: StepFunResponseState(), generationID: transcriber.recordingGenerationID)
        transcriber.diagnosticSessionID = UUID()
        defer { transcriber.resetIntermediateTranscriptionPublishing() }

        await transcriber.handleStepFunRealtimeMessage(
            #"{"type":"conversation.item.input_audio_transcription.completed","transcript":"final-only"}"#,
            context: finalContext)
        XCTAssertTrue(output.lines.isEmpty)
        XCTAssertEqual(transcriber.pendingIntermediateTranscription, "final-only")
        transcriber.resetIntermediateTranscriptionPublishing()
        transcriber.recordingGenerationID = UUID()
        transcriber.diagnosticSessionID = UUID()
        transcriber.sessionAllowsRealtimeTextDisplay = false
        let partialContext = StepFunStreamingContext(session: session, ws: socket,
            responseState: StepFunResponseState(), generationID: transcriber.recordingGenerationID)
        await transcriber.handleStepFunRealtimeMessage(
            #"{"type":"conversation.item.input_audio_transcription.delta","delta":"synthetic private"}"#,
            context: partialContext)
        await transcriber.handleStepFunRealtimeMessage(
            #"{"type":"conversation.item.input_audio_transcription.delta","delta":" response"}"#,
            context: partialContext)
        await transcriber.handleStepFunRealtimeMessage(
            #"{"type":"conversation.item.input_audio_transcription.completed","transcript":"final"}"#,
            context: partialContext)

        XCTAssertEqual(output.lines.count, 1)
        let line = try XCTUnwrap(output.lines.first)
        let event = try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data(line.utf8))
        XCTAssertEqual(event.kind, .asrFirstPartial)
        XCTAssertEqual(event.sessionID, transcriber.diagnosticSessionID)
        XCTAssertEqual(event.requestID, partialContext.generationID)
        XCTAssertTrue(transcriber.transcribedText.isEmpty)
        XCTAssertFalse(line.contains("private"))
        XCTAssertEqual(socket.state, .suspended)

        transcriber.recordingGenerationID = UUID()
        transcriber.diagnosticSessionID = UUID()
        await transcriber.handleStepFunRealtimeMessage(
            #"{"type":"conversation.item.input_audio_transcription.delta","delta":"stale"}"#,
            context: partialContext)
        XCTAssertEqual(output.lines.count, 1)
    }

    func testRemotePartialDiagnosticsIgnoreDisplayPreferenceAndDeduplicate() throws {
        let suiteName = "RemoteASR.PartialTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let output = RemotePartialDiagnosticOutput()
        let diagnostics = RecordingDiagnosticRecorder(defaults: defaults, sink: { output.append($0) })
        diagnostics.setEnabled(true)
        let transcriber = RemoteASRTranscriber()
        transcriber.diagnosticRecorderOverride = diagnostics
        let sessionID = UUID()
        transcriber.diagnosticSessionID = sessionID
        transcriber.sessionAllowsRealtimeTextDisplay = false
        let generationID = transcriber.recordingGenerationID
        let sensitiveText = "synthetic transcript https://private.example/?token=secret /Users/test/private"

        transcriber.publishIntermediateTranscription(sensitiveText, isPartial: true,
            generationID: generationID)
        transcriber.publishIntermediateTranscription("another partial", isPartial: true,
            generationID: generationID)
        transcriber.publishIntermediateTranscription("final result", isPartial: false,
            generationID: generationID)

        let events = try output.lines.map { try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data($0.utf8)) }
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events.first?.kind, .asrFirstPartial)
        XCTAssertEqual(events.first?.sessionID, sessionID)
        XCTAssertEqual(events.first?.requestID, generationID)
        XCTAssertEqual(events.first?.backend, .remote)
        XCTAssertEqual(events.first?.count, sensitiveText.count)
        XCTAssertTrue(transcriber.transcribedText.isEmpty)
        XCTAssertNil(transcriber.pendingIntermediateTranscription)
        XCTAssertFalse(output.lines.joined().contains("private"))
        XCTAssertFalse(output.lines.joined().contains("secret"))
    }

    func testFinalOnlyRemoteResponseDoesNotCreatePartialDiagnostic() {
        let suiteName = "RemoteASR.FinalTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let output = RemotePartialDiagnosticOutput()
        let diagnostics = RecordingDiagnosticRecorder(defaults: defaults, sink: { output.append($0) })
        diagnostics.setEnabled(true)
        let transcriber = RemoteASRTranscriber()
        transcriber.diagnosticRecorderOverride = diagnostics
        transcriber.diagnosticSessionID = UUID()
        defer { transcriber.resetIntermediateTranscriptionPublishing() }

        transcriber.publishIntermediateTranscription("final result", isPartial: false,
            generationID: transcriber.recordingGenerationID)

        XCTAssertTrue(output.lines.isEmpty)
        XCTAssertEqual(transcriber.pendingIntermediateTranscription, "final result")
    }

    func testCancelledTaskCannotPublishRemotePartial() async {
        let suiteName = "RemoteASR.CancelledPartialTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let output = RemotePartialDiagnosticOutput()
        let diagnostics = RecordingDiagnosticRecorder(defaults: defaults, sink: { output.append($0) })
        diagnostics.setEnabled(true)
        let transcriber = RemoteASRTranscriber()
        transcriber.diagnosticRecorderOverride = diagnostics
        transcriber.diagnosticSessionID = UUID()
        let task = Task { @MainActor in
            transcriber.publishIntermediateTranscription("cancelled", isPartial: true,
                generationID: transcriber.recordingGenerationID)
        }
        task.cancel()
        await task.value

        XCTAssertTrue(output.lines.isEmpty)
        XCTAssertNil(transcriber.pendingIntermediateTranscription)
        XCTAssertTrue(transcriber.transcribedText.isEmpty)
    }

    func testLateRemotePartialCannotBeAttributedToNewSession() throws {
        let suiteName = "RemoteASR.StalePartialTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let output = RemotePartialDiagnosticOutput()
        let diagnostics = RecordingDiagnosticRecorder(defaults: defaults, sink: { output.append($0) })
        diagnostics.setEnabled(true)
        let transcriber = RemoteASRTranscriber()
        transcriber.diagnosticRecorderOverride = diagnostics
        let oldGeneration = transcriber.recordingGenerationID
        transcriber.recordingGenerationID = UUID()
        transcriber.diagnosticSessionID = UUID()
        transcriber.sessionAllowsRealtimeTextDisplay = false

        transcriber.publishIntermediateTranscription("stale", isPartial: true,
            generationID: oldGeneration)
        XCTAssertTrue(output.lines.isEmpty)
        transcriber.publishIntermediateTranscription("current", isPartial: true,
            generationID: transcriber.recordingGenerationID)
        let line = try XCTUnwrap(output.lines.first)
        let event = try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data(line.utf8))
        XCTAssertEqual(event.sessionID, transcriber.diagnosticSessionID)
        XCTAssertEqual(event.requestID, transcriber.recordingGenerationID)
    }
}

nonisolated private final class RemotePartialDiagnosticOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []
    var lines: [String] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
    func append(_ line: String) {
        lock.lock()
        storage.append(line)
        lock.unlock()
    }
}
