import XCTest
@testable import Voxt

@MainActor
final class SpeechDiagnosticCompletionTests: XCTestCase {
    func testTransportErrorPreservesPartialTextWithoutReportingSuccess() throws {
        try withTranscriber { transcriber, recorder, output in
            transcriber.sessionReportsPartialResultsOverride = false
            let privateText = "synthetic transcript https://private.invalid/?token=secret /Users/test/private"
            transcriber.handleRecognitionUpdate(text: privateText, isFinal: false, error: nil,
                generationID: transcriber.recognitionGenerationID)
            var deliveredText: String?
            transcriber.onTranscriptionFinished = { deliveredText = $0 }
            transcriber.handleRecognitionUpdate(text: nil, isFinal: false,
                error: NSError(domain: NSURLErrorDomain, code: NSURLErrorNetworkConnectionLost,
                    userInfo: [NSLocalizedDescriptionKey: privateText]), generationID: transcriber.recognitionGenerationID)
            recorder.record(.init(.asrCompleted, sessionID: transcriber.diagnosticSessionID, outcome: .success))
            recorder.record(.init(.sessionFinished, sessionID: transcriber.diagnosticSessionID, outcome: .success))
            let completion = try XCTUnwrap(output.events().first { $0.kind == .asrCompleted })
            XCTAssertEqual(completion.outcome, .failed)
            XCTAssertEqual(completion.failure, .transport)
            XCTAssertEqual(deliveredText, privateText)
            XCTAssertTrue(transcriber.transcribedText.isEmpty)
            XCTAssertEqual(try output.events().filter { $0.kind == .asrCompleted }.count, 1)
            XCTAssertEqual(try output.events().last?.outcome, .failed)
            XCTAssertFalse(output.lines.joined().contains("private"))
            XCTAssertFalse(output.lines.joined().contains("secret"))
        }
    }

    func testErrorWithoutTextIsFailureRatherThanEmpty() throws {
        try withTranscriber { transcriber, _, output in
            transcriber.handleRecognitionUpdate(text: nil, isFinal: false, error: URLError(.timedOut),
                generationID: transcriber.recognitionGenerationID)
            let completion = try XCTUnwrap(output.events().last)
            XCTAssertEqual(completion.outcome, .failed)
            XCTAssertEqual(completion.failure, .timeout)
        }
    }

    func testFinalizationTimeoutKeepsPartialAndTerminatesOnlyOnce() throws {
        try withTranscriber { transcriber, _, output in
            let generationID = transcriber.recognitionGenerationID
            transcriber.handleRecognitionUpdate(text: "partial", isFinal: false, error: nil, generationID: generationID)
            var delivered: [String] = []
            transcriber.onTranscriptionFinished = { delivered.append($0) }
            transcriber.forceFinalizeIfNeeded()
            transcriber.forceFinalizeIfNeeded()
            transcriber.handleRecognitionUpdate(text: "late final", isFinal: true, error: nil, generationID: generationID)
            XCTAssertEqual(delivered, ["partial"])
            let completions = try output.events().filter { $0.kind == .asrCompleted }
            XCTAssertEqual(completions.count, 1)
            XCTAssertEqual(completions.first?.failure, .timeout)
        }
    }

    func testCancellationDoesNotDeliverSalvagedText() throws {
        try withTranscriber { transcriber, _, output in
            let generationID = transcriber.recognitionGenerationID
            transcriber.handleRecognitionUpdate(text: "private partial", isFinal: false, error: nil, generationID: generationID)
            var delivered: [String] = []
            transcriber.onTranscriptionFinished = { delivered.append($0) }
            transcriber.handleRecognitionUpdate(text: nil, isFinal: false,
                error: NSError(domain: "kAFAssistantErrorDomain", code: 216), generationID: generationID)
            XCTAssertEqual(delivered, [""])
            XCTAssertEqual(try output.events().last?.outcome, .cancelled)
            XCTAssertEqual(try output.events().last?.failure, .cancelled)
        }
    }

    func testNoSpeechIsEmptyAndSuccessfulFinalIsSuccess() throws {
        try withTranscriber { transcriber, _, output in
            transcriber.handleRecognitionUpdate(text: nil, isFinal: false,
                error: NSError(domain: "kAFAssistantErrorDomain", code: 1110), generationID: transcriber.recognitionGenerationID)
            XCTAssertEqual(try output.events().last?.outcome, .empty)
            XCTAssertEqual(try output.events().last?.failure, RecordingDiagnosticEvent.Failure.none)
        }
        try withTranscriber { transcriber, _, output in
            transcriber.sessionReportsPartialResultsOverride = false
            transcriber.handleRecognitionUpdate(text: "final", isFinal: true, error: nil, generationID: transcriber.recognitionGenerationID)
            XCTAssertEqual(try output.events().last?.outcome, .success)
            XCTAssertEqual(transcriber.transcribedText, "final")
        }
    }

    func testStaleRecognitionGenerationCannotFinishCurrentSession() throws {
        try withTranscriber { transcriber, _, output in
            let before = output.lines.count
            transcriber.handleRecognitionUpdate(text: "stale", isFinal: true, error: URLError(.timedOut), generationID: UUID())
            XCTAssertEqual(output.lines.count, before)
            XCTAssertTrue(transcriber.transcribedText.isEmpty)
        }
    }

    private func withTranscriber(
        _ assertion: (SpeechTranscriber, RecordingDiagnosticRecorder, SpeechDiagnosticOutput) throws -> Void
    ) throws {
        let suiteName = "Voxt.SpeechDiagnostics.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let output = SpeechDiagnosticOutput()
        let recorder = RecordingDiagnosticRecorder(defaults: defaults, sink: { output.append($0) })
        recorder.setEnabled(true)
        let transcriber = SpeechTranscriber(diagnostics: recorder, initializeRecognizer: false)
        transcriber.diagnosticSessionID = UUID()
        recorder.record(.init(.sessionStarted, sessionID: transcriber.diagnosticSessionID))
        try assertion(transcriber, recorder, output)
    }
}

private nonisolated final class SpeechDiagnosticOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedLines: [String] = []
    var lines: [String] { lock.lock(); defer { lock.unlock() }; return recordedLines }
    func append(_ line: String) { lock.lock(); recordedLines.append(line); lock.unlock() }
    func events() throws -> [RecordingDiagnosticEvent] {
        try lines.map { try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data($0.utf8)) }
    }
}
