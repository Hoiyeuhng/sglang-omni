import Foundation
import Logging

@main
struct RecordingDiagnosticHandlerTests {
    static func main() throws {
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let fileStore = VoxtLogFileStore(baseDirectoryURL: scratch)
        let output = HandlerTestOutput()
        var file = VoxtFileLogHandler(label: "private label", category: .asr, fileStore: fileStore)
        var oslog = VoxtOSLogHandler(label: "private label", category: .asr, sink: { line, _ in output.append(line) })
        file.metadata = ["apiKey": "private key", "transcript": "private transcript"]
        oslog.metadata = file.metadata
        let secret = "private transcript prompt model-reply /Users/test/private.wav https://private.example/?token=secret cookie=secret"
        let unsafe = LogEvent(level: .error, message: "\(secret)", metadata: ["response": "private model response"], source: secret, file: secret, function: secret, line: 1)
        file.log(event: unsafe)
        oslog.log(event: unsafe)
        fileStore.flush()
        precondition(fileStore.latestLines(limit: 100).isEmpty)
        precondition(output.snapshot().isEmpty)
        let line = try RecordingDiagnosticEvent(.requestCompleted, outcome: .failed, failure: .transport).encodedLine()
        let allowed = LogEvent(level: .error, message: "\(line)", metadata: ["authorization": "private token"], source: secret, file: secret, function: secret, line: 1)
        file.log(event: allowed)
        oslog.log(event: allowed)
        fileStore.flush()
        precondition(fileStore.latestLines(limit: 100) == [line])
        precondition(output.snapshot() == [line])
        let withUnknownFields = line.dropLast() + ",\"transcript\":\"private text\"}"
        let injected = LogEvent(level: .error, message: "\(withUnknownFields)", metadata: nil, source: nil, file: secret, function: secret, line: 1)
        file.log(event: injected)
        oslog.log(event: injected)
        fileStore.flush()
        precondition(!fileStore.latestLines(limit: 100).joined().contains("private"))
        precondition(!output.snapshot().joined().contains("private"))
        let logger = Logger(label: "Voxt.app") { label in
            VoxtMultiplexLogHandler(label: label, category: .app, fileStore: fileStore,
                osLogSink: { line, _ in output.append(line) })
        }
        let app = VoxtLogger(category: .app, logger: logger)
        app.record(.init(.applicationStarted))
        var evaluatedLegacyContent = false
        app.warning({ evaluatedLegacyContent = true; return secret }(), metadata: ["credential": "private"])
        app.info({ evaluatedLegacyContent = true; return secret }())
        VoxtLogger.recordDiagnosticLine(try RecordingDiagnosticEvent(.sessionStarted, sessionID: UUID()).encodedLine(), logger: logger)
        fileStore.flush()
        let persisted = fileStore.latestLines(limit: 100)
        precondition(persisted == output.snapshot())
        precondition(!evaluatedLegacyContent)
        let events = try persisted.map { try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data($0.utf8)) }
        precondition(events.contains { $0.kind == .applicationStarted && $0.category == .app })
        precondition(events.contains { $0.kind == .legacyDiagnostic && $0.severity == .warning && $0.category == .app })
        precondition(events.last?.kind == .sessionStarted)
        precondition(!persisted.joined().contains("private"))
        print("PASS 12 real swift-log handler, legacy-warning and multiplex routing assertions")
    }
}

nonisolated enum VoxtLog { static let verboseEnabled = false }

nonisolated final class HandlerTestOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func append(_ line: String) { lock.lock(); lines.append(line); lock.unlock() }
    func snapshot() -> [String] { lock.lock(); defer { lock.unlock() }; return lines }
}
