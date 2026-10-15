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
        print("PASS 6 real swift-log handler privacy assertions")
    }
}

nonisolated final class HandlerTestOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func append(_ line: String) { lock.lock(); lines.append(line); lock.unlock() }
    func snapshot() -> [String] { lock.lock(); defer { lock.unlock() }; return lines }
}
