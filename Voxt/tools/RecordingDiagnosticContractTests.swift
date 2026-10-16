import Foundation

@main
struct RecordingDiagnosticContractTests {
    static func main() throws {
        let suiteName = "Voxt.Diagnostics.Tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let clock = TestClock()
        let output = TestOutput()
        let recorder = RecordingDiagnosticRecorder(defaults: defaults, monotonicMilliseconds: { clock.read() }, sink: { try output.append($0) })
        var tests = 0
        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            precondition(condition(), message)
            tests += 1
        }
        var evaluated = false
        recorder.record({ evaluated = true; return RecordingDiagnosticEvent(.sessionStarted) }())
        check(!evaluated && output.lines.isEmpty, "disabled diagnostics must not evaluate the event")
        recorder.setEnabled(true)
        let restored = RecordingDiagnosticRecorder(defaults: defaults, sink: { try output.append($0) })
        check(restored.isEnabled, "enablement must persist")
        let first = UUID()
        let second = UUID()
        recorder.record(.init(.sessionStarted, sessionID: first))
        clock.advance(16 * 60 * 1_000)
        recorder.record(.init(.sessionStarted, sessionID: second))
        recorder.record(.init(.asrFirstPartial, sessionID: first))
        recorder.record(.init(.asrFirstPartial, sessionID: first))
        recorder.record(.init(.asrCompleted, sessionID: first, outcome: .empty))
        recorder.record(.init(.vadFallback, sessionID: second, outcome: .fallback, failure: .timeout))
        recorder.record(.init(.sessionFinished, sessionID: first, outcome: .success))
        recorder.record(.init(.sessionFinished, sessionID: first, outcome: .success))
        recorder.record(.init(.sessionCancelled, sessionID: second, outcome: .cancelled))
        recorder.record(.init(.sessionFinished, sessionID: second, outcome: .success))
        recorder.record(.init(.sessionCleanup, sessionID: second))
        let terminalCount = output.lines.count
        recorder.record(.init(.asrFirstPartial, sessionID: second))
        recorder.record(.init(.asrCompleted, sessionID: second))
        recorder.record(.init(.inputReady, sessionID: second))
        recorder.record(.init(.captureSummary, sessionID: second, count: 9))
        recorder.record(.init(.enhancementCompleted, sessionID: second, outcome: .success))
        recorder.record(.init(.sessionCleanup, sessionID: second))
        check(output.lines.count == terminalCount, "late callbacks after cancellation and cleanup must not revive a session")
        let enhancementSession = UUID()
        recorder.record(.init(.sessionStarted, sessionID: enhancementSession))
        recorder.record(.init(.asrCompleted, sessionID: enhancementSession, outcome: .success))
        recorder.record(.init(.enhancementCompleted, sessionID: enhancementSession, outcome: .failed, failure: .transport))
        recorder.record(.init(.sessionFinished, sessionID: enhancementSession, outcome: .success))
        let enhancementTerminal = try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data(output.lines.last!.utf8))
        check(enhancementTerminal.outcome == .failed && enhancementTerminal.failure == .transport,
              "session completion must preserve an enhancement failure after successful ASR")
        recorder.record(.init(.sessionCleanup, sessionID: enhancementSession))
        for _ in 0..<300 {
            let completedSession = UUID()
            recorder.record(.init(.sessionStarted, sessionID: completedSession))
            recorder.record(.init(.sessionFinished, sessionID: completedSession))
            recorder.record(.init(.sessionCleanup, sessionID: completedSession))
        }
        let beforeOldCallback = output.lines.count
        recorder.record(.init(.asrFirstPartial, sessionID: second))
        recorder.record(.init(.inputReady, sessionID: second))
        check(output.lines.count == beforeOldCallback, "completed sessions must remain closed beyond 256 recordings")
        let events = try output.lines.map { try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data($0.utf8)) }
        check(events.filter { $0.kind == .asrFirstPartial }.count == 1, "first partial must be recorded once")
        check(events.filter { $0.kind == .sessionFinished && $0.sessionID == first }.count == 1, "session finish must be recorded once")
        check(events.first { $0.kind == .asrFirstPartial }?.elapsedMilliseconds == 960_000, "elapsed time must use injected monotonic clock")
        check(events.first { $0.kind == .vadFallback }?.sessionID == second, "concurrent sessions must remain separate")
        check(recorder.isEnabled, "diagnostics must stay enabled beyond fifteen minutes")
        check(events.contains { $0.kind == .sessionCleanup && $0.sessionID == second }, "cleanup must survive cancellation")
        let sensitive = "private transcript https://private.example/?token=secret /Users/test/private.wav sk-secret"
        check(RecordingDiagnosticEvent.safeLine(sensitive) == nil, "free text must be rejected")
        let metadata = RecordingDiagnosticExportMetadata(macOSMajorVersion: 26, macOSMinorVersion: 0,
            macOSPatchVersion: 0, recordingDiagnosticsEnabled: true, diagnosticWriteFailed: false)
        let metadataKeys = try JSONSerialization.jsonObject(with: Data(metadata.encodedText().utf8)) as! [String: Any]
        check(Set(metadataKeys.keys) == Set(["schemaVersion", "macOSMajorVersion", "macOSMinorVersion", "macOSPatchVersion", "recordingDiagnosticsEnabled", "diagnosticWriteFailed"]), "export metadata must have only allowed fields")
        let valid = output.lines[0].dropLast() + ",\"transcript\":\"\(sensitive)\",\"endpoint\":\"private\"}"
        let canonical = RecordingDiagnosticEvent.safeLine(String(valid))
        check(canonical != nil && !canonical!.contains("private"), "unknown JSON fields must not reach persisted diagnostics")
        check(RecordingDiagnosticEvent.safeLine(String(repeating: "x", count: 100_000)) == nil, "oversized events must be rejected")
        output.shouldFail = true
        recorder.record(.init(.runtimeStarted, runtimeID: UUID()))
        check(recorder.failedWriteCount == 1, "sink failures must be visible")
        output.shouldFail = false
        recorder.record(.init(.runtimeReady, runtimeID: UUID()))
        check(output.lines.last?.contains("runtimeReady") == true, "recording must continue after a failed write")
        let error = NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut,
            userInfo: [NSLocalizedDescriptionKey: sensitive, NSURLErrorFailingURLErrorKey: URL(string: "https://private.example")!])
        let classification = RecordingDiagnosticEvent.Failure.classify(error)
        check(classification == .timeout, "errors must be classified without rendering their private description")
        let classLine = try RecordingDiagnosticEvent(.requestCompleted, failure: classification).encodedLine()
        check(!classLine.contains("private"), "error descriptions must not enter diagnostics")
        let beforeConcurrent = output.lines.count
        DispatchQueue.concurrentPerform(iterations: 20) { _ in
            let session = UUID()
            recorder.record(.init(.sessionStarted, sessionID: session))
            recorder.record(.init(.asrFirstPartial, sessionID: session))
            recorder.record(.init(.asrCompleted, sessionID: session, outcome: .success))
            recorder.record(.init(.sessionFinished, sessionID: session, outcome: .success))
            recorder.record(.init(.sessionCleanup, sessionID: session))
        }
        check(output.lines.count - beforeConcurrent == 100, "concurrent sessions must retain all independent stages")
        recorder.setEnabled(false)
        let count = output.lines.count
        recorder.record(.init(.runtimeReleased))
        check(output.lines.count == count, "disable must stop collection")

        let store = VoxtLogFileStore(maxFileBytes: 512, retainedArchiveCount: 2, baseDirectoryURL: scratch)
        for _ in 0..<20 { store.append(String(repeating: "a", count: 200)) }
        store.flush()
        check(!store.latestLines(limit: 100).isEmpty, "rotation must retain recent logs")
        store.append("last-event")
        store.flush()
        check(store.latestLines(limit: 1) == ["last-event"], "logging must continue after rotation")
        store.append(String(repeating: "a", count: 5_000))
        store.flush()
        let size = try FileManager.default.attributesOfItem(atPath: store.currentLogURL.path)[.size] as! NSNumber
        check(size.uint64Value <= 512, "one oversized line must not break file bound")
        let invalidRoot = scratch.appendingPathComponent("not-a-directory")
        try Data().write(to: invalidRoot)
        let failingStore = VoxtLogFileStore(baseDirectoryURL: invalidRoot)
        failingStore.append("test")
        failingStore.flush()
        check(failingStore.lastWriteFailed, "file failures must be observable")
        try FileManager.default.removeItem(at: invalidRoot)
        failingStore.append("recovered")
        failingStore.flush()
        check(!failingStore.lastWriteFailed && failingStore.latestLines(limit: 1) == ["recovered"], "storage must recover after a temporary directory failure")
        let exportStore = VoxtLogFileStore(baseDirectoryURL: scratch.appendingPathComponent("export"))
        exportStore.append(sensitive)
        exportStore.append(String(valid))
        exportStore.flush()
        let exported = VoxtLogExportStore.latestLogDisplayText(store: exportStore, metadataProvider: { metadata })
        check(exported.contains("sessionStarted"), "actual viewer/export must retain safe typed events")
        check(!exported.contains("private") && !exported.contains("sk-secret"), "actual viewer/export must filter legacy private lines and unknown fields")
        print("PASS \(tests) recording diagnostic contract assertions")
    }
}

nonisolated enum AppLocalization {
    static func localizedString(_ value: String) -> String { value }
}

nonisolated final class TestClock: @unchecked Sendable {
    private let lock = NSLock()
    private var milliseconds: UInt64 = 0
    func read() -> UInt64 { lock.lock(); defer { lock.unlock() }; return milliseconds }
    func advance(_ amount: UInt64) { lock.lock(); milliseconds += amount; lock.unlock() }
}

nonisolated final class TestOutput: @unchecked Sendable {
    var lines: [String] = []
    var shouldFail = false
    func append(_ line: String) throws {
        if shouldFail { throw CocoaError(.fileWriteOutOfSpace) }
        lines.append(line)
    }
}
