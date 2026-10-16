// VoxtLoggerTests.swift
// Provides Voxt Logger Tests for Voxt test coverage.

import XCTest
import Logging
@testable import Voxt

final class VoxtLoggerTests: XCTestCase {
    @MainActor
    func testHotkeyAndModelSwitchesControlTypedEventsWithoutEvaluatingDisabledPayloads() throws {
        let suiteName = "Voxt.TechnicalLogging.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let output = TechnicalLogOutput()
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let fileStore = VoxtLogFileStore(baseDirectoryURL: scratch)
        let logger = Logger(label: "Voxt.model") { label in
            VoxtMultiplexLogHandler(label: label, category: .model, fileStore: fileStore,
                osLogSink: { line, _ in output.append(line) })
        }
        let modelLogger = VoxtLogger(category: .model, logger: logger)
        let hotkeyLogger = VoxtLogger(category: .hotkey, logger: logger)
        var evaluated = false
        VoxtLog.model({ evaluated = true; return .init(.modelRoutingChanged) }(), defaults: defaults, logger: modelLogger)
        VoxtLog.hotkey({ evaluated = true; return .init(.hotkeyPressed) }(), defaults: defaults, logger: hotkeyLogger)
        XCTAssertFalse(evaluated)
        XCTAssertTrue(output.lines.isEmpty)
        defaults.set(true, forKey: AppPreferenceKey.hotkeyDebugLoggingEnabled)
        VoxtLog.hotkey(.init(.hotkeyPressed, action: .transcription), defaults: defaults, logger: hotkeyLogger)
        VoxtLog.model(.init(.modelRoutingChanged), defaults: defaults, logger: modelLogger)
        XCTAssertEqual(output.lines.count, 1)
        defaults.set(true, forKey: AppPreferenceKey.llmDebugLoggingEnabled)
        VoxtLog.model(.init(.modelRoutingChanged), defaults: defaults, logger: modelLogger)
        let events = try output.lines.map { try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data($0.utf8)) }
        XCTAssertEqual(events.map(\.kind), [.hotkeyPressed, .modelRoutingChanged])
        XCTAssertEqual(events.map(\.category), [.hotkey, .model])
        XCTAssertFalse(defaults.bool(forKey: RecordingDiagnosticRecorder.preferenceKey))
    }

    func testVerboseLogDoesNotEvaluateMessageWhenVerboseLoggingIsDisabled() {
        let previousVerboseEnabled = VoxtLog.verboseEnabled
        VoxtLog.verboseEnabled = false
        defer {
            VoxtLog.verboseEnabled = previousVerboseEnabled
        }

        let logger = VoxtLogger(category: .app)
        var didEvaluateMessage = false

        logger.info({
            didEvaluateMessage = true
            return "expensive verbose message"
        }(), verbose: true)

        XCTAssertFalse(didEvaluateMessage)
    }
}

private nonisolated final class TechnicalLogOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedLines: [String] = []
    var lines: [String] { lock.lock(); defer { lock.unlock() }; return recordedLines }
    func append(_ line: String) { lock.lock(); recordedLines.append(line); lock.unlock() }
}
