// VoxtOSLogHandler.swift
// Bridges swift-log records into Apple Unified Logging.

import Foundation
import Logging
import OSLog

struct VoxtOSLogHandler: LogHandler {
    var metadata: Logging.Logger.Metadata = [:]
    var metadataProvider: Logging.Logger.MetadataProvider?
    var logLevel: Logging.Logger.Level = .info

    private let sink: @Sendable (String, Logging.Logger.Level) -> Void

    init(label: String, category: VoxtLogCategory, sink: (@Sendable (String, Logging.Logger.Level) -> Void)? = nil) {
        let bundleID = Bundle.main.bundleIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines)
        let osLog = OSLog(
            subsystem: bundleID?.isEmpty == false ? bundleID! : "app.voxt.Voxt",
            category: category.rawValue
        )
        self.sink = sink ?? { output, level in
            os_log("%{public}@", log: osLog, type: Self.osLogType(for: level), output)
        }
    }

    subscript(metadataKey metadataKey: String) -> Logging.Logger.Metadata.Value? {
        get { metadata[metadataKey] }
        set { metadata[metadataKey] = newValue }
    }

    func log(event: LogEvent) {
        guard let output = RecordingDiagnosticEvent.safeLine(event.message.description) else { return }
        sink(output, event.level)
    }

    private nonisolated static func osLogType(for level: Logging.Logger.Level) -> OSLogType {
        switch level {
        case .trace, .debug:
            return .debug
        case .info, .notice:
            return .info
        case .warning:
            return .default
        case .error:
            return .error
        case .critical:
            return .fault
        }
    }
}
