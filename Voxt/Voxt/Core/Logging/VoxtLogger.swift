// VoxtLogger.swift
// Provides typed category loggers backed by swift-log.

import Foundation
import Logging

struct VoxtLogger: Sendable {
    private let category: VoxtLogCategory
    private let logger: Logger

    nonisolated init(category: VoxtLogCategory, logger: Logger? = nil) {
        self.category = category
        if let logger {
            self.logger = logger
        } else {
            LoggingBootstrap.bootstrapIfNeeded()
            self.logger = Logger(label: "Voxt.\(category.rawValue)")
        }
    }

    nonisolated static func recordDiagnosticLine(_ line: String, logger: Logger? = nil) {
        if let logger {
            logger.info("\(line)")
        } else {
            LoggingBootstrap.bootstrapIfNeeded()
            Logger(label: "Voxt.asr").info("\(line)")
        }
    }

    nonisolated func record(_ event: RecordingDiagnosticEvent, level: Logger.Level? = nil) {
        var event = event
        let level = level ?? (event.outcome == .failed ? .error : .info)
        event.category = category
        event.severity = level >= .error ? .error : (level >= .warning ? .warning : .info)
        event.timestampMilliseconds = UInt64(max(0, Date().timeIntervalSince1970 * 1_000))
        guard let line = try? event.encodedLine() else { return }
        logger.log(level: level, "\(line)")
    }

    nonisolated func trace(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic,
        verbose: Bool = false
    ) {
        log(level: .trace, message: message, metadata: metadata, privacy: privacy, verbose: verbose)
    }

    nonisolated func debug(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic,
        verbose: Bool = false
    ) {
        log(level: .debug, message: message, metadata: metadata, privacy: privacy, verbose: verbose)
    }

    nonisolated func info(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic,
        verbose: Bool = false
    ) {
        log(level: .info, message: message, metadata: metadata, privacy: privacy, verbose: verbose)
    }

    nonisolated func notice(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic,
        verbose: Bool = false
    ) {
        log(level: .notice, message: message, metadata: metadata, privacy: privacy, verbose: verbose)
    }

    nonisolated func warning(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic
    ) {
        log(level: .warning, message: message, metadata: metadata, privacy: privacy)
    }

    nonisolated func error(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic
    ) {
        log(level: .error, message: message, metadata: metadata, privacy: privacy)
    }

    nonisolated func critical(
        _ message: @autoclosure () -> String,
        metadata: Logger.Metadata? = nil,
        privacy: VoxtLogPrivacy = .automatic
    ) {
        log(level: .critical, message: message, metadata: metadata, privacy: privacy)
    }

    private nonisolated func log(
        level: Logger.Level,
        message: () -> String,
        metadata: Logger.Metadata?,
        privacy: VoxtLogPrivacy,
        verbose: Bool = false
    ) {
        guard !verbose || VoxtLog.verboseEnabled else { return }
        // Note (Voxt): legacy warnings retain category/severity, never arbitrary content or metadata.
        guard level >= .warning else { return }
        record(.init(.legacyDiagnostic, outcome: .failed), level: level)
    }
}
