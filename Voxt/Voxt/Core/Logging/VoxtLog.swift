// VoxtLog.swift
// Provides the compatibility logging facade for the app.

import Darwin
import AppKit
import Foundation
import Logging

enum VoxtLog {
    struct ExportPayload {
        let filename: String
        let content: String
    }

    nonisolated(unsafe) static var verboseEnabled = false

    nonisolated static func info(_ message: @autoclosure () -> String, verbose: Bool = false) {
        appLogger.info(message(), verbose: verbose)
    }

    nonisolated static func audio(_ message: @autoclosure () -> String, verbose: Bool = false) {
        audioLogger.info(message(), verbose: verbose)
    }

    nonisolated static func audioWarning(_ message: @autoclosure () -> String) {
        audioLogger.warning(message())
    }

    nonisolated static func dictionary(_ message: @autoclosure () -> String, verbose: Bool = false) {
        dictionaryLogger.info(message(), verbose: verbose)
    }

    nonisolated static func dictionaryWarning(_ message: @autoclosure () -> String) {
        dictionaryLogger.warning(message())
    }

    nonisolated static func history(_ message: @autoclosure () -> String, verbose: Bool = false) {
        historyLogger.info(message(), verbose: verbose)
    }

    nonisolated static func historyWarning(_ message: @autoclosure () -> String) {
        historyLogger.warning(message())
    }

    nonisolated static func hotkey(_ message: @autoclosure () -> String, verbose: Bool = false) {
        guard isHotkeyDebugLoggingEnabled,
              !verbose || verboseEnabled
        else {
            return
        }
        hotkeyLogger.info(message())
    }

    nonisolated static var isHotkeyDebugLoggingEnabled: Bool {
        UserDefaults.standard.bool(forKey: AppPreferenceKey.hotkeyDebugLoggingEnabled)
    }

    nonisolated static func input(_ message: @autoclosure () -> String, verbose: Bool = false) {
        inputLogger.info(message(), verbose: verbose)
    }

    nonisolated static func inputWarning(_ message: @autoclosure () -> String) {
        inputLogger.warning(message())
    }

    nonisolated static func asr(_ message: @autoclosure () -> String, verbose: Bool = false) {
        asrLogger.info(message(), verbose: verbose)
    }

    nonisolated static func asrWarning(_ message: @autoclosure () -> String) {
        asrLogger.warning(message())
    }

    nonisolated static func asrError(_ message: @autoclosure () -> String) {
        asrLogger.error(message())
    }

    nonisolated static func llm(_ message: @autoclosure () -> String) {
        guard UserDefaults.standard.bool(forKey: AppPreferenceKey.llmDebugLoggingEnabled) else { return }
        // LLM prompts, user input, and generated output can contain arbitrary private
        // content. The regular model-debug switch must never make that content part of
        // persistent diagnostics; callers should log dimensions and timings separately.
        // Do not evaluate the autoclosure either, which avoids constructing and scanning
        // large prompt/response strings merely to redact them.
        _ = message
    }

    nonisolated static func llmDebug(_ message: @autoclosure () -> String) {
        guard UserDefaults.standard.bool(forKey: AppPreferenceKey.llmDebugLoggingEnabled) else { return }
        llmLogger.info(message())
    }

    nonisolated static func llmInfo(_ message: @autoclosure () -> String, verbose: Bool = false) {
        llmLogger.info(message(), verbose: verbose)
    }

    nonisolated static func llmWarning(_ message: @autoclosure () -> String) {
        llmLogger.warning(message())
    }

    nonisolated static func model(_ message: @autoclosure () -> String) {
        guard UserDefaults.standard.bool(forKey: AppPreferenceKey.llmDebugLoggingEnabled) else { return }
        modelLogger.info(message(), privacy: .preview(limit: 4_000))
    }

    nonisolated static func vad(_ message: @autoclosure () -> String) {
        guard UserDefaults.standard.bool(forKey: AppPreferenceKey.llmDebugLoggingEnabled) else { return }
        modelLogger.info(message(), privacy: .preview(limit: 4_000))
    }

    nonisolated static func modelInfo(_ message: @autoclosure () -> String, verbose: Bool = false) {
        modelLogger.info(message(), verbose: verbose)
    }

    nonisolated static func modelWarning(_ message: @autoclosure () -> String) {
        modelLogger.warning(message())
    }

    nonisolated static func modelError(_ message: @autoclosure () -> String) {
        modelLogger.error(message())
    }

    nonisolated static func meeting(_ message: @autoclosure () -> String, verbose: Bool = false) {
        meetingLogger.info(message(), verbose: verbose)
    }

    nonisolated static func meetingWarning(_ message: @autoclosure () -> String) {
        meetingLogger.warning(message())
    }

    nonisolated static func meetingError(_ message: @autoclosure () -> String) {
        meetingLogger.error(message())
    }

    nonisolated static func network(_ message: @autoclosure () -> String, verbose: Bool = false) {
        networkLogger.info(message(), verbose: verbose)
    }

    nonisolated static func networkWarning(_ message: @autoclosure () -> String) {
        networkLogger.warning(message())
    }

    nonisolated static func settings(_ message: @autoclosure () -> String, verbose: Bool = false) {
        settingsLogger.info(message(), verbose: verbose)
    }

    nonisolated static func settingsWarning(_ message: @autoclosure () -> String) {
        settingsLogger.warning(message())
    }

    nonisolated static func persistence(_ message: @autoclosure () -> String, verbose: Bool = false) {
        persistenceLogger.info(message(), verbose: verbose)
    }

    nonisolated static func persistenceWarning(_ message: @autoclosure () -> String) {
        persistenceLogger.warning(message())
    }

    nonisolated static func persistenceError(_ message: @autoclosure () -> String) {
        persistenceLogger.error(message())
    }

    nonisolated static func translation(_ message: @autoclosure () -> String, verbose: Bool = false) {
        translationLogger.info(message(), verbose: verbose)
    }

    nonisolated static func translationWarning(_ message: @autoclosure () -> String) {
        translationLogger.warning(message())
    }

    nonisolated static func update(_ message: @autoclosure () -> String, verbose: Bool = false) {
        updateLogger.info(message(), verbose: verbose)
    }

    nonisolated static func updateWarning(_ message: @autoclosure () -> String) {
        updateLogger.warning(message())
    }

    nonisolated static func updateError(_ message: @autoclosure () -> String) {
        updateLogger.error(message())
    }

    nonisolated static func securityWarning(_ message: @autoclosure () -> String) {
        securityLogger.warning(message())
    }

    nonisolated static func llmPreview(_ text: String, limit: Int = 1200) -> String {
        VoxtLogRedactor.preview(text, limit: limit)
    }

    nonisolated static func warning(_ message: @autoclosure () -> String) {
        appLogger.warning(message())
    }

    nonisolated static func error(_ message: @autoclosure () -> String) {
        appLogger.error(message())
    }

    nonisolated static func latestLogUpdateDate() -> Date? {
        VoxtLogExportStore.latestLogUpdateDate()
    }

    nonisolated static func latestLogExportPayload(limit: Int = 1000) -> ExportPayload {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let filename = "voxt-log-\(formatter.string(from: Date())).txt"
        let content = VoxtLogExportStore.latestLogDisplayText(limit: limit) {
            MainActorSync.run {
                diagnosticsMetadataText()
            }
        }
        return ExportPayload(filename: filename, content: content)
    }

    nonisolated static func latestLogDisplayText(limit: Int = 1000) -> String {
        VoxtLogExportStore.latestLogDisplayText(limit: limit) {
            MainActorSync.run {
                diagnosticsMetadataText()
            }
        }
    }

    nonisolated static func exportLatestLogs(limit: Int = 1000) throws -> URL {
        let payload = latestLogExportPayload(limit: limit)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(payload.filename)
        try payload.content.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private nonisolated static let appLogger = VoxtLogger(category: .app)
    private nonisolated static let audioLogger = VoxtLogger(category: .audio)
    private nonisolated static let dictionaryLogger = VoxtLogger(category: .dictionary)
    private nonisolated static let historyLogger = VoxtLogger(category: .history)
    private nonisolated static let hotkeyLogger = VoxtLogger(category: .hotkey)
    private nonisolated static let inputLogger = VoxtLogger(category: .input)
    private nonisolated static let asrLogger = VoxtLogger(category: .asr)
    private nonisolated static let llmLogger = VoxtLogger(category: .llm)
    private nonisolated static let modelLogger = VoxtLogger(category: .model)
    private nonisolated static let meetingLogger = VoxtLogger(category: .meeting)
    private nonisolated static let networkLogger = VoxtLogger(category: .network)
    private nonisolated static let settingsLogger = VoxtLogger(category: .settings)
    private nonisolated static let persistenceLogger = VoxtLogger(category: .persistence)
    private nonisolated static let translationLogger = VoxtLogger(category: .translation)
    private nonisolated static let updateLogger = VoxtLogger(category: .update)
    private nonisolated static let securityLogger = VoxtLogger(category: .security)

    @MainActor
    private static func diagnosticsMetadataText(defaults: UserDefaults = .standard) -> RecordingDiagnosticExportMetadata {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return RecordingDiagnosticExportMetadata(
            macOSMajorVersion: version.majorVersion,
            macOSMinorVersion: version.minorVersion,
            macOSPatchVersion: version.patchVersion,
            recordingDiagnosticsEnabled: defaults.bool(forKey: RecordingDiagnosticRecorder.preferenceKey),
            diagnosticWriteFailed: VoxtLogFileStore.shared.lastWriteFailed
        )
    }

}
