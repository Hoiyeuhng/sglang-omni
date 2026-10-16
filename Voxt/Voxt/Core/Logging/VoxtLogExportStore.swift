// VoxtLogExportStore.swift
// Provides display and export payloads for local diagnostic logs.

import Foundation

enum VoxtLogExportStore {
    nonisolated static func latestLogUpdateDate() -> Date? {
        VoxtLogFileStore.shared.latestLogUpdateDate()
    }

    nonisolated static func latestLogDisplayText(
        limit: Int = 1_000,
        store: VoxtLogFileStore = .shared,
        metadataProvider: () -> RecordingDiagnosticExportMetadata
    ) -> String {
        let selectedLines = store.latestLines(limit: limit).compactMap(RecordingDiagnosticEvent.safeLine)
        let unavailableText = MainActorSync.run {
            AppLocalization.localizedString("No logs available")
        }
        let appMetaTitle = MainActorSync.run {
            AppLocalization.localizedString("App Meta")
        }
        let logText = selectedLines.isEmpty
            ? "[Voxt] <\(unavailableText)>"
            : selectedLines.joined(separator: "\n")

        return [
            logText,
            "========== \(appMetaTitle) ==========",
            metadataProvider().encodedText()
        ].joined(separator: "\n\n")
    }
}
