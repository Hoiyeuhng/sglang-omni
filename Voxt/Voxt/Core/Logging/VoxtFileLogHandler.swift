// VoxtFileLogHandler.swift
// Bridges swift-log records into Voxt rolling diagnostic files.

import Foundation
import Logging

struct VoxtFileLogHandler: LogHandler {
    var metadata: Logger.Metadata = [:]
    var metadataProvider: Logger.MetadataProvider?
    var logLevel: Logger.Level = .info

    private let fileStore: VoxtLogFileStore

    init(label: String, category: VoxtLogCategory, fileStore: VoxtLogFileStore = .shared) {
        self.fileStore = fileStore
    }

    subscript(metadataKey metadataKey: String) -> Logger.Metadata.Value? {
        get { metadata[metadataKey] }
        set { metadata[metadataKey] = newValue }
    }

    func log(event: LogEvent) {
        guard !VoxtRuntimeEnvironment.isRunningUnitTests else { return }
        guard let line = RecordingDiagnosticEvent.safeLine(event.message.description) else { return }
        fileStore.append(line)
    }

}
