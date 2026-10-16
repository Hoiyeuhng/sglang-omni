// VoxtMultiplexLogHandler.swift
// Sends swift-log records to OSLog and local rolling diagnostic files.

import Logging

struct VoxtMultiplexLogHandler: LogHandler {
    var metadataProvider: Logger.MetadataProvider? {
        get { fileHandler.metadataProvider }
        set {
            fileHandler.metadataProvider = newValue
            osLogHandler.metadataProvider = newValue
        }
    }

    var metadata: Logger.Metadata {
        get { fileHandler.metadata }
        set {
            fileHandler.metadata = newValue
            osLogHandler.metadata = newValue
        }
    }

    var logLevel: Logger.Level {
        get { fileHandler.logLevel }
        set {
            fileHandler.logLevel = newValue
            osLogHandler.logLevel = newValue
        }
    }

    private var fileHandler: VoxtFileLogHandler
    private var osLogHandler: VoxtOSLogHandler

    init(
        label: String, category: VoxtLogCategory,
        fileStore: VoxtLogFileStore = .shared,
        osLogSink: (@Sendable (String, Logger.Level) -> Void)? = nil
    ) {
        fileHandler = VoxtFileLogHandler(label: label, category: category, fileStore: fileStore)
        osLogHandler = VoxtOSLogHandler(label: label, category: category, sink: osLogSink)
        logLevel = .info
    }

    subscript(metadataKey metadataKey: String) -> Logger.Metadata.Value? {
        get { fileHandler[metadataKey: metadataKey] }
        set {
            fileHandler[metadataKey: metadataKey] = newValue
            osLogHandler[metadataKey: metadataKey] = newValue
        }
    }

    func log(event: LogEvent) {
        fileHandler.log(event: event)
        osLogHandler.log(event: event)
    }
}
