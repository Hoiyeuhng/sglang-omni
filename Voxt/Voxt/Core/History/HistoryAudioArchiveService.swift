// HistoryAudioArchiveService.swift
// Provides History Audio Archive Service for history storage and display support.

import Foundation

@MainActor
protocol HistoryAudioArchiveManaging: AnyObject {
    func importArchive(
        from sourceURL: URL,
        kind: TranscriptionHistoryKind,
        preferredFileName: String?
    ) throws -> String
    func replaceArchive(for entry: TranscriptionHistoryEntry, with sourceURL: URL) throws -> String
    func audioURL(for entry: TranscriptionHistoryEntry) -> URL?
    func removeArchive(for entry: TranscriptionHistoryEntry)
    func removeArchive(relativePath: String?)
    func exportAllArchives(
        to destinationDirectoryURL: URL,
        forEachHistoryBatch: (([TranscriptionHistoryEntry]) -> Void) -> Void
    ) throws -> HistoryAudioExportSummary
    func storageStats(audioPaths: [String]) -> HistoryAudioStorageStats
    func rootURL() throws -> URL
    func releaseImportedArchive(path: String)
    func managedRootURLs() throws -> [URL]
}

final class HistoryAudioArchiveService: HistoryAudioArchiveManaging {
    private let fileManager: FileManager
    private let isolatedRootURL: URL?
    private var importedUses: [String: HistoryAudioFileUse] = [:]

    init(fileManager: FileManager = .default, rootURL: URL? = nil) {
        self.fileManager = fileManager
        self.isolatedRootURL = rootURL
    }

    func importArchive(
        from sourceURL: URL,
        kind: TranscriptionHistoryKind,
        preferredFileName: String? = nil
    ) throws -> String {
        let resolvedFileName = sanitizedFileName(
            preferredFileName?.trimmingCharacters(in: .whitespacesAndNewlines),
            fallbackKind: kind
        )
        let relativePath = "\(folderName(for: kind))/\(UUID().uuidString)-\(resolvedFileName)"
        let root = try rootURL()
        importedUses[relativePath] = HistoryAudioFileUse(url: root.appendingPathComponent(relativePath))
        do {
            try HistoryAudioQuota(rootURL: root).importArchive(from: sourceURL, path: relativePath)
        } catch {
            importedUses.removeValue(forKey: relativePath)
            throw error
        }
        return relativePath
    }

    func releaseImportedArchive(path: String) {
        importedUses.removeValue(forKey: path)
    }

    func replaceArchive(for entry: TranscriptionHistoryEntry, with sourceURL: URL) throws -> String {
        try importArchive(from: sourceURL, kind: entry.kind)
    }

    func audioURL(for entry: TranscriptionHistoryEntry) -> URL? {
        let relativePath = entry.audioRelativePath ?? entry.transcriptAudioRelativePath
        guard let relativePath, !relativePath.isEmpty else {
            return nil
        }
        let matches = (try? managedRootURLs())?.compactMap {
            try? HistoryAudioQuota(rootURL: $0).validatedURL(path: relativePath)
        }
        return matches?.count == 1 ? matches?[0] : nil
    }

    func removeArchive(for entry: TranscriptionHistoryEntry) {
        let relativePath = entry.audioRelativePath ?? entry.transcriptAudioRelativePath
        removeArchive(relativePath: relativePath)
    }

    func removeArchive(relativePath: String?) {
        guard let relativePath, !relativePath.isEmpty else { return }
        do {
            let url = try rootURL().appendingPathComponent(relativePath)
            if fileManager.fileExists(atPath: url.path) {
                try fileManager.removeItem(at: url)
            }
        } catch {
            return
        }
    }

    func exportAllArchives(
        to destinationDirectoryURL: URL,
        forEachHistoryBatch: (([TranscriptionHistoryEntry]) -> Void) -> Void
    ) throws -> HistoryAudioExportSummary {
        try fileManager.createDirectory(at: destinationDirectoryURL, withIntermediateDirectories: true)

        var exportedCount = 0
        var skippedCount = 0
        var failedCount = 0

        forEachHistoryBatch { batch in
            for entry in batch {
                guard let sourceURL = audioURL(for: entry),
                      fileManager.fileExists(atPath: sourceURL.path)
                else {
                    skippedCount += 1
                    continue
                }

                let audioUse = HistoryAudioFileUse(url: sourceURL)
                defer { withExtendedLifetime(audioUse) {} }
                do {
                    let folderURL = destinationDirectoryURL.appendingPathComponent(folderName(for: entry.kind), isDirectory: true)
                    try fileManager.createDirectory(at: folderURL, withIntermediateDirectories: true)
                    let destinationURL = folderURL.appendingPathComponent(exportFileName(for: entry))
                    if fileManager.fileExists(atPath: destinationURL.path) {
                        try fileManager.removeItem(at: destinationURL)
                    }
                    try fileManager.copyItem(at: sourceURL, to: destinationURL)
                    exportedCount += 1
                } catch {
                    failedCount += 1
                }
            }
        }

        return HistoryAudioExportSummary(
            exportedCount: exportedCount,
            skippedCount: skippedCount,
            failedCount: failedCount
        )
    }

    func storageStats(audioPaths: [String]) -> HistoryAudioStorageStats {
        Self.storageStats(rootURL: try? rootURL(), audioPaths: audioPaths)
    }

    func rootURL() throws -> URL {
        if let isolatedRootURL {
            try fileManager.createDirectory(at: isolatedRootURL, withIntermediateDirectories: true)
            return isolatedRootURL
        }
        return try HistoryAudioStorageDirectoryManager.ensureRootDirectoryExists()
    }

    func managedRootURLs() throws -> [URL] {
        isolatedRootURL == nil ? HistoryAudioStorageDirectoryManager.knownRootURLs() : [try rootURL()]
    }

    nonisolated static func storageStats(rootURL: URL?, audioPaths: [String]) -> HistoryAudioStorageStats {
        guard let rootURL else { return HistoryAudioStorageStats(storedFileCount: 0, totalBytes: 0, measurementFailed: true) }
        do {
            let quota = HistoryAudioQuota(rootURL: rootURL)
            try quota.registerExisting(paths: audioPaths)
            let snapshot = quota.snapshot(limitBytes: nil)
            return HistoryAudioStorageStats(
                storedFileCount: snapshot.fileCount, totalBytes: snapshot.totalBytes,
                measurementFailed: snapshot.measurementFailed
            )
        } catch {
            return HistoryAudioStorageStats(storedFileCount: 0, totalBytes: 0, measurementFailed: true)
        }
    }

    private func folderName(for kind: TranscriptionHistoryKind) -> String {
        switch kind {
        case .normal:
            return "transcription"
        case .translation:
            return "translation"
        case .rewrite:
            return "rewrite"
        case .transcript:
            return "transcript"
        }
    }

    private func sanitizedFileName(_ preferredFileName: String?, fallbackKind: TranscriptionHistoryKind) -> String {
        let trimmedPreferred = preferredFileName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let baseName = trimmedPreferred.isEmpty ? "\(folderName(for: fallbackKind))-\(UUID().uuidString)" : trimmedPreferred
        let filtered = baseName.map { character -> Character in
            if character.isLetter || character.isNumber || character == "-" || character == "_" {
                return character
            }
            return "-"
        }
        let normalized = String(filtered).trimmingCharacters(in: CharacterSet(charactersIn: "-_"))
        let resolved = normalized.isEmpty ? "\(folderName(for: fallbackKind))-\(UUID().uuidString)" : normalized
        return resolved.hasSuffix(".wav") ? resolved : "\(resolved).wav"
    }

    private func exportFileName(for entry: TranscriptionHistoryEntry) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "\(formatter.string(from: entry.createdAt))-\(folderName(for: entry.kind))-\(entry.id.uuidString).wav"
    }
}

struct HistoryAudioExportSummary: Equatable {
    let exportedCount: Int
    let skippedCount: Int
    let failedCount: Int
}

struct HistoryAudioStorageStats: Equatable {
    let storedFileCount: Int
    let totalBytes: Int64
    var measurementFailed = false
}

extension HistoryAudioArchiveManaging {
    func releaseImportedArchive(path: String) {}
    func managedRootURLs() throws -> [URL] { [try rootURL()] }
}
