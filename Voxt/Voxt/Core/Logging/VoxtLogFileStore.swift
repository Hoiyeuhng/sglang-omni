// VoxtLogFileStore.swift
// Stores local rolling diagnostic logs for user export.

import Darwin
import Foundation

nonisolated final class VoxtLogFileStore: @unchecked Sendable {
    nonisolated static let shared = VoxtLogFileStore()
    private nonisolated static let queue = DispatchQueue(label: "app.voxt.logging.file-store", qos: .utility)

    private let fileManager: FileManager
    private let maxFileBytes: UInt64
    private let retainedArchiveCount: Int
    private let baseDirectoryURL: URL?
    private let quotaDefaults: UserDefaults?
    private var writeFailed = false

    static let quotaPreferenceKey = "diagnosticLogStorageLimitMiB"
    static let quotaChoices = [4, 12, 32, 64, 0]

    var lastWriteFailed: Bool { Self.queue.sync { writeFailed } }

    private var storageLimitBytes: UInt64? {
        guard let quotaDefaults else { return maxFileBytes * UInt64(max(0, retainedArchiveCount) + 1) }
        let configured = quotaDefaults.object(forKey: Self.quotaPreferenceKey) as? Int ?? 12
        let mebibytes = Self.quotaChoices.contains(configured) ? configured : 12
        return mebibytes == 0 ? nil : UInt64(mebibytes) * 1_048_576
    }

    func storageUsage() -> (bytes: Int64?, cleanupFailed: Bool, overLimit: Bool) {
        Self.queue.sync {
            prepareStorageIfNeeded()
            do {
                let bytes = try managedLogFiles().reduce(Int64(0)) { $0 + $1.bytes }
                return (bytes, writeFailed, storageLimitBytes.map { UInt64(bytes) > $0 } ?? false)
            } catch {
                return (nil, writeFailed, false)
            }
        }
    }

    func applyStorageQuota() {
        Self.queue.async { [weak self] in
            guard let self else { return }
            self.prepareStorageIfNeeded()
            do { try self.pruneToQuota(); self.writeFailed = false }
            catch { self.writeFailed = true }
        }
    }
    private var recentLines: [String] = []
    private let maxMemoryLines = 2_000
    private var didPrepareStorage = false

    init(
        fileManager: FileManager = FileManager(),
        maxFileBytes: UInt64 = 2 * 1_024 * 1_024,
        retainedArchiveCount: Int = 5,
        baseDirectoryURL: URL? = nil,
        defaults: UserDefaults? = nil
    ) {
        self.fileManager = fileManager
        self.maxFileBytes = maxFileBytes
        self.retainedArchiveCount = retainedArchiveCount
        self.baseDirectoryURL = baseDirectoryURL
        self.quotaDefaults = defaults ?? (baseDirectoryURL == nil ? .standard : nil)
    }

    var logsDirectoryURL: URL {
        if let baseDirectoryURL {
            return baseDirectoryURL
        }
        let supportDirectory = try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let base = supportDirectory ?? fileManager.temporaryDirectory
        return base
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("Logs", isDirectory: true)
    }

    var currentLogURL: URL {
        logsDirectoryURL.appendingPathComponent("current.log")
    }

    var archiveDirectoryURL: URL {
        logsDirectoryURL.appendingPathComponent("archive", isDirectory: true)
    }

    var legacyDirectoryURL: URL {
        logsDirectoryURL.appendingPathComponent("legacy", isDirectory: true)
    }

    func append(_ line: String) {
        Self.queue.async { [weak self] in
            self?.appendSynchronously(line)
        }
    }

    func latestLogUpdateDate() -> Date? {
        Self.queue.sync {
            prepareStorageIfNeeded()
            guard didPrepareStorage else { return nil }
            return (try? fileManager.attributesOfItem(atPath: currentLogURL.path)[.modificationDate]) as? Date
        }
    }

    func latestLines(limit: Int) -> [String] {
        Self.queue.sync {
            prepareStorageIfNeeded()
            guard didPrepareStorage else { return [] }
            let selectedLimit = max(1, limit)
            var lines = recentLines
            if lines.count < selectedLimit {
                lines = readPersistedLines(limit: selectedLimit)
            }
            return Array(lines.suffix(selectedLimit))
        }
    }

    func flush() {
        Self.queue.sync {
            prepareStorageIfNeeded()
        }
    }

    private func appendSynchronously(_ line: String) {
        prepareStorageIfNeeded()
        do {
            let encoded = Data((line + "\n").utf8)
            guard UInt64(encoded.count) <= min(maxFileBytes, storageLimitBytes ?? maxFileBytes) else {
                writeFailed = true
                return
            }
            try rotateIfNeeded(incomingLine: line)
            try pruneToQuota()
            try appendLineData(encoded)
            try pruneToQuota()
            recentLines.append(line)
            if recentLines.count > maxMemoryLines { recentLines = Array(recentLines.suffix(maxMemoryLines)) }
            writeFailed = false
        } catch {
            writeFailed = true
        }
    }

    private func appendLineData(_ data: Data) throws {
        let descriptor = open(currentLogURL.path, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw CocoaError(.fileWriteUnknown) }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        try handle.write(contentsOf: data)
    }

    private func prepareStorageIfNeeded() {
        guard !didPrepareStorage else { return }
        do {
            try validateLogDirectories()
            try fileManager.createDirectory(at: logsDirectoryURL, withIntermediateDirectories: true)
            try fileManager.createDirectory(at: archiveDirectoryURL, withIntermediateDirectories: true)
            _ = try managedLogFiles()
            migrateLegacyLogIfNeeded()
            _ = try managedLogFiles()
            removePersistedUnitTestLaunchLogsIfNeeded()
            recentLines = readPersistedLines(limit: maxMemoryLines)
            didPrepareStorage = true
        } catch {
            writeFailed = true
        }
    }

    private func migrateLegacyLogIfNeeded() {
        let legacyLogURL = logsDirectoryURL.appendingPathComponent("voxt.log")
        guard fileManager.fileExists(atPath: legacyLogURL.path) else { return }
        try? fileManager.createDirectory(at: legacyDirectoryURL, withIntermediateDirectories: true)
        let dateText = legacyDateFormatter.string(from: Date())
        let destination = legacyDirectoryURL.appendingPathComponent("voxt-legacy-\(dateText).log")
        if fileManager.fileExists(atPath: destination.path) {
            try? fileManager.removeItem(at: destination)
        }
        try? fileManager.moveItem(at: legacyLogURL, to: destination)
    }

    private func rotateIfNeeded(incomingLine: String) throws {
        let files = try managedLogFiles()
        let currentBytes = files.first { $0.url == currentLogURL }?.bytes ?? 0
        guard UInt64(currentBytes) + UInt64(incomingLine.utf8.count + 1) > maxFileBytes, currentBytes > 0 else { return }
        let archives = files.filter { $0.url.deletingLastPathComponent() == archiveDirectoryURL }
        let indexedArchives = archives.compactMap { archive -> (URL, Int)? in
            guard let index = Int(archive.url.deletingPathExtension().lastPathComponent.dropFirst(5)), index > 0,
                  index < Int.max else { return nil }
            return (archive.url, index)
        }
        for (source, index) in indexedArchives.sorted(by: { $0.1 > $1.1 }) {
            try fileManager.moveItem(at: source, to: archiveDirectoryURL.appendingPathComponent("voxt-\(index + 1).log"))
        }
        try fileManager.moveItem(at: currentLogURL, to: archiveDirectoryURL.appendingPathComponent("voxt-1.log"))
    }

    private func validateLogDirectories() throws {
        for directory in [logsDirectoryURL, archiveDirectoryURL, legacyDirectoryURL] {
            var attributes = stat()
            if lstat(directory.path, &attributes) == 0 {
                guard attributes.st_mode & S_IFMT == S_IFDIR else { throw CocoaError(.fileReadUnknown) }
            } else if errno != ENOENT {
                throw CocoaError(.fileReadUnknown)
            }
        }
    }

    private func managedLogFiles() throws -> [(url: URL, bytes: Int64, modified: Date)] {
        try validateLogDirectories()
        var urls = [currentLogURL, logsDirectoryURL.appendingPathComponent("voxt.log")]
        for directory in [archiveDirectoryURL, legacyDirectoryURL] {
            guard fileManager.fileExists(atPath: directory.path) else { continue }
            let attributes = try fileManager.attributesOfItem(atPath: directory.path)
            guard attributes[.type] as? FileAttributeType == .typeDirectory else { throw CocoaError(.fileReadUnknown) }
            let children = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            urls += children.filter { url in
                let name = url.lastPathComponent
                if directory == archiveDirectoryURL {
                    return name.hasPrefix("voxt-") && name.hasSuffix(".log")
                        && Int(name.dropFirst(5).dropLast(4)) != nil
                }
                return name.range(of: "^voxt-legacy-[0-9]{8}\\.log$", options: .regularExpression) != nil
            }.map { directory.appendingPathComponent($0.lastPathComponent) }
        }
        return try urls.compactMap { url in
            var fileStatus = stat()
            guard lstat(url.path, &fileStatus) == 0 else {
                if errno == ENOENT { return nil }
                throw CocoaError(.fileReadUnknown)
            }
            guard fileStatus.st_mode & S_IFMT == S_IFREG, fileStatus.st_nlink == 1 else {
                throw CocoaError(.fileReadUnknown)
            }
            let attributes = try fileManager.attributesOfItem(atPath: url.path)
            guard attributes[.type] as? FileAttributeType == .typeRegular,
                  let bytes = attributes[.size] as? NSNumber else { throw CocoaError(.fileReadUnknown) }
            return (url, bytes.int64Value, attributes[.modificationDate] as? Date ?? .distantPast)
        }
    }

    private func pruneToQuota() throws {
        guard let limit = storageLimitBytes else { return }
        let files = try managedLogFiles()
        var bytes = files.reduce(Int64(0)) { $0 + $1.bytes }
        for file in files.filter({ $0.url != currentLogURL }).sorted(by: { $0.modified < $1.modified }) {
            guard bytes > Int64(limit) else { break }
            try fileManager.removeItem(at: file.url)
            bytes -= file.bytes
        }
        if bytes > Int64(limit) {
            throw CocoaError(.fileWriteOutOfSpace)
        }
    }

    private func readPersistedLines(limit: Int) -> [String] {
        var lines: [String] = []
        for url in readableLogURLs() {
            guard let text = try? String(contentsOf: url, encoding: .utf8), !text.isEmpty else { continue }
            lines.append(contentsOf: Self.removingUnitTestLaunchLines(from: text.split(whereSeparator: \.isNewline).map(String.init)))
            if lines.count > limit * 2 {
                lines = Array(lines.suffix(limit))
            }
        }
        return Array(lines.suffix(limit))
    }

    private func readableLogURLs() -> [URL] {
        let archived = (1..<max(1, retainedArchiveCount + 1))
            .reversed()
            .map { archiveDirectoryURL.appendingPathComponent("voxt-\($0).log") }
        return archived + [currentLogURL]
    }

    private func removePersistedUnitTestLaunchLogsIfNeeded() {
        for url in readableLogURLs() where fileManager.fileExists(atPath: url.path) {
            guard let text = try? String(contentsOf: url, encoding: .utf8), !text.isEmpty else { continue }
            let originalLines = text.split(whereSeparator: \.isNewline).map(String.init)
            let cleanedLines = Self.removingUnitTestLaunchLines(from: originalLines)
            guard cleanedLines.count != originalLines.count else { continue }
            try? cleanedLines.joined(separator: "\n").appending(cleanedLines.isEmpty ? "" : "\n").write(
                to: url,
                atomically: true,
                encoding: .utf8
            )
        }
    }

    static func removingUnitTestLaunchLines(from lines: [String]) -> [String] {
        var removedIndexes = Set<Int>()
        for (index, line) in lines.enumerated() where line.contains("Voxt launch running under XCTest") {
            removedIndexes.insert(index)
            let runtimeIndex = index - 1
            if runtimeIndex >= 0, lines[runtimeIndex].contains("[app] Runtime system version:") {
                removedIndexes.insert(runtimeIndex)
            }
            let launchIndex = index - 2
            if launchIndex >= 0, lines[launchIndex].contains("[app] Voxt launching.") {
                removedIndexes.insert(launchIndex)
            }
        }
        guard !removedIndexes.isEmpty else { return lines }
        return lines.enumerated().compactMap { removedIndexes.contains($0.offset) ? nil : $0.element }
    }

    private var legacyDateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd"
        return formatter
    }
}
