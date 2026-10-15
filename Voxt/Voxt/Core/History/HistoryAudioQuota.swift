import Darwin
import Foundation

nonisolated enum HistoryAudioQuotaPreference {
    static let limitKey = "historyAudioStorageLimitMB"
    static let migrationKey = "historyAudioQuotaMigrationPending"
    static let choices = [50, 100, 250, 500, 0]

    static func limitMegabytes(_ value: Int) -> Int {
        choices.contains(value) ? value : 100
    }

    static func initialize(defaults: UserDefaults, hasExistingData: Bool) {
        guard !defaults.bool(forKey: "historyAudioQuotaPolicyInitialized")
            || defaults.object(forKey: limitKey) == nil else { return }
        defaults.set(hasExistingData ? 0 : 100, forKey: limitKey)
        defaults.set(hasExistingData, forKey: migrationKey)
        defaults.set(true, forKey: "historyAudioQuotaPolicyInitialized")
    }

    static func limitBytes(defaults: UserDefaults) -> Int64? {
        let value = limitMegabytes(defaults.object(forKey: limitKey) as? Int ?? 100)
        return value == 0 ? nil : Int64(value) * 1_000_000
    }
}

nonisolated struct HistoryAudioQuotaSnapshot: Equatable, Sendable {
    var fileCount = 0
    var totalBytes: Int64 = 0
    var measurementFailed = false
    var cleanupFailed = false
    var cleanupDeferred = false
    var overLimit = false
}

nonisolated enum HistoryAudioQuotaError: Error {
    case unsafePath, filesystem, invalidCatalog, oversized
}

nonisolated final class HistoryAudioFileUse: @unchecked Sendable {
    static let releasedNotification = Notification.Name("VoxtHistoryAudioUseReleased")
    private static let lock = NSRecursiveLock()
    private static var counts: [String: Int] = [:]
    private let path: String

    init(url: URL) {
        path = url.resolvingSymlinksInPath().standardizedFileURL.path
        Self.lock.lock()
        Self.counts[path, default: 0] += 1
        Self.lock.unlock()
    }

    deinit {
        Self.lock.lock()
        Self.counts[path, default: 1] -= 1
        if Self.counts[path] == 0 { Self.counts.removeValue(forKey: path) }
        Self.lock.unlock()
        NotificationCenter.default.post(name: Self.releasedNotification, object: nil)
    }

    static func whileUnused<Result>(url: URL, operation: () throws -> Result) rethrows -> Result? {
        lock.lock()
        defer { lock.unlock() }
        guard counts[url.resolvingSymlinksInPath().standardizedFileURL.path] == nil else { return nil }
        return try operation()
    }
}

nonisolated final class HistoryAudioQuota: @unchecked Sendable {
    private struct Archive: Codable {
        let path: String
        let createdAt: Date
        var deletionPending = false
    }

    private static let lock = NSRecursiveLock()
    private let rootURL: URL
    private let removeFile: @Sendable (Int32, String) throws -> Void
    private let catalogName = ".voxt-audio-catalog.json"

    init(
        rootURL: URL,
        removeFile: @escaping @Sendable (Int32, String) throws -> Void = { directory, name in
            guard unlinkat(directory, name, 0) == 0 || errno == ENOENT else {
                throw HistoryAudioQuotaError.filesystem
            }
        }
    ) {
        self.rootURL = rootURL.standardizedFileURL
        self.removeFile = removeFile
    }

    func importArchive(from sourceURL: URL, path: String) throws {
        Self.lock.lock()
        defer { Self.lock.unlock() }
        let directory = try openRoot()
        defer { close(directory) }
        let parts = try components(path)
        guard mkdirat(directory, parts[0], 0o700) == 0 || errno == EEXIST else {
            throw HistoryAudioQuotaError.filesystem
        }
        try register(path: path)
        try withParent(path: path, root: directory) { parent, name in
            let source = open(sourceURL.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
            guard source >= 0 else { throw HistoryAudioQuotaError.filesystem }
            defer { close(source) }
            var attributes = stat()
            guard fstat(source, &attributes) == 0, attributes.st_mode & S_IFMT == S_IFREG, attributes.st_nlink == 1 else {
                throw HistoryAudioQuotaError.unsafePath
            }
            if renameatx_np(AT_FDCWD, sourceURL.path, parent, name, UInt32(RENAME_EXCL)) != 0 {
                guard errno == EXDEV else { throw HistoryAudioQuotaError.filesystem }
                let destination = openat(parent, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
                guard destination >= 0 else { throw HistoryAudioQuotaError.filesystem }
                let input = FileHandle(fileDescriptor: source, closeOnDealloc: false)
                let output = FileHandle(fileDescriptor: destination, closeOnDealloc: true)
                do {
                    while let chunk = try input.read(upToCount: 1_048_576), !chunk.isEmpty {
                        try output.write(contentsOf: chunk)
                    }
                    try output.synchronize()
                    guard unlink(sourceURL.path) == 0 else { throw HistoryAudioQuotaError.filesystem }
                } catch {
                    unlinkat(parent, name, 0)
                    throw error
                }
            }
        }
    }

    func validatedURL(path: String) throws -> URL? {
        let directory = try openRoot()
        defer { close(directory) }
        return try attributes(path: path, root: directory) == nil ? nil : rootURL.appendingPathComponent(path)
    }

    func register(path: String, createdAt: Date = Date()) throws {
        Self.lock.lock()
        defer { Self.lock.unlock() }
        let directory = try openRoot()
        defer { close(directory) }
        _ = try components(path)
        var archives = try readCatalog(directory: directory)
        if !archives.contains(where: { $0.path == path }) {
            archives.append(Archive(path: path, createdAt: createdAt))
            try writeCatalog(archives, directory: directory)
        }
    }

    func registerExisting(paths: [String]) throws {
        let directory = try openRoot()
        defer { close(directory) }
        var discovered: [Archive] = []
        for path in Set(paths).sorted() {
            if let attributes = try attributes(path: path, root: directory) {
                discovered.append(Archive(path: path, createdAt: Date(timeIntervalSince1970: Double(attributes.st_mtimespec.tv_sec))))
            }
        }
        Self.lock.lock()
        defer { Self.lock.unlock() }
        var archives = try readCatalog(directory: directory)
        let knownPaths = Set(archives.map(\.path))
        let additions = discovered.filter { !knownPaths.contains($0.path) }
        if !additions.isEmpty {
            archives.append(contentsOf: additions)
            try writeCatalog(archives, directory: directory)
        }
    }

    func requestDeletion(paths: Set<String>) throws {
        guard !paths.isEmpty else { return }
        Self.lock.lock()
        defer { Self.lock.unlock() }
        let directory = try openRoot()
        defer { close(directory) }
        var archives = try readCatalog(directory: directory)
        for index in archives.indices where paths.contains(archives[index].path) {
            archives[index].deletionPending = true
        }
        try writeCatalog(archives, directory: directory)
    }

    func snapshot(limitBytes: Int64?) -> HistoryAudioQuotaSnapshot {
        do {
            let directory = try openRoot()
            defer { close(directory) }
            return try snapshot(archives: readCatalog(directory: directory), root: directory, limitBytes: limitBytes)
        } catch {
            return HistoryAudioQuotaSnapshot(measurementFailed: true)
        }
    }

    func enforce(
        limitBytes: Int64?,
        protectedPaths: Set<String> = [],
        onlyPaths: Set<String>? = nil,
        clearReferences: (String) throws -> Void
    ) -> HistoryAudioQuotaSnapshot {
        var cleanupFailed = false
        do {
            let directory = try openRoot()
            defer { close(directory) }
            Self.lock.lock()
            let initialArchives: [Archive]
            do { initialArchives = try readCatalog(directory: directory) }
            catch { Self.lock.unlock(); throw error }
            Self.lock.unlock()
            let sizes = try Dictionary(uniqueKeysWithValues: initialArchives.map { archive in
                (archive.path, Int64(try attributes(path: archive.path, root: directory)?.st_size ?? 0))
            })
            var totalBytes = sizes.values.reduce(0, +)
            // Note (Codex): remove individually oversized archives before sacrificing smaller recordings.
            let orderedPaths = initialArchives.sorted { left, right in
                let leftOversized = limitBytes.map { sizes[left.path, default: 0] > $0 } ?? false
                let rightOversized = limitBytes.map { sizes[right.path, default: 0] > $0 } ?? false
                if leftOversized != rightOversized { return leftOversized }
                return left.createdAt == right.createdAt ? left.path < right.path : left.createdAt < right.createdAt
            }.map(\.path)
            for path in orderedPaths {
                Self.lock.lock()
                defer { Self.lock.unlock() }
                var archives = try readCatalog(directory: directory)
                guard let index = archives.firstIndex(where: { $0.path == path }) else { continue }
                let pending = archives[index].deletionPending
                if let onlyPaths, !onlyPaths.contains(path), !pending { continue }
                guard pending || limitBytes.map({ totalBytes > $0 }) == true else { continue }
                guard !protectedPaths.contains(path) else { continue }
                do {
                    let deleted = try HistoryAudioFileUse.whileUnused(url: rootURL.appendingPathComponent(path)) {
                        archives[index].deletionPending = true
                        try writeCatalog(archives, directory: directory)
                        if try attributes(path: path, root: directory) != nil {
                            try withParent(path: path, root: directory) { parent, name in
                                try removeFile(parent, name)
                            }
                            totalBytes -= sizes[path, default: 0]
                        }
                        try clearReferences(path)
                        archives.remove(at: index)
                        try writeCatalog(archives, directory: directory)
                        return true
                    }
                    if deleted == nil { continue }
                } catch {
                    cleanupFailed = true
                }
            }
            var result = try snapshot(archives: readCatalog(directory: directory), root: directory, limitBytes: limitBytes)
            result.cleanupFailed = cleanupFailed
            return result
        } catch {
            return HistoryAudioQuotaSnapshot(measurementFailed: true, cleanupFailed: cleanupFailed)
        }
    }

    static func reconcile(
        roots: [URL],
        referencedPaths: [String],
        requestedDeletions: Set<String> = [],
        limitBytes: Int64?,
        protectedPaths: Set<String>,
        clearReferences: (String) throws -> Void
    ) -> HistoryAudioQuotaSnapshot {
        do {
            var seenRoots = Set<String>()
            let quotas = roots.filter { seenRoots.insert($0.resolvingSymlinksInPath().path).inserted }
                .map { HistoryAudioQuota(rootURL: $0) }
            for path in Set(referencedPaths).union(requestedDeletions) {
                let owners = try quotas.filter { try $0.validatedURL(path: path) != nil }
                guard owners.count <= 1 else { throw HistoryAudioQuotaError.unsafePath }
                try owners.first?.registerExisting(paths: [path])
                if owners.isEmpty, requestedDeletions.contains(path) { try clearReferences(path) }
            }
            for quota in quotas { try quota.requestDeletion(paths: requestedDeletions) }
            var candidates: [(quota: HistoryAudioQuota, archive: Archive, bytes: Int64)] = []
            for quota in quotas {
                let directory = try quota.openRoot()
                defer { close(directory) }
                for archive in try quota.readCatalog(directory: directory) {
                    let bytes = Int64(try quota.attributes(path: archive.path, root: directory)?.st_size ?? 0)
                    candidates.append((quota, archive, bytes))
                }
            }
            guard Set(candidates.map { $0.archive.path }).count == candidates.count else {
                throw HistoryAudioQuotaError.unsafePath
            }
            candidates.sort { left, right in
                let leftOversized = limitBytes.map { left.bytes > $0 } ?? false
                let rightOversized = limitBytes.map { right.bytes > $0 } ?? false
                if leftOversized != rightOversized { return leftOversized }
                return left.archive.createdAt == right.archive.createdAt
                    ? left.archive.path < right.archive.path : left.archive.createdAt < right.archive.createdAt
            }
            var totalBytes = candidates.reduce(Int64(0)) { $0 + $1.bytes }
            var result = HistoryAudioQuotaSnapshot()
            for candidate in candidates {
                guard candidate.archive.deletionPending || limitBytes.map({ totalBytes > $0 }) == true else { continue }
                let before = candidate.quota.snapshot(limitBytes: nil)
                let after = candidate.quota.enforce(
                    limitBytes: 0, protectedPaths: protectedPaths, onlyPaths: [candidate.archive.path],
                    clearReferences: clearReferences
                )
                totalBytes -= before.totalBytes - after.totalBytes
                result.cleanupFailed = result.cleanupFailed || after.cleanupFailed
                result.measurementFailed = result.measurementFailed || after.measurementFailed
            }
            for quota in quotas {
                let snapshot = quota.snapshot(limitBytes: nil)
                result.fileCount += snapshot.fileCount
                result.totalBytes += snapshot.totalBytes
                result.measurementFailed = result.measurementFailed || snapshot.measurementFailed
            }
            result.overLimit = limitBytes.map { result.totalBytes > $0 } ?? false
            return result
        } catch {
            return HistoryAudioQuotaSnapshot(measurementFailed: true)
        }
    }

    private func snapshot(archives: [Archive], root: Int32, limitBytes: Int64?) throws -> HistoryAudioQuotaSnapshot {
        var result = HistoryAudioQuotaSnapshot()
        for archive in archives {
            if let attributes = try attributes(path: archive.path, root: root) {
                result.fileCount += 1
                result.totalBytes += Int64(attributes.st_size)
            }
        }
        result.overLimit = limitBytes.map { result.totalBytes > $0 } ?? false
        return result
    }

    private func components(_ path: String) throws -> [String] {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 2,
              ["transcription", "translation", "rewrite", "transcript"].contains(parts[0]),
              !parts[1].isEmpty, parts[1] != ".", parts[1] != "..", parts[1].hasSuffix(".wav"),
              !path.contains("\0") else { throw HistoryAudioQuotaError.unsafePath }
        return parts
    }

    private func openRoot() throws -> Int32 {
        let directory = open(rootURL.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        guard directory >= 0 else { throw HistoryAudioQuotaError.filesystem }
        return directory
    }

    private func withParent<Result>(path: String, root: Int32, operation: (Int32, String) throws -> Result) throws -> Result {
        let parts = try components(path)
        let parent = openat(root, parts[0], O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        guard parent >= 0 else { throw HistoryAudioQuotaError.unsafePath }
        defer { close(parent) }
        return try operation(parent, parts[1])
    }

    private func attributes(path: String, root: Int32) throws -> stat? {
        let parts = try components(path)
        var directoryAttributes = stat()
        if fstatat(root, parts[0], &directoryAttributes, AT_SYMLINK_NOFOLLOW) != 0, errno == ENOENT { return nil }
        return try withParent(path: path, root: root) { parent, name in
            var attributes = stat()
            guard fstatat(parent, name, &attributes, AT_SYMLINK_NOFOLLOW) == 0 else {
                if errno == ENOENT { return nil }
                throw HistoryAudioQuotaError.filesystem
            }
            guard attributes.st_mode & S_IFMT == S_IFREG, attributes.st_nlink == 1 else {
                throw HistoryAudioQuotaError.unsafePath
            }
            return attributes
        }
    }

    private func readCatalog(directory: Int32) throws -> [Archive] {
        let descriptor = openat(directory, catalogName, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else {
            if errno == ENOENT { return [] }
            throw HistoryAudioQuotaError.filesystem
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        var attributes = stat()
        guard fstat(descriptor, &attributes) == 0, attributes.st_mode & S_IFMT == S_IFREG,
              attributes.st_size <= 16_000_000 else { throw HistoryAudioQuotaError.invalidCatalog }
        let archives = try JSONDecoder().decode([Archive].self, from: handle.readToEnd() ?? Data())
        guard Set(archives.map(\.path)).count == archives.count else { throw HistoryAudioQuotaError.invalidCatalog }
        for archive in archives { _ = try components(archive.path) }
        return archives
    }

    private func writeCatalog(_ archives: [Archive], directory: Int32) throws {
        let temporaryName = ".voxt-audio-catalog-\(UUID().uuidString).tmp"
        let descriptor = openat(directory, temporaryName, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw HistoryAudioQuotaError.filesystem }
        defer { unlinkat(directory, temporaryName, 0) }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        try handle.write(contentsOf: JSONEncoder().encode(archives))
        try handle.synchronize()
        guard renameat(directory, temporaryName, directory, catalogName) == 0 else {
            throw HistoryAudioQuotaError.filesystem
        }
        guard fsync(directory) == 0 else { throw HistoryAudioQuotaError.filesystem }
    }
}
