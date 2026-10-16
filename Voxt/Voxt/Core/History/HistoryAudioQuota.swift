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
    var unavailableRootPaths: [String] = []
}

nonisolated enum HistoryAudioQuotaError: Error {
    case unsafePath, filesystem, invalidCatalog, oversized
}

nonisolated final class HistoryAudioFileUse: @unchecked Sendable {
    static let releasedNotification = Notification.Name("VoxtHistoryAudioUseReleased")
    private static let condition = NSCondition()
    private static var counts: [String: Int] = [:]
    private static var deletingPaths = Set<String>()
    private let path: String

    init(url: URL) {
        path = url.resolvingSymlinksInPath().standardizedFileURL.path
        Self.condition.lock()
        while Self.deletingPaths.contains(path) { Self.condition.wait() }
        Self.counts[path, default: 0] += 1
        Self.condition.unlock()
    }

    deinit {
        Self.condition.lock()
        Self.counts[path, default: 1] -= 1
        if Self.counts[path] == 0 { Self.counts.removeValue(forKey: path) }
        Self.condition.unlock()
        NotificationCenter.default.post(name: Self.releasedNotification, object: nil)
    }

    static func withUnusedURLs(
        _ urls: [URL],
        select: (Set<String>) -> Set<String>,
        operation: (Set<String>) -> Void
    ) {
        let paths = urls.map { $0.resolvingSymlinksInPath().standardizedFileURL.path }
        condition.lock()
        let unusedPaths = Set(paths.filter { counts[$0] == nil && !deletingPaths.contains($0) })
        let selectedPaths = select(unusedPaths)
        precondition(selectedPaths.isSubset(of: unusedPaths))
        deletingPaths.formUnion(selectedPaths)
        condition.unlock()
        defer {
            condition.lock()
            deletingPaths.subtract(selectedPaths)
            condition.broadcast()
            condition.unlock()
        }
        operation(selectedPaths)
    }

}

nonisolated final class HistoryAudioQuota: @unchecked Sendable {
    struct Archive: Codable {
        let path: String
        let createdAt: Date
        var deletionPending = false
    }

    static let lock = NSRecursiveLock()
    let rootURL: URL
    let removeFile: @Sendable (Int32, String) throws -> Void
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
        guard try !readCatalog(directory: directory).contains(where: { $0.path == path }) else { throw HistoryAudioQuotaError.filesystem }
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

    func openRoot() throws -> Int32 {
        let directory = open(rootURL.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        guard directory >= 0 else { throw HistoryAudioQuotaError.filesystem }
        return directory
    }

    func withParent<Result>(path: String, root: Int32, operation: (Int32, String) throws -> Result) throws -> Result {
        let parts = try components(path)
        let parent = openat(root, parts[0], O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        guard parent >= 0 else { throw HistoryAudioQuotaError.unsafePath }
        defer { close(parent) }
        return try operation(parent, parts[1])
    }

    func attributes(path: String, root: Int32) throws -> stat? {
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

    func readCatalog(directory: Int32) throws -> [Archive] {
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

    func writeCatalog(_ archives: [Archive], directory: Int32) throws {
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
