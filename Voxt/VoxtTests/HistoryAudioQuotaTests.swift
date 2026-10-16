import Darwin
import Foundation
#if !VOXT_STORAGE_CONTRACT_TESTS
import XCTest
@testable import Voxt
#endif

nonisolated enum HistoryAudioQuotaContracts {
    enum Failure: Error { case assertion(String), injected }

    static func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
        if !condition() { throw Failure.assertion(message) }
    }

    static func withRoot(_ body: (URL, HistoryAudioQuota) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-quota-test-\(UUID())")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("transcription"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root, HistoryAudioQuota(rootURL: root))
    }

    static func archive(_ root: URL, _ quota: HistoryAudioQuota, _ name: String, bytes: Int, age: Double = 0) throws -> String {
        let path = "transcription/\(name).wav"
        try Data(repeating: 7, count: bytes).write(to: root.appendingPathComponent(path))
        try quota.register(path: path, createdAt: Date(timeIntervalSince1970: age))
        return path
    }

    static func boundaries() throws {
        for limit in [9, 10, 11] {
            try withRoot { root, quota in
                let path = try archive(root, quota, "audio", bytes: 10)
                var cleared: [String] = []
                let result = quota.enforce(limitBytes: Int64(limit)) { cleared.append($0) }
                try check(!result.measurementFailed && !result.cleanupFailed, "boundary operation failed")
                try check(result.totalBytes == (limit < 10 ? 0 : 10), "quota boundary")
                try check(cleared == (limit < 10 ? [path] : []), "references cleared incorrectly")
            }
        }
    }

    static func oldestAndOversized() throws {
        try withRoot { root, quota in
            let old = try archive(root, quota, "old", bytes: 4, age: 1)
            let middle = try archive(root, quota, "middle", bytes: 4, age: 2)
            _ = try archive(root, quota, "new", bytes: 4, age: 3)
            var cleared: [String] = []
            let result = quota.enforce(limitBytes: 4) { cleared.append($0) }
            try check(cleared == [old, middle] && result.totalBytes == 4, "oldest first")
            let oversized = try archive(root, quota, "oversized", bytes: 100, age: 4)
            cleared = []
            let next = quota.enforce(limitBytes: 8) { cleared.append($0) }
            try check(cleared == [oversized] && next.totalBytes == 4, "oversized must not destroy small history")
        }
    }

    static func blockedOversizedPreservesSmallerArchives() throws {
        for failure in [false, true] {
            try withRoot { root, quota in
                let oversized = try archive(root, quota, "oversized", bytes: 120)
                let small = try archive(root, quota, "small", bytes: 20)
                let explicit = try archive(root, quota, "explicit", bytes: 10)
                try quota.requestDeletion(paths: [explicit])
                let use = failure ? nil : HistoryAudioFileUse(url: root.appendingPathComponent(oversized))
                let testedQuota = HistoryAudioQuota(rootURL: root, removeFile: { directory, name in
                    if name == "oversized.wav", failure { throw Failure.injected }
                    guard unlinkat(directory, name, 0) == 0 else { throw Failure.injected }
                })
                var cleared: [String] = []
                let result = withExtendedLifetime(use) {
                    testedQuota.enforce(limitBytes: 100) { cleared.append($0) }
                }
                try check(result.overLimit && result.totalBytes == 140, "blocked oversized archive must not sacrifice smaller history")
                try check(cleared == [explicit], "explicit deletion must proceed independently of blocked capacity cleanup")
                try check(FileManager.default.fileExists(atPath: root.appendingPathComponent(small).path), "small archive lost")
            }
        }
    }

    static func unavailableRootPreservesAudioButCompletesExplicitDeletion() throws {
        try withRoot { root, quota in
            let retained = try archive(root, quota, "retained", bytes: 20)
            let removed = try archive(root, quota, "explicit", bytes: 10)
            let missing = root.appendingPathComponent("offline-volume")
            var cleared: [String] = []
            let result = HistoryAudioQuota.reconcile(
                roots: [root, missing], referencedPaths: [retained], requestedDeletions: [removed],
                limitBytes: 1, protectedPaths: [], clearReferences: { cleared.append($0) }
            )
            try check(result.measurementFailed, "offline storage must not be reported as zero")
            try check(result.unavailableRootPaths == [missing.path], "offline folder must be actionable")
            try check(cleared == [removed], "offline root must not block explicit deletion from available storage")
            try check(FileManager.default.fileExists(atPath: root.appendingPathComponent(retained).path), "unknown total usage must not trigger blind eviction")
        }
    }

    static func largeCatalogMaintainsQuotaAndReferences() throws {
        try withRoot { root, quota in
            let paths = (0..<1_000).map { "transcription/scale-\($0).wav" }
            for path in paths { try Data(repeating: 1, count: 10).write(to: root.appendingPathComponent(path)) }
            var cleared = Set<String>()
            let initial = HistoryAudioQuota.reconcile(
                roots: [root], referencedPaths: paths, limitBytes: nil, protectedPaths: [], clearReferences: { cleared.insert($0) }
            )
            try check(initial.fileCount == 1_000 && initial.totalBytes == 10_000 && cleared.isEmpty, "large catalog registration")
            let pruned = quota.enforce(limitBytes: 100) { cleared.insert($0) }
            try check(!pruned.measurementFailed && !pruned.cleanupFailed, "large catalog cleanup failed")
            try check(pruned.fileCount == 10 && pruned.totalBytes == 100 && cleared.count == 990, "large catalog eviction contract")
            let restarted = HistoryAudioQuota(rootURL: root).snapshot(limitBytes: 100)
            try check(restarted == pruned, "large catalog restart lost durable state")
        }
    }

    static func preferences() throws {
        let suite = "voxt-quota-test-\(UUID())"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.register(defaults: ["historyAudioStorageEnabled": false])
        HistoryAudioQuotaPreference.initialize(defaults: defaults, hasExistingData: false)
        try check(HistoryAudioQuotaPreference.limitBytes(defaults: defaults) == 100_000_000, "decimal default")
        try check(!defaults.bool(forKey: "historyAudioStorageEnabled"), "recording must remain off")
        defaults.set(-1, forKey: HistoryAudioQuotaPreference.limitKey)
        try check(HistoryAudioQuotaPreference.limitBytes(defaults: defaults) == 100_000_000, "invalid limit")
        defaults.removeObject(forKey: "historyAudioQuotaPolicyInitialized")
        defaults.removeObject(forKey: HistoryAudioQuotaPreference.limitKey)
        HistoryAudioQuotaPreference.initialize(defaults: defaults, hasExistingData: true)
        try check(HistoryAudioQuotaPreference.limitBytes(defaults: defaults) == nil, "legacy deletion needs consent")
        try check(defaults.bool(forKey: HistoryAudioQuotaPreference.migrationKey), "migration notice")
        defaults.removeObject(forKey: HistoryAudioQuotaPreference.limitKey)
        HistoryAudioQuotaPreference.initialize(defaults: defaults, hasExistingData: true)
        try check(HistoryAudioQuotaPreference.limitBytes(defaults: defaults) == nil, "interrupted legacy migration must not delete")
        try withRoot { root, quota in
            _ = try archive(root, quota, "unlimited", bytes: 100)
            let result = quota.enforce(limitBytes: nil) { _ in throw Failure.injected }
            try check(result.totalBytes == 100 && !result.cleanupFailed, "unlimited deletes")
        }
    }

    static func deletionFailure() throws {
        try withRoot { root, quota in
            _ = try archive(root, quota, "denied", bytes: 20)
            let failing = HistoryAudioQuota(rootURL: root, removeFile: { _, _ in throw Failure.injected })
            var cleared = false
            let result = failing.enforce(limitBytes: 10) { _ in cleared = true }
            try check(result.totalBytes == 20 && result.overLimit && result.cleanupFailed, "failed removal falsely frees bytes")
            try check(!cleared, "failed removal must preserve references")
        }
    }

    static func databaseFailureAndRestart() throws {
        try withRoot { root, quota in
            let path = try archive(root, quota, "retry", bytes: 20)
            let failed = quota.enforce(limitBytes: 10) { _ in throw Failure.injected }
            try check(failed.cleanupFailed && failed.totalBytes == 0, "database failure state")
            var cleared: [String] = []
            let restarted = HistoryAudioQuota(rootURL: root).enforce(limitBytes: nil) { cleared.append($0) }
            try check(cleared == [path] && !restarted.cleanupFailed, "restart must finish deletion journal even with Unlimited")
        }
    }

    static func duplicatesAndOrphans() throws {
        try withRoot { root, quota in
            let path = try archive(root, quota, "shared", bytes: 20)
            try quota.registerExisting(paths: [path, path])
            try check(quota.snapshot(limitBytes: nil).totalBytes == 20, "duplicate accounting")
            _ = try archive(root, quota, "orphan", bytes: 10, age: -1)
            try check(quota.snapshot(limitBytes: nil).totalBytes == 30, "managed unreferenced files counted")
            try Data(repeating: 1, count: 100).write(to: root.appendingPathComponent("unrelated.txt"))
            _ = quota.enforce(limitBytes: 0) { _ in }
            try check(FileManager.default.fileExists(atPath: root.appendingPathComponent("unrelated.txt").path), "unrelated file removed")
        }
    }

    static func pathsAndSymlinks() throws {
        try withRoot { root, quota in
            for path in ["../outside.wav", "/transcription/audio.wav", "transcription/../outside.wav", "transcription/nested/audio.wav"] {
                do { try quota.register(path: path); throw Failure.assertion("accepted unsafe path") }
                catch is HistoryAudioQuotaError {}
            }
            let target = root.appendingPathComponent("unrelated.wav")
            try Data(repeating: 5, count: 10).write(to: target)
            let link = root.appendingPathComponent("transcription/link.wav")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
            try quota.register(path: "transcription/link.wav")
            let result = quota.enforce(limitBytes: 0) { _ in throw Failure.injected }
            try check(result.measurementFailed, "symlink statistics must fail, not zero")
            let targetBytes = try Data(contentsOf: target)
            try check(targetBytes.count == 10, "symlink target changed")
        }
    }

    static func directorySwitchAndImports() throws {
        try withRoot { first, quota in
            try withRoot { second, secondQuota in
                let old = try archive(first, quota, "old-root", bytes: 10, age: 1)
                let new = try archive(second, secondQuota, "new-root", bytes: 10, age: 2)
                var cleared: [String] = []
                let result = HistoryAudioQuota.reconcile(
                    roots: [first, second], referencedPaths: [old, new], limitBytes: 10,
                    protectedPaths: [], clearReferences: { cleared.append($0) }
                )
                try check(result.totalBytes == 10 && cleared == [old], "global quota after directory switch")
                let source = second.appendingPathComponent("source.wav")
                try Data(repeating: 2, count: 5).write(to: source)
                try secondQuota.importArchive(from: source, path: "transcription/imported.wav")
                try check(secondQuota.snapshot(limitBytes: nil).totalBytes == 15, "imported file accounted before database")
                let collision = second.appendingPathComponent("collision.wav")
                try Data(repeating: 3, count: 7).write(to: collision)
                do {
                    try secondQuota.importArchive(from: collision, path: "transcription/imported.wav")
                    throw Failure.assertion("import replaced existing file")
                } catch is HistoryAudioQuotaError {}
                try check(FileManager.default.fileExists(atPath: collision.path), "failed import destroyed source")
            }
        }
    }

    static func logRotationAndFailure() throws {
        try withRoot { root, _ in
            let store = VoxtLogFileStore(maxFileBytes: 32, retainedArchiveCount: 1, baseDirectoryURL: root)
            for index in 0..<20 { store.append("technical event \(index)") }
            store.flush()
            let usage = store.storageUsage()
            try check(usage.bytes != nil && usage.bytes! <= 64 && !usage.cleanupFailed, "log quota exceeded: \(usage)")
            try check(store.latestLines(limit: 1) == ["technical event 19"], "logging stopped at quota")
            store.append(String(repeating: "x", count: 1_000))
            store.flush()
            try check(store.lastWriteFailed, "oversized event silently accepted")
            store.append("recovered")
            store.flush()
            try check(!store.lastWriteFailed && store.latestLines(limit: 1) == ["recovered"], "write recovery")
        }
        try withRoot { root, _ in
            try FileManager.default.createDirectory(at: root.appendingPathComponent("current.log"), withIntermediateDirectories: false)
            let store = VoxtLogFileStore(baseDirectoryURL: root)
            store.append("failure")
            store.flush()
            try check(store.lastWriteFailed && store.storageUsage().bytes == nil, "write/stat failure hidden as zero")
        }
    }

    static func rootAliasesAndDeferredDeletion() throws {
        try withRoot { root, quota in
            let path = try archive(root, quota, "retained", bytes: 20)
            let canonicalRoot = root.resolvingSymlinksInPath()
            let result = HistoryAudioQuota.reconcile(
                roots: [root, canonicalRoot], referencedPaths: [path], limitBytes: nil,
                protectedPaths: [], clearReferences: { _ in }
            )
            try check(!result.measurementFailed && result.totalBytes == 20, "directory aliases counted twice")
            var use: HistoryAudioFileUse? = HistoryAudioFileUse(url: root.appendingPathComponent(path))
            try quota.requestDeletion(paths: [path])
            try withExtendedLifetime(use) {
                let protected = quota.enforce(limitBytes: nil) { _ in }
                try check(protected.totalBytes == 20, "active explicit deletion")
            }
            use = nil
            let retry = HistoryAudioQuota(rootURL: canonicalRoot).enforce(limitBytes: nil) { _ in }
            try check(retry.totalBytes == 0, "deferred explicit deletion lost after restart")
        }
    }

    static func catalogAndParentFailures() throws {
        try withRoot { root, quota in
            try Data("corrupt".utf8).write(to: root.appendingPathComponent(".voxt-audio-catalog.json"))
            let result = quota.enforce(limitBytes: 0) { _ in }
            try check(result.measurementFailed, "corrupt catalog displayed as empty")
        }
        try withRoot { root, quota in
            try FileManager.default.removeItem(at: root.appendingPathComponent("transcription"))
            let outside = root.appendingPathComponent("unrelated")
            try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
            let audio = outside.appendingPathComponent("audio.wav")
            try Data(repeating: 8, count: 10).write(to: audio)
            try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("transcription"), withDestinationURL: outside)
            try quota.register(path: "transcription/audio.wav")
            let result = quota.enforce(limitBytes: 0) { _ in }
            try check(result.measurementFailed && FileManager.default.fileExists(atPath: audio.path), "parent symlink escaped")
        }
    }

    static func hostileImportAndLogRoots() throws {
        try withRoot { root, quota in
            let source = root.appendingPathComponent("source.wav")
            try Data(repeating: 3, count: 10).write(to: source)
            let alias = root.appendingPathComponent("alias.wav")
            try FileManager.default.linkItem(at: source, to: alias)
            do {
                try quota.importArchive(from: source, path: "transcription/hardlink.wav")
                throw Failure.assertion("hard-linked source accepted")
            } catch is HistoryAudioQuotaError {}
            let fifo = root.appendingPathComponent("fifo.wav")
            try check(mkfifo(fifo.path, 0o600) == 0, "FIFO fixture creation")
            do {
                try quota.importArchive(from: fifo, path: "transcription/fifo.wav")
                throw Failure.assertion("FIFO source accepted")
            } catch is HistoryAudioQuotaError {}
            let outside = root.appendingPathComponent("outside")
            try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: false)
            let original = Data("unrelated synthetic log".utf8)
            try original.write(to: outside.appendingPathComponent("current.log"))
            let logRoot = root.appendingPathComponent("logs")
            try FileManager.default.createSymbolicLink(at: logRoot, withDestinationURL: outside)
            let store = VoxtLogFileStore(baseDirectoryURL: logRoot)
            store.append("must not write")
            store.flush()
            try check(store.storageUsage().bytes == nil && store.lastWriteFailed, "symlink log root accepted")
            try check(store.latestLines(limit: 10).isEmpty, "symlink log root read")
            let persisted = try Data(contentsOf: outside.appendingPathComponent("current.log"))
            try check(persisted == original, "outside log changed")
            try check(!FileManager.default.fileExists(atPath: outside.appendingPathComponent("archive").path), "outside directory created")
        }
    }

    static let cases: [(String, () throws -> Void)] = [
        ("slowDeletionDoesNotBlockUnrelatedAudioUse", slowDeletionDoesNotBlockUnrelatedAudioUse),
        ("blockedOversizedPreservesSmallerArchives", blockedOversizedPreservesSmallerArchives),
        ("unavailableRootPreservesAudioButCompletesExplicitDeletion", unavailableRootPreservesAudioButCompletesExplicitDeletion),
        ("largeCatalogMaintainsQuotaAndReferences", largeCatalogMaintainsQuotaAndReferences),
        ("batchedDeletionPreservesConcurrentImportsAndFailedReferences", batchedDeletionPreservesConcurrentImportsAndFailedReferences),
        ("hostileImportAndLogRoots", hostileImportAndLogRoots), ("boundaries", boundaries), ("oldestAndOversized", oldestAndOversized),
        ("preferences", preferences), ("deletionFailure", deletionFailure),
        ("databaseFailureAndRestart", databaseFailureAndRestart),
        ("duplicatesAndOrphans", duplicatesAndOrphans), ("pathsAndSymlinks", pathsAndSymlinks),
        ("activeFilesAndConcurrentRegistration", activeFilesAndConcurrentRegistration),
        ("directorySwitchAndImports", directorySwitchAndImports), ("logRotationAndFailure", logRotationAndFailure),
        ("rootAliasesAndDeferredDeletion", rootAliasesAndDeferredDeletion), ("catalogAndParentFailures", catalogAndParentFailures)
    ]
}

#if VOXT_STORAGE_CONTRACT_TESTS
@main struct HistoryAudioQuotaContractRunner {
    static func main() {
        var failures = 0
        for (name, test) in HistoryAudioQuotaContracts.cases {
            do { try test(); print("PASS \(name)") }
            catch { failures += 1; print("FAIL \(name): \(error)") }
        }
        print("\(HistoryAudioQuotaContracts.cases.count) cases, \(failures) failures")
        if failures > 0 { exit(1) }
    }
}
#else
final class HistoryAudioQuotaTests: XCTestCase {
    func testSlowDeletionDoesNotBlockUnrelatedAudioUse() throws { try HistoryAudioQuotaContracts.slowDeletionDoesNotBlockUnrelatedAudioUse() }
    func testBlockedOversizedPreservesSmallerArchives() throws { try HistoryAudioQuotaContracts.blockedOversizedPreservesSmallerArchives() }
    func testUnavailableRootPreservesAudioButCompletesExplicitDeletion() throws { try HistoryAudioQuotaContracts.unavailableRootPreservesAudioButCompletesExplicitDeletion() }
    func testLargeCatalogMaintainsQuotaAndReferences() throws { try HistoryAudioQuotaContracts.largeCatalogMaintainsQuotaAndReferences() }
    func testBatchedDeletionPreservesConcurrentImportsAndFailedReferences() throws { try HistoryAudioQuotaContracts.batchedDeletionPreservesConcurrentImportsAndFailedReferences() }
    func testHostileImportAndLogRoots() throws { try HistoryAudioQuotaContracts.hostileImportAndLogRoots() }
    func testRootAliasesAndDeferredDeletion() throws { try HistoryAudioQuotaContracts.rootAliasesAndDeferredDeletion() }
    func testCatalogAndParentFailures() throws { try HistoryAudioQuotaContracts.catalogAndParentFailures() }
    func testDirectorySwitchAndImports() throws { try HistoryAudioQuotaContracts.directorySwitchAndImports() }
    func testLogRotationAndFailure() throws { try HistoryAudioQuotaContracts.logRotationAndFailure() }
    func testBoundaries() throws { try HistoryAudioQuotaContracts.boundaries() }
    func testOldestAndOversized() throws { try HistoryAudioQuotaContracts.oldestAndOversized() }
    func testPreferences() throws { try HistoryAudioQuotaContracts.preferences() }
    func testDeletionFailure() throws { try HistoryAudioQuotaContracts.deletionFailure() }
    func testDatabaseFailureAndRestart() throws { try HistoryAudioQuotaContracts.databaseFailureAndRestart() }
    func testDuplicatesAndOrphans() throws { try HistoryAudioQuotaContracts.duplicatesAndOrphans() }
    func testPathsAndSymlinks() throws { try HistoryAudioQuotaContracts.pathsAndSymlinks() }
    func testActiveFilesAndConcurrentRegistration() throws { try HistoryAudioQuotaContracts.activeFilesAndConcurrentRegistration() }
}
#endif
