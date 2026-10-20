import Darwin
import Foundation

extension HistoryAudioQuota {
    private nonisolated struct Candidate {
        let quota: HistoryAudioQuota
        let archive: Archive
        let bytes: Int64
    }

    nonisolated func enforce(
        limitBytes: Int64?,
        protectedPaths: Set<String> = [],
        clearReferences: (String) throws -> Void
    ) -> HistoryAudioQuotaSnapshot {
        Self.reconcile(
            quotas: [self], referencedPaths: [], requestedDeletions: [], limitBytes: limitBytes,
            protectedPaths: protectedPaths, clearReferences: clearReferences
        )
    }

    nonisolated static func reconcile(
        roots: [URL],
        referencedPaths: [String],
        requestedDeletions: Set<String> = [],
        limitBytes: Int64?,
        protectedPaths: Set<String>,
        clearReferences: (String) throws -> Void
    ) -> HistoryAudioQuotaSnapshot {
        var seenRoots = Set<String>()
        let quotas = roots.filter { seenRoots.insert($0.resolvingSymlinksInPath().path).inserted }
            .map { HistoryAudioQuota(rootURL: $0) }
        return reconcile(
            quotas: quotas, referencedPaths: referencedPaths, requestedDeletions: requestedDeletions,
            limitBytes: limitBytes, protectedPaths: protectedPaths, clearReferences: clearReferences
        )
    }

    private nonisolated static func reconcile(
        quotas: [HistoryAudioQuota],
        referencedPaths: [String],
        requestedDeletions: Set<String>,
        limitBytes: Int64?,
        protectedPaths: Set<String>,
        clearReferences: (String) throws -> Void
    ) -> HistoryAudioQuotaSnapshot {
        var candidates: [Candidate] = []
        var result = HistoryAudioQuotaSnapshot()
        let requestedPaths = Set(referencedPaths).union(requestedDeletions)
        for quota in quotas {
            do {
                let directory = try quota.openRoot()
                defer { close(directory) }
                lock.lock()
                defer { lock.unlock() }
                var archives = try quota.readCatalog(directory: directory)
                let knownPaths = Set(archives.map(\.path))
                var changed = false
                for path in requestedPaths.subtracting(knownPaths) {
                    if let attributes = try quota.attributes(path: path, root: directory) {
                        archives.append(Archive(path: path, createdAt: Date(timeIntervalSince1970: Double(attributes.st_mtimespec.tv_sec))))
                        changed = true
                    }
                }
                for index in archives.indices where requestedDeletions.contains(archives[index].path) && !archives[index].deletionPending {
                    archives[index].deletionPending = true
                    changed = true
                }
                let measured = try archives.map { archive in
                    Candidate(quota: quota, archive: archive, bytes: Int64(try quota.attributes(path: archive.path, root: directory)?.st_size ?? 0))
                }
                if changed { try quota.writeCatalog(archives, directory: directory) }
                candidates += measured
            } catch {
                result.unavailableRootPaths.append(quota.rootURL.path)
                result.measurementFailed = true
            }
        }
        let knownPaths = Set(candidates.map { $0.archive.path })
        guard knownPaths.count == candidates.count else {
            return HistoryAudioQuotaSnapshot(measurementFailed: true)
        }
        if !result.measurementFailed {
            for path in requestedDeletions.subtracting(knownPaths) {
                do { try clearReferences(path) }
                catch { result.cleanupFailed = true }
            }
        }
        candidates.sort {
            $0.archive.createdAt == $1.archive.createdAt
                ? $0.archive.path < $1.archive.path : $0.archive.createdAt < $1.archive.createdAt
        }
        let eligible = candidates.filter { !protectedPaths.contains($0.archive.path) }
        var removed = Set<String>()
        let pending = deleteCandidates(eligible.filter { $0.archive.deletionPending }, bytesToFree: nil, clearReferences: clearReferences)
        removed.formUnion(pending.removed)
        result.cleanupFailed = result.cleanupFailed || pending.failed
        var remainingBytes = candidates.reduce(Int64(0)) { $0 + (removed.contains($1.archive.path) ? 0 : $1.bytes) }
        if !result.measurementFailed, let limitBytes, remainingBytes > limitBytes {
            let oversized = deleteCandidates(
                eligible.filter { $0.bytes > limitBytes && !removed.contains($0.archive.path) && !$0.archive.deletionPending },
                bytesToFree: nil, clearReferences: clearReferences
            )
            removed.formUnion(oversized.removed)
            result.cleanupFailed = result.cleanupFailed || oversized.failed
            remainingBytes = candidates.reduce(Int64(0)) { $0 + (removed.contains($1.archive.path) ? 0 : $1.bytes) }
            // Note (Codex): a blocked oversized file cannot be made to fit by deleting smaller recordings.
            let oversizedRemains = candidates.contains { $0.bytes > limitBytes && !removed.contains($0.archive.path) }
            if !oversizedRemains, remainingBytes > limitBytes {
                let oldest = deleteCandidates(
                    eligible.filter { !removed.contains($0.archive.path) && !$0.archive.deletionPending && $0.bytes > 0 },
                    bytesToFree: remainingBytes - limitBytes, clearReferences: clearReferences
                )
                result.cleanupFailed = result.cleanupFailed || oldest.failed
            }
        }
        for quota in quotas {
            let snapshot = quota.snapshot(limitBytes: nil)
            result.fileCount += snapshot.fileCount
            result.totalBytes += snapshot.totalBytes
            result.measurementFailed = result.measurementFailed || snapshot.measurementFailed
            if snapshot.measurementFailed, !result.unavailableRootPaths.contains(quota.rootURL.path) {
                result.unavailableRootPaths.append(quota.rootURL.path)
            }
        }
        result.overLimit = limitBytes.map { result.totalBytes > $0 } ?? false
        return result
    }

    private nonisolated static func deleteCandidates(
        _ candidates: [Candidate], bytesToFree: Int64?, clearReferences: (String) throws -> Void
    ) -> (removed: Set<String>, failed: Bool) {
        guard !candidates.isEmpty else { return ([], false) }
        var deleted: [Candidate] = []
        var failed = false
        let candidatePaths = candidates.map { candidate in
            candidate.quota.rootURL.appendingPathComponent(candidate.archive.path).resolvingSymlinksInPath().standardizedFileURL.path
        }
        HistoryAudioFileUse.withUnusedURLs(candidatePaths.map { URL(fileURLWithPath: $0) }, select: { unusedPaths in
            var selectedPaths = Set<String>()
            var selectedBytes: Int64 = 0
            for (candidate, path) in zip(candidates, candidatePaths) {
                guard unusedPaths.contains(path) else { continue }
                if let bytesToFree, selectedBytes >= bytesToFree { break }
                selectedPaths.insert(path)
                selectedBytes += candidate.bytes
            }
            return selectedPaths
        }) { selectedPaths in
            let selected = zip(candidates, candidatePaths).filter { selectedPaths.contains($0.1) }.map { $0.0 }
            for group in Dictionary(grouping: selected, by: { $0.quota.rootURL.path }).values {
                guard let quota = group.first?.quota else { continue }
                do {
                    let directory = try quota.openRoot()
                    defer { close(directory) }
                    do {
                        lock.lock()
                        defer { lock.unlock() }
                        var archives = try quota.readCatalog(directory: directory)
                        let paths = Set(group.map { $0.archive.path })
                        for index in archives.indices where paths.contains(archives[index].path) { archives[index].deletionPending = true }
                        try quota.writeCatalog(archives, directory: directory)
                    }
                    for candidate in group {
                        do {
                            if try quota.attributes(path: candidate.archive.path, root: directory) != nil {
                                try quota.withParent(path: candidate.archive.path, root: directory) { parent, name in
                                    try quota.removeFile(parent, name)
                                }
                            }
                            deleted.append(candidate)
                        } catch { failed = true }
                    }
                } catch { failed = true }
            }
        }

        var cleared: [Candidate] = []
        for candidate in deleted {
            do { try clearReferences(candidate.archive.path); cleared.append(candidate) }
            catch { failed = true }
        }
        lock.lock()
        defer { lock.unlock() }
        for group in Dictionary(grouping: cleared, by: { $0.quota.rootURL.path }).values {
            guard let quota = group.first?.quota else { continue }
            do {
                let directory = try quota.openRoot()
                defer { close(directory) }
                let removedPaths = Set(group.map { $0.archive.path })
                let archives = try quota.readCatalog(directory: directory).filter { !removedPaths.contains($0.path) }
                try quota.writeCatalog(archives, directory: directory)
            } catch { failed = true }
        }
        return (Set(deleted.map { $0.archive.path }), failed)
    }

}
