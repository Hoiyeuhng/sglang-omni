import Darwin
import Foundation
#if !VOXT_STORAGE_CONTRACT_TESTS
@testable import Voxt
#endif

nonisolated extension HistoryAudioQuotaContracts {
    static func slowDeletionDoesNotBlockUnrelatedAudioUse() throws {
        try withRoot { root, quota in
            let removed = try archive(root, quota, "removed", bytes: 10, age: 1)
            let retained = try archive(root, quota, "retained", bytes: 10, age: 2)
            let deletionStarted = DispatchSemaphore(value: 0)
            let deletionMayFinish = DispatchSemaphore(value: 0)
            let deletionFinished = DispatchSemaphore(value: 0)
            let retainedUseAcquired = DispatchSemaphore(value: 0)
            let removedUseAcquired = DispatchSemaphore(value: 0)
            let testedQuota = HistoryAudioQuota(rootURL: root, removeFile: { directory, name in
                deletionStarted.signal()
                guard deletionMayFinish.wait(timeout: .now() + 5) == .success else { throw Failure.injected }
                guard unlinkat(directory, name, 0) == 0 else { throw Failure.injected }
            })
            DispatchQueue.global().async {
                _ = testedQuota.enforce(limitBytes: 10) { _ in }
                deletionFinished.signal()
            }
            let didStart = deletionStarted.wait(timeout: .now() + 5) == .success
            DispatchQueue.global().async {
                let use = HistoryAudioFileUse(url: root.appendingPathComponent(retained))
                retainedUseAcquired.signal()
                withExtendedLifetime(use) {}
            }
            DispatchQueue.global().async {
                let use = HistoryAudioFileUse(url: root.appendingPathComponent(removed))
                removedUseAcquired.signal()
                withExtendedLifetime(use) {}
            }
            let retainedWasAvailable = retainedUseAcquired.wait(timeout: .now() + 1) == .success
            let removedWaited = removedUseAcquired.wait(timeout: .now() + 0.05) == .timedOut
            deletionMayFinish.signal()
            let didFinish = deletionFinished.wait(timeout: .now() + 5) == .success
            let removedEventuallyAcquired = removedUseAcquired.wait(timeout: .now() + 5) == .success
            try check(didStart && didFinish, "slow deletion did not finish")
            try check(retainedWasAvailable, "an unrelated retained file could not acquire a lease during slow deletion")
            try check(removedWaited && removedEventuallyAcquired, "same-file lease did not wait for deletion reservation")
            try check(FileManager.default.fileExists(atPath: root.appendingPathComponent(retained).path), "retained file was deleted")
            try check(!FileManager.default.fileExists(atPath: root.appendingPathComponent(removed).path), "selected file was not deleted")
        }
    }

    static func activeFilesAndConcurrentRegistration() throws {
        try withRoot { root, quota in
            let path = try archive(root, quota, "playing", bytes: 20)
            var lease: HistoryAudioFileUse? = HistoryAudioFileUse(url: root.appendingPathComponent(path))
            try withExtendedLifetime(lease) {
                let result = quota.enforce(limitBytes: 10) { _ in }
                try check(result.overLimit && result.totalBytes == 20, "active audio deleted")
            }
            lease = nil
            try check(quota.enforce(limitBytes: 10) { _ in }.totalBytes == 0, "released lease not reclaimed")
            DispatchQueue.concurrentPerform(iterations: 30) { index in
                do { try quota.register(path: "transcription/concurrent-\(index).wav") }
                catch { fatalError("registration failed: \(error)") }
            }
            try check(!quota.snapshot(limitBytes: nil).measurementFailed, "concurrent catalog corrupt")
        }
    }

    static func batchedDeletionPreservesConcurrentImportsAndFailedReferences() throws {
        try withRoot { root, quota in
            let first = try archive(root, quota, "first", bytes: 10, age: 1)
            let second = try archive(root, quota, "second", bytes: 10, age: 2)
            var imported: String?
            let result = quota.enforce(limitBytes: 0) { path in
                if path == first { throw Failure.injected }
                if path == second { imported = try archive(root, quota, "concurrent", bytes: 4, age: 3) }
            }
            try check(result.cleanupFailed && result.totalBytes == 4, "batch must retain concurrent imports and report failed reference cleanup")
            var retried: [String] = []
            let restarted = HistoryAudioQuota(rootURL: root).enforce(limitBytes: nil) { retried.append($0) }
            try check(retried == [first] && restarted.totalBytes == 4 && !restarted.cleanupFailed, "batch restart must retry only unfinished references")
            try check(imported != nil && FileManager.default.fileExists(atPath: root.appendingPathComponent(imported!).path), "concurrent imported file lost")
        }
    }

}
