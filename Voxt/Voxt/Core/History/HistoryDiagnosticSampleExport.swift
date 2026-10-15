import Darwin
import Foundation

nonisolated enum HistoryDiagnosticSampleError: Error, Equatable {
    case noContentSelected
    case audioUnavailable
    case unsafeAudioPath
    case destinationConflict
    case ioFailure
    case sourceChanged
    case privatePermissionsUnavailable
    case cleanupFailed(URL)
}

nonisolated final class HistoryDiagnosticAudioSource: @unchecked Sendable {
    let byteCount: Int64
    private let descriptor: Int32
    private let modificationSeconds: Int
    private let modificationNanoseconds: Int

    init(rootURL: URL, relativePath: String, resolvedAudioURL: URL? = nil) throws {
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        let folders = ["transcription", "translation", "rewrite", "transcript"]
        guard components.count == 2, folders.contains(components[0]),
              !components[1].isEmpty, components[1] != ".", components[1] != "..",
              !relativePath.contains("\0"), components[1].hasSuffix(".wav") else {
            throw HistoryDiagnosticSampleError.unsafeAudioPath
        }
        if let resolvedAudioURL {
            guard resolvedAudioURL.lastPathComponent == components[1],
                  resolvedAudioURL.deletingLastPathComponent().lastPathComponent == components[0] else {
                throw HistoryDiagnosticSampleError.unsafeAudioPath
            }
        }
        let selectedRootURL = resolvedAudioURL?.deletingLastPathComponent().deletingLastPathComponent() ?? rootURL
        let rootDescriptor = open(selectedRootURL.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard rootDescriptor >= 0 else { throw HistoryDiagnosticSampleError.audioUnavailable }
        defer { close(rootDescriptor) }
        let folderDescriptor = openat(rootDescriptor, components[0], O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard folderDescriptor >= 0 else { throw HistoryDiagnosticSampleError.audioUnavailable }
        defer { close(folderDescriptor) }
        let audioDescriptor = openat(folderDescriptor, components[1], O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
        guard audioDescriptor >= 0 else { throw HistoryDiagnosticSampleError.audioUnavailable }
        var attributes = stat()
        guard fstat(audioDescriptor, &attributes) == 0, attributes.st_mode & S_IFMT == S_IFREG else {
            close(audioDescriptor)
            throw HistoryDiagnosticSampleError.unsafeAudioPath
        }
        descriptor = audioDescriptor
        byteCount = attributes.st_size
        modificationSeconds = attributes.st_mtimespec.tv_sec
        modificationNanoseconds = attributes.st_mtimespec.tv_nsec
    }

    var previewURL: URL {
        URL(fileURLWithPath: "/dev/fd/\(descriptor)")
    }

    deinit { close(descriptor) }

    func copy(to outputDescriptor: Int32, checkpoint: (HistoryDiagnosticSampleExporter.Checkpoint) throws -> Void) throws {
        var buffer = [UInt8](repeating: 0, count: 256 * 1024)
        var offset: Int64 = 0
        while offset < byteCount {
            try Task.checkCancellation()
            try checkpoint(.copyingAudio)
            let count = pread(descriptor, &buffer, min(buffer.count, Int(byteCount - offset)), offset)
            guard count > 0 else { throw HistoryDiagnosticSampleError.ioFailure }
            try buffer.withUnsafeBytes { bytes in
                var written = 0
                while written < count {
                    let amount = Darwin.write(outputDescriptor, bytes.baseAddress!.advanced(by: written), count - written)
                    guard amount > 0 else { throw HistoryDiagnosticSampleError.ioFailure }
                    written += amount
                }
            }
            offset += Int64(count)
        }
        var attributes = stat()
        guard fstat(descriptor, &attributes) == 0,
              attributes.st_size == byteCount,
              attributes.st_mtimespec.tv_sec == modificationSeconds,
              attributes.st_mtimespec.tv_nsec == modificationNanoseconds else {
            throw HistoryDiagnosticSampleError.sourceChanged
        }
    }
}

nonisolated struct HistoryDiagnosticSampleRequest: Sendable {
    let historyText: String?
    let audio: HistoryDiagnosticAudioSource?
    let includeAudio: Bool
    let expectedTranscription: String?
    let appVersion: String

    var fileNames: [String] {
        var names = ["manifest.json"]
        if historyText != nil { names.append("history-text.txt") }
        if includeAudio { names.append("recording.wav") }
        if let expectedTranscription, !expectedTranscription.isEmpty { names.append("expected-transcription.txt") }
        return names
    }
}

nonisolated enum HistoryDiagnosticSampleExporter {
    enum Checkpoint: Sendable {
        case securingStaging
        case stagingCreated
        case copyingAudio
        case beforePublish
        case beforeCleanup
    }

    private enum TextOrigin: String, Encodable {
        case storedFinal = "stored_final"
    }

    private struct Manifest: Encodable {
        let schemaVersion = 2
        let appVersion: String
        let files: [String]
        let audioBytes: Int64?
        let textOrigin: TextOrigin?
    }

    static func export(
        _ request: HistoryDiagnosticSampleRequest,
        to destinationURL: URL,
        folderName: String = "voxt-diagnostic-sample-\(UUID().uuidString)",
        checkpoint: (Checkpoint) throws -> Void = { _ in }
    ) throws -> URL {
        guard request.historyText != nil || request.includeAudio else {
            throw HistoryDiagnosticSampleError.noContentSelected
        }
        guard !request.includeAudio || request.audio != nil else {
            throw HistoryDiagnosticSampleError.audioUnavailable
        }
        guard !folderName.isEmpty, folderName != ".", folderName != "..",
              !folderName.contains("/"), !folderName.contains("\0") else {
            throw HistoryDiagnosticSampleError.destinationConflict
        }
        try Task.checkCancellation()
        let destinationDescriptor = open(destinationURL.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard destinationDescriptor >= 0 else { throw HistoryDiagnosticSampleError.ioFailure }
        defer { close(destinationDescriptor) }
        let stagingName = ".voxt-sample-\(UUID().uuidString).partial"
        let stagingURL = destinationURL.appendingPathComponent(stagingName, isDirectory: true)
        guard mkdirat(destinationDescriptor, stagingName, 0o700) == 0 else {
            throw HistoryDiagnosticSampleError.ioFailure
        }
        let stagingDescriptor = openat(destinationDescriptor, stagingName, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard stagingDescriptor >= 0 else {
            if unlinkat(destinationDescriptor, stagingName, AT_REMOVEDIR) != 0 {
                throw HistoryDiagnosticSampleError.cleanupFailed(stagingURL)
            }
            throw HistoryDiagnosticSampleError.ioFailure
        }
        defer { close(stagingDescriptor) }
        do {
            try checkpoint(.securingStaging)
            try enforcePrivatePermissions(stagingDescriptor, mode: 0o700)
            try checkpoint(.stagingCreated)
            if let historyText = request.historyText {
                try write(Data(historyText.utf8), named: "history-text.txt", directoryDescriptor: stagingDescriptor)
            }
            if let expected = request.expectedTranscription, !expected.isEmpty {
                try write(Data(expected.utf8), named: "expected-transcription.txt", directoryDescriptor: stagingDescriptor)
            }
            if request.includeAudio, let audio = request.audio {
                let descriptor = openat(stagingDescriptor, "recording.wav", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
                guard descriptor >= 0 else { throw HistoryDiagnosticSampleError.ioFailure }
                defer { close(descriptor) }
                try enforcePrivatePermissions(descriptor, mode: 0o600)
                try audio.copy(to: descriptor, checkpoint: checkpoint)
                guard fsync(descriptor) == 0 else { throw HistoryDiagnosticSampleError.ioFailure }
            }
            let version = request.appVersion
            let safeVersion = !version.isEmpty && version.count <= 40 && version.allSatisfy({ "0123456789.".contains($0) })
                ? version : "unknown"
            let manifest = Manifest(
                appVersion: safeVersion,
                files: request.fileNames,
                audioBytes: request.includeAudio ? request.audio?.byteCount : nil,
                textOrigin: request.historyText == nil ? nil : .storedFinal
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try write(encoder.encode(manifest), named: "manifest.json", directoryDescriptor: stagingDescriptor)
            try checkpoint(.beforePublish)
            guard fsync(stagingDescriptor) == 0 else { throw HistoryDiagnosticSampleError.ioFailure }
            try Task.checkCancellation()
            guard renameatx_np(destinationDescriptor, stagingName, destinationDescriptor, folderName, UInt32(RENAME_EXCL)) == 0 else {
                throw errno == EEXIST ? HistoryDiagnosticSampleError.destinationConflict : HistoryDiagnosticSampleError.ioFailure
            }
            return destinationURL.appendingPathComponent(folderName, isDirectory: true)
        } catch {
            do {
                try checkpoint(.beforeCleanup)
                for name in ["history-text.txt", "expected-transcription.txt", "recording.wav", "manifest.json"] {
                    if unlinkat(stagingDescriptor, name, 0) != 0 && errno != ENOENT {
                        throw HistoryDiagnosticSampleError.cleanupFailed(stagingURL)
                    }
                }
                guard unlinkat(destinationDescriptor, stagingName, AT_REMOVEDIR) == 0 else {
                    throw HistoryDiagnosticSampleError.cleanupFailed(stagingURL)
                }
            } catch {
                throw HistoryDiagnosticSampleError.cleanupFailed(stagingURL)
            }
            throw error
        }
    }

    private static func enforcePrivatePermissions(_ descriptor: Int32, mode: mode_t) throws {
        guard let permissions = acl_init(0) else {
            throw HistoryDiagnosticSampleError.privatePermissionsUnavailable
        }
        defer { acl_free(UnsafeMutableRawPointer(permissions)) }
        guard acl_set_fd_np(descriptor, permissions, ACL_TYPE_EXTENDED) == 0,
              fchmod(descriptor, mode) == 0 else {
            throw HistoryDiagnosticSampleError.privatePermissionsUnavailable
        }
        if let effectivePermissions = acl_get_fd_np(descriptor, ACL_TYPE_EXTENDED) {
            defer { acl_free(UnsafeMutableRawPointer(effectivePermissions)) }
            var entry: acl_entry_t?
            guard acl_valid(effectivePermissions) == 0,
                  acl_get_entry(effectivePermissions, ACL_FIRST_ENTRY.rawValue, &entry) == -1,
                  errno == EINVAL else {
                throw HistoryDiagnosticSampleError.privatePermissionsUnavailable
            }
        } else if errno != ENOENT {
            throw HistoryDiagnosticSampleError.privatePermissionsUnavailable
        }
        // note (Codex): Darwin returns ENOENT for a valid descriptor with no extended ACL.
        var attributes = stat()
        guard fstat(descriptor, &attributes) == 0,
              attributes.st_mode & 0o7777 == mode,
              attributes.st_uid == geteuid() else {
            throw HistoryDiagnosticSampleError.privatePermissionsUnavailable
        }
    }

    private static func write(_ contents: Data, named name: String, directoryDescriptor: Int32) throws {
        let descriptor = openat(directoryDescriptor, name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw HistoryDiagnosticSampleError.ioFailure }
        defer { close(descriptor) }
        try enforcePrivatePermissions(descriptor, mode: 0o600)
        try contents.withUnsafeBytes { bytes in
            var written = 0
            while written < bytes.count {
                try Task.checkCancellation()
                let amount = Darwin.write(descriptor, bytes.baseAddress!.advanced(by: written), bytes.count - written)
                guard amount > 0 else { throw HistoryDiagnosticSampleError.ioFailure }
                written += amount
            }
        }
        guard fsync(descriptor) == 0 else { throw HistoryDiagnosticSampleError.ioFailure }
    }
}
