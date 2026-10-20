import Darwin
import Foundation
#if !VOXT_STANDALONE_TESTS
@testable import Voxt
#endif

nonisolated extension HistoryDiagnosticSampleChecks {
    static func privatePermissions() throws {
        try withDirectory { root in
            let output = try HistoryDiagnosticSampleExporter.export(request(text: "private"), to: root)
            let directoryPermissions = try FileManager.default.attributesOfItem(atPath: output.path)[.posixPermissions] as! NSNumber
            try require(directoryPermissions.intValue == 0o700, "Public sample directory")
            for file in try files(output) {
                let permissions = try FileManager.default.attributesOfItem(atPath: output.appendingPathComponent(file).path)[.posixPermissions] as! NSNumber
                try require(permissions.intValue == 0o600, "Public sensitive file")
            }
        }
    }

    static func installACL(_ entry: String, on directory: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/chmod")
        process.arguments = ["+a", entry, directory.path]
        try process.run()
        process.waitUntilExit()
        try require(process.terminationStatus == 0, "Could not install fixture ACL")
    }

    static func assertEmptyACL(_ url: URL) throws {
        guard let permissions = acl_get_file(url.path, ACL_TYPE_EXTENDED) else {
            let missingACLError = errno
            try require(missingACLError == ENOENT && FileManager.default.fileExists(atPath: url.path), "Could not inspect sample ACL")
            return
        }
        defer { acl_free(UnsafeMutableRawPointer(permissions)) }
        var entry: acl_entry_t?
        let result = acl_get_entry(permissions, ACL_FIRST_ENTRY.rawValue, &entry)
        try require(result == -1 && errno == EINVAL, "Sample inherited permissions from destination")
    }

    static func inheritedACLIsolation() throws {
        try withDirectory { root in
            try installACL("everyone allow read,write,execute,file_inherit,directory_inherit", on: root)
            let audio = try source(in: root)
            var inspectedStaging = false
            let output = try HistoryDiagnosticSampleExporter.export(request(text: "private", audio: audio, includeAudio: true), to: root) { phase in
                if phase == .stagingCreated || phase == .beforePublish {
                    let directories = try files(root).filter { $0.hasPrefix(".voxt-sample-") }
                    try require(directories.count == 1, "Expected one staging directory")
                    let staging = root.appendingPathComponent(directories.first!)
                    try assertEmptyACL(staging)
                    for name in try files(staging) { try assertEmptyACL(staging.appendingPathComponent(name)) }
                    inspectedStaging = true
                }
            }
            try require(inspectedStaging, "Staging permissions were not inspected")
            try assertEmptyACL(output)
            for name in try files(output) { try assertEmptyACL(output.appendingPathComponent(name)) }
            guard let parentPermissions = acl_get_file(root.path, ACL_TYPE_EXTENDED) else {
                throw Failure.assertion("Could not inspect destination ACL")
            }
            defer { acl_free(UnsafeMutableRawPointer(parentPermissions)) }
            var entry: acl_entry_t?
            try require(acl_get_entry(parentPermissions, ACL_FIRST_ENTRY.rawValue, &entry) == 0, "Export changed the destination ACL")
        }
    }

    static func permissionFailureCleanup() throws {
        try withDirectory { root in
            var stagingURL: URL?
            defer {
                if let stagingURL { _ = chflags(stagingURL.path, 0) }
            }
            try expectFailure({
                _ = try HistoryDiagnosticSampleExporter.export(request(text: "private"), to: root) { phase in
                    if phase == .securingStaging {
                        stagingURL = root.appendingPathComponent(try files(root).first!)
                        try require(chflags(stagingURL!.path, UInt32(UF_IMMUTABLE)) == 0, "Could not lock fixture permissions")
                    } else if phase == .beforeCleanup, let stagingURL {
                        let contents = try files(stagingURL)
                        try require(contents.isEmpty, "Sensitive content was written before private permissions were established")
                        try require(chflags(stagingURL.path, 0) == 0, "Could not unlock fixture for cleanup")
                    }
                }
            }, matching: .privatePermissionsUnavailable)
            let remaining = try files(root)
            try require(remaining.isEmpty, "Failed permission enforcement left a partial sample")
        }
    }

}
