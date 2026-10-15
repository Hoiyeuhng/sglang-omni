import AVFoundation
import Darwin
import Foundation
#if !VOXT_STANDALONE_TESTS
@testable import Voxt
#endif

nonisolated enum HistoryDiagnosticSampleChecks {
    enum Failure: Error { case assertion(String), injected }

    static let cases: [(String, () throws -> Void)] = [
        ("text-only", textOnly), ("audio-only", audioOnly), ("both-and-expected", bothAndExpected),
        ("nothing-selected", nothingSelected), ("missing-audio", missingAudio),
        ("destination-conflicts", destinationConflicts), ("copy-failure-cleanup", copyFailureCleanup),
        ("cancel-cleanup", cancelCleanup), ("publish-failure-cleanup", publishFailureCleanup),
        ("cleanup-failure-disclosure", cleanupFailureDisclosure), ("source-path-boundaries", sourcePathBoundaries),
        ("source-unlinked", sourceUnlinked), ("source-mutated", sourceMutated),
        ("manifest-privacy", manifestPrivacy), ("private-permissions", privatePermissions),
        ("synthetic-audio-roundtrip", syntheticAudioRoundtrip), ("retained-audio-preview", retainedAudioPreview)
    ]

    static func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
        guard condition() else { throw Failure.assertion(message) }
    }

    static func withDirectory(_ test: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-sample-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: directory) }
        try test(directory)
    }

    static func source(in root: URL, contents: Data = Data(repeating: 7, count: 600_000)) throws -> HistoryDiagnosticAudioSource {
        let folder = root.appendingPathComponent("transcription")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try contents.write(to: folder.appendingPathComponent("sample.wav"))
        return try HistoryDiagnosticAudioSource(rootURL: root, relativePath: "transcription/sample.wav")
    }

    static func request(text: String? = nil, audio: HistoryDiagnosticAudioSource? = nil, includeAudio: Bool = false, expected: String? = nil, version: String = "1.2.3") -> HistoryDiagnosticSampleRequest {
        HistoryDiagnosticSampleRequest(transcription: text, audio: audio, includeAudio: includeAudio, expectedTranscription: expected, appVersion: version)
    }

    static func files(_ directory: URL) throws -> Set<String> {
        Set(try FileManager.default.contentsOfDirectory(atPath: directory.path))
    }

    static func expectFailure(_ operation: () throws -> Void, matching expected: HistoryDiagnosticSampleError? = nil) throws {
        do {
            try operation()
        } catch {
            if let expected {
                try require(error as? HistoryDiagnosticSampleError == expected, "Wrong failure: \(error)")
            }
            return
        }
        throw Failure.assertion("Operation unexpectedly succeeded")
    }

    static func textOnly() throws {
        try withDirectory { root in
            let audio = try source(in: root)
            let output = try HistoryDiagnosticSampleExporter.export(request(text: "chosen text", audio: audio), to: root)
            let outputFiles = try files(output)
            try require(outputFiles == ["manifest.json", "transcription.txt"], "Unselected audio escaped")
            let text = try String(contentsOf: output.appendingPathComponent("transcription.txt"), encoding: .utf8)
            try require(text == "chosen text", "Text changed")
        }
    }

    static func audioOnly() throws {
        try withDirectory { root in
            let contents = Data(repeating: 9, count: 600_000)
            let audio = try source(in: root, contents: contents)
            let output = try HistoryDiagnosticSampleExporter.export(request(audio: audio, includeAudio: true), to: root)
            let outputFiles = try files(output)
            try require(outputFiles == ["manifest.json", "recording.wav"], "Unselected text escaped")
            let copied = try Data(contentsOf: output.appendingPathComponent("recording.wav"))
            try require(copied == contents, "Copy changed bytes")
        }
    }

    static func bothAndExpected() throws {
        try withDirectory { root in
            let audio = try source(in: root)
            let selected = request(text: "actual", audio: audio, includeAudio: true, expected: "expected")
            let output = try HistoryDiagnosticSampleExporter.export(selected, to: root)
            let outputFiles = try files(output)
            try require(outputFiles == Set(selected.fileNames), "Preview and export differ")
            let expected = try String(contentsOf: output.appendingPathComponent("expected-transcription.txt"), encoding: .utf8)
            try require(expected == "expected", "Expected transcript missing")
        }
    }

    static func nothingSelected() throws {
        try withDirectory { root in
            try expectFailure({ _ = try HistoryDiagnosticSampleExporter.export(request(expected: "not sufficient"), to: root) }, matching: .noContentSelected)
            let remaining = try files(root)
            try require(remaining.isEmpty, "Empty selection wrote files")
        }
    }

    static func missingAudio() throws {
        try withDirectory { root in
            try expectFailure({ _ = try HistoryDiagnosticSampleExporter.export(request(text: "private", includeAudio: true), to: root) }, matching: .audioUnavailable)
            let remaining = try files(root)
            try require(remaining.isEmpty, "Missing audio published partial text")
        }
    }

    static func destinationConflicts() throws {
        try withDirectory { root in
            for name in ["existing-file", "existing-directory", "existing-link"] {
                let url = root.appendingPathComponent(name)
                if name == "existing-directory" {
                    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
                    try Data("original".utf8).write(to: url.appendingPathComponent("sentinel"))
                } else if name == "existing-link" {
                    try FileManager.default.createSymbolicLink(at: url, withDestinationURL: root.appendingPathComponent("existing-file"))
                } else {
                    try Data("original".utf8).write(to: url)
                }
                try expectFailure({ _ = try HistoryDiagnosticSampleExporter.export(request(text: "replacement"), to: root, folderName: name) }, matching: .destinationConflict)
            }
            let text = try String(contentsOf: root.appendingPathComponent("existing-file"), encoding: .utf8)
            let sentinel = try String(contentsOf: root.appendingPathComponent("existing-directory/sentinel"), encoding: .utf8)
            let remaining = try files(root)
            try require(text == "original" && sentinel == "original", "Existing export overwritten")
            try require(remaining == ["existing-file", "existing-directory", "existing-link"], "Conflict left partial files")
        }
    }

    static func copyFailureCleanup() throws {
        try withDirectory { root in
            let audio = try source(in: root)
            try expectFailure {
                _ = try HistoryDiagnosticSampleExporter.export(request(text: "sensitive", audio: audio, includeAudio: true), to: root) { phase in
                    if phase == .copyingAudio { throw Failure.injected }
                }
            }
            let remaining = try files(root)
            try require(remaining == ["transcription"], "Copy failure leaked partial text")
        }
    }

    static func cancelCleanup() throws {
        try withDirectory { root in
            let audio = try source(in: root)
            for cancellationPhase: HistoryDiagnosticSampleExporter.Checkpoint in [.stagingCreated, .copyingAudio, .beforePublish] {
                var wasCancelled = false
                do {
                    _ = try HistoryDiagnosticSampleExporter.export(request(text: "private", audio: audio, includeAudio: true), to: root) { phase in
                        if phase == cancellationPhase { throw CancellationError() }
                    }
                } catch is CancellationError { wasCancelled = true }
                try require(wasCancelled, "Cancellation was lost")
                let remaining = try files(root)
                try require(remaining == ["transcription"], "Cancellation left sensitive artifacts")
            }
        }
    }

    static func publishFailureCleanup() throws {
        try withDirectory { root in
            try expectFailure {
                _ = try HistoryDiagnosticSampleExporter.export(request(text: "private"), to: root) { phase in
                    if phase == .beforePublish { throw Failure.injected }
                }
            }
            let remaining = try files(root)
            try require(remaining.isEmpty, "Publish failure left partial sample")
        }
    }

    static func cleanupFailureDisclosure() throws {
        try withDirectory { root in
            var disclosedURL: URL?
            do {
                _ = try HistoryDiagnosticSampleExporter.export(request(text: "private"), to: root) { phase in
                    if phase == .beforePublish || phase == .beforeCleanup { throw Failure.injected }
                }
            } catch HistoryDiagnosticSampleError.cleanupFailed(let url) { disclosedURL = url }
            try require(disclosedURL != nil, "Cleanup failure was hidden")
            let remaining = try files(root)
            try require(remaining == Set([disclosedURL!.lastPathComponent]), "Wrong partial sample disclosed")
        }
    }

    static func sourcePathBoundaries() throws {
        try withDirectory { root in
            _ = try source(in: root)
            try Data("unrelated-secret".utf8).write(to: root.appendingPathComponent("unrelated.wav"))
            for path in ["../unrelated.wav", "/transcription/sample.wav", "transcription/../unrelated.wav", "transcription//sample.wav", "transcription/", "transcription/sample.wav\0"] {
                try expectFailure({ _ = try HistoryDiagnosticAudioSource(rootURL: root, relativePath: path) }, matching: .unsafeAudioPath)
            }
            let link = root.appendingPathComponent("transcription/link.wav")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root.appendingPathComponent("unrelated.wav"))
            try expectFailure { _ = try HistoryDiagnosticAudioSource(rootURL: root, relativePath: "transcription/link.wav") }
            try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("translation"), withDestinationURL: root.appendingPathComponent("transcription"))
            try expectFailure { _ = try HistoryDiagnosticAudioSource(rootURL: root, relativePath: "translation/sample.wav") }
            let directory = root.appendingPathComponent("transcription/directory.wav")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            try expectFailure({ _ = try HistoryDiagnosticAudioSource(rootURL: root, relativePath: "transcription/directory.wav") }, matching: .unsafeAudioPath)
            let unrelated = try String(contentsOf: root.appendingPathComponent("unrelated.wav"), encoding: .utf8)
            try require(unrelated == "unrelated-secret", "Unrelated file modified")
        }
    }

    static func sourceUnlinked() throws {
        try withDirectory { root in
            let contents = Data("opened-original".utf8)
            let audio = try source(in: root, contents: contents)
            let sourceURL = root.appendingPathComponent("transcription/sample.wav")
            try FileManager.default.removeItem(at: sourceURL)
            try Data("replacement".utf8).write(to: sourceURL)
            let output = try HistoryDiagnosticSampleExporter.export(request(audio: audio, includeAudio: true), to: root)
            let exported = try Data(contentsOf: output.appendingPathComponent("recording.wav"))
            try require(exported == contents, "Descriptor did not protect selected source")
        }
    }

    static func sourceMutated() throws {
        try withDirectory { root in
            let audio = try source(in: root)
            let handle = try FileHandle(forWritingTo: root.appendingPathComponent("transcription/sample.wav"))
            try handle.truncate(atOffset: 1)
            try handle.close()
            try expectFailure { _ = try HistoryDiagnosticSampleExporter.export(request(audio: audio, includeAudio: true), to: root) }
            let remaining = try files(root)
            try require(remaining == ["transcription"], "Source mutation published bad copy")
        }
    }

    static func manifestPrivacy() throws {
        try withDirectory { root in
            let secrets = "private-transcript https://private.example/token sk-secret /Users/private/model"
            try Data(secrets.utf8).write(to: root.appendingPathComponent("other-history.json"))
            let output = try HistoryDiagnosticSampleExporter.export(request(text: "explicitly selected", version: secrets), to: root)
            let contents = try Data(contentsOf: output.appendingPathComponent("manifest.json"))
            let manifest = try JSONSerialization.jsonObject(with: contents) as! [String: Any]
            try require(Set(manifest.keys) == ["schemaVersion", "appVersion", "files"], "Manifest contains unapproved keys")
            try require(manifest["appVersion"] as? String == "unknown", "Private custom version escaped")
            for file in try files(output) {
                let text = try String(contentsOf: output.appendingPathComponent(file), encoding: .utf8)
                try require(!text.contains(secrets), "Other history/config leaked")
            }
        }
    }

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

    static func retainedAudioPreview() throws {
        try withDirectory { root in
            let url = root.appendingPathComponent("transcription/sample.wav")
            _ = try HistoryAudioArchiveSupport.exportWAV(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000, to: url)
            let audio = try HistoryDiagnosticAudioSource(rootURL: root, relativePath: "transcription/sample.wav")
            try FileManager.default.removeItem(at: url)
            let player = try AVAudioPlayer(contentsOf: audio.previewURL)
            try require(abs(player.duration - 1) < 0.01, "Retained audio preview cannot decode the selected archive")
            try require(!player.isPlaying, "Preview started without user input")
        }
    }

    static func taskCancellation() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-sample-cancel-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let audio = try source(in: root)
        let selected = request(text: "private", audio: audio, includeAudio: true)
        let started = DispatchSemaphore(value: 0)
        let resume = DispatchSemaphore(value: 0)
        let worker = Task.detached {
            try HistoryDiagnosticSampleExporter.export(selected, to: root) { phase in
                if phase == .copyingAudio {
                    started.signal()
                    guard resume.wait(timeout: .now() + 10) == .success else { throw Failure.injected }
                }
            }
        }
        let didStart = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(returning: started.wait(timeout: .now() + 10) == .success)
            }
        }
        worker.cancel()
        resume.signal()
        var wasCancelled = false
        do { _ = try await worker.value } catch is CancellationError { wasCancelled = true }
        try require(didStart && wasCancelled, "Task cancellation did not stop copying")
        let remaining = try files(root)
        try require(remaining == ["transcription"], "Cancelled task left partial files")
    }

    static func syntheticAudioRoundtrip() throws {
        try withDirectory { root in
            let url = root.appendingPathComponent("transcription/sample.wav")
            let samples: [Float] = [0, 0.25, -0.25, 0.5, -0.5]
            _ = try HistoryAudioArchiveSupport.exportWAV(samples: samples, sampleRate: 16_000, to: url)
            let audio = try HistoryDiagnosticAudioSource(rootURL: root, relativePath: "transcription/sample.wav")
            let output = try HistoryDiagnosticSampleExporter.export(request(text: "synthetic fixture", audio: audio, includeAudio: true), to: root)
            let decoded = try HistoryAudioArchiveSupport.readWAVSamples(from: output.appendingPathComponent("recording.wav"))
            try require(decoded.count == samples.count, "Sample count changed")
            for (actual, expected) in zip(decoded, samples) {
                try require(abs(actual - expected) < 0.001, "Fixture audio changed")
            }
        }
    }
}
