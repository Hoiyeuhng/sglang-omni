// HistoryAudioArchiveSupportTests.swift
// Provides History Audio Archive Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class HistoryAudioArchiveSupportTests: XCTestCase {
    @MainActor
    func testStoppingPreviousFolderManagementPreservesFilesAndCurrentFolder() throws {
        let suite = "VoxtAudioDirectories-\(UUID())"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-audio-directories-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let current = root.appendingPathComponent("current", isDirectory: true)
        let previous = root.appendingPathComponent("previous", isDirectory: true)
        try FileManager.default.createDirectory(at: previous, withIntermediateDirectories: true)
        let recording = previous.appendingPathComponent("retained.wav")
        try Data("retained synthetic audio".utf8).write(to: recording)
        defaults.set(current.path, forKey: AppPreferenceKey.historyAudioStorageRootPath)
        defaults.set([previous.path, current.path], forKey: "historyAudioPreviousRootPaths")

        XCTAssertFalse(HistoryAudioStorageDirectoryManager.stopManagingPreviousRoot(path: current.path, defaults: defaults))
        XCTAssertTrue(HistoryAudioStorageDirectoryManager.stopManagingPreviousRoot(path: previous.path, defaults: defaults))
        XCTAssertEqual(HistoryAudioStorageDirectoryManager.knownRootURLs(defaults: defaults), [current])
        XCTAssertEqual(try String(contentsOf: recording, encoding: .utf8), "retained synthetic audio")
        XCTAssertThrowsError(try HistoryAudioStorageDirectoryManager.reauthorizeRootURL(
            current, previousPath: previous.path, defaults: defaults
        ))
        XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.historyAudioStorageRootPath), current.path)
    }

    func testExportWAVRoundTripsAsMono16BitSamples() throws {
        let samples: [Float] = [0, 0.25, -0.25, 0.75, -0.75]
        let destinationURL = HistoryAudioArchiveSupport.temporaryArchiveURL(prefix: "history-audio-roundtrip")

        defer {
            try? FileManager.default.removeItem(at: destinationURL)
        }

        let didExport = try HistoryAudioArchiveSupport.exportWAV(
            samples: samples,
            sampleRate: HistoryAudioArchiveSupport.targetSampleRate,
            to: destinationURL
        )

        XCTAssertTrue(didExport)
        let decodedSamples = try HistoryAudioArchiveSupport.readWAVSamples(from: destinationURL)
        XCTAssertEqual(decodedSamples.count, samples.count)
        for (expected, actual) in zip(samples, decodedSamples) {
            XCTAssertEqual(actual, expected, accuracy: 0.02)
        }
    }

    func testMergedRewriteArchiveAddsSilenceGapBeforeAppendedAudio() throws {
        let existingURL = HistoryAudioArchiveSupport.temporaryArchiveURL(prefix: "history-audio-existing")
        let appendedURL = HistoryAudioArchiveSupport.temporaryArchiveURL(prefix: "history-audio-appended")

        defer {
            try? FileManager.default.removeItem(at: existingURL)
            try? FileManager.default.removeItem(at: appendedURL)
        }

        _ = try HistoryAudioArchiveSupport.exportWAV(
            samples: [Float](repeating: 0.4, count: 1_600),
            sampleRate: HistoryAudioArchiveSupport.targetSampleRate,
            to: existingURL
        )
        _ = try HistoryAudioArchiveSupport.exportWAV(
            samples: [Float](repeating: 0.8, count: 1_600),
            sampleRate: HistoryAudioArchiveSupport.targetSampleRate,
            to: appendedURL
        )

        let mergedURL = try HistoryAudioArchiveSupport.mergedRewriteArchive(
            existingArchiveURL: existingURL,
            appendedArchiveURL: appendedURL
        )

        defer {
            try? FileManager.default.removeItem(at: mergedURL)
        }

        let mergedSamples = try HistoryAudioArchiveSupport.readWAVSamples(from: mergedURL)
        let expectedGapCount = Int((HistoryAudioArchiveSupport.rewriteJoinGapSeconds * HistoryAudioArchiveSupport.targetSampleRate).rounded())

        XCTAssertEqual(mergedSamples.count, 1_600 + expectedGapCount + 1_600)
        XCTAssertEqual(mergedSamples[100], 0.4, accuracy: 0.02)
        XCTAssertEqual(mergedSamples[1_600 + (expectedGapCount / 2)], 0, accuracy: 0.001)
        XCTAssertEqual(mergedSamples.last ?? 0, 0.8, accuracy: 0.02)
    }
}
