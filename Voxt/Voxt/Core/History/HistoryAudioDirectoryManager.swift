// HistoryAudioDirectoryManager.swift
// Provides History Audio Directory Manager for history storage and display support.

import Foundation
import AppKit

enum HistoryAudioStorageDirectoryManager {
    private static var securityScopedURLs: [String: URL] = [:]
    private static let fileManager = FileManager.default

    static var defaultRootURL: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support", isDirectory: true)
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("transcription-history-audio", isDirectory: true)
    }

    static func resolvedRootURL() -> URL {
        let defaults = UserDefaults.standard
        if let bookmarkData = defaults.data(forKey: AppPreferenceKey.historyAudioStorageRootBookmark),
           let bookmarkedURL = resolveSecurityScopedURL(from: bookmarkData) {
            return bookmarkedURL
        }

        if let path = defaults.string(forKey: AppPreferenceKey.historyAudioStorageRootPath), !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }

        return defaultRootURL
    }

    static func knownRootURLs() -> [URL] {
        let defaults = UserDefaults.standard
        let current = resolvedRootURL()
        var roots = [current]
        let bookmarks = defaults.dictionary(forKey: "historyAudioPreviousRootBookmarks") as? [String: Data] ?? [:]
        for path in defaults.stringArray(forKey: "historyAudioPreviousRootPaths") ?? [] {
            if let bookmark = bookmarks[path], let resolved = resolveSecurityScopedURL(from: bookmark) {
                roots.append(resolved)
            } else {
                roots.append(URL(fileURLWithPath: path, isDirectory: true))
            }
        }
        var seen = Set<String>()
        return roots.filter { seen.insert($0.resolvingSymlinksInPath().standardizedFileURL.path).inserted }
    }

    static func saveUserSelectedRootURL(_ url: URL) throws {
        let normalized = url.standardizedFileURL
        let bookmark = try normalized.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )

        let defaults = UserDefaults.standard
        let previous = resolvedRootURL()
        var paths = defaults.stringArray(forKey: "historyAudioPreviousRootPaths") ?? []
        if !paths.contains(previous.path) { paths.append(previous.path) }
        defaults.set(paths, forKey: "historyAudioPreviousRootPaths")
        var bookmarks = defaults.dictionary(forKey: "historyAudioPreviousRootBookmarks") as? [String: Data] ?? [:]
        if let previousBookmark = defaults.data(forKey: AppPreferenceKey.historyAudioStorageRootBookmark) {
            bookmarks[previous.path] = previousBookmark
        }
        defaults.set(bookmarks, forKey: "historyAudioPreviousRootBookmarks")
        defaults.set(normalized.path, forKey: AppPreferenceKey.historyAudioStorageRootPath)
        defaults.set(bookmark, forKey: AppPreferenceKey.historyAudioStorageRootBookmark)

        _ = resolveSecurityScopedURL(from: bookmark)
    }

    static func ensureRootDirectoryExists() throws -> URL {
        let url = resolvedRootURL()
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func openRootInFinder() {
        if let url = try? ensureRootDirectoryExists() {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        }
    }

    private static func resolveSecurityScopedURL(from bookmarkData: Data) -> URL? {
        var isStale = false
        guard let resolved = try? URL(
            resolvingBookmarkData: bookmarkData,
            options: [.withSecurityScope, .withoutUI],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        ) else {
            return nil
        }

        if securityScopedURLs[resolved.path] == nil, resolved.startAccessingSecurityScopedResource() {
            securityScopedURLs[resolved.path] = resolved
        }

        if isStale,
           let refreshed = try? resolved.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil
           ) {
            let defaults = UserDefaults.standard
            if defaults.string(forKey: AppPreferenceKey.historyAudioStorageRootPath) == resolved.path {
                defaults.set(refreshed, forKey: AppPreferenceKey.historyAudioStorageRootBookmark)
            } else {
                var bookmarks = defaults.dictionary(forKey: "historyAudioPreviousRootBookmarks") as? [String: Data] ?? [:]
                bookmarks[resolved.path] = refreshed
                defaults.set(bookmarks, forKey: "historyAudioPreviousRootBookmarks")
            }
        }

        return resolved
    }
}
