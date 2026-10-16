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

    static func resolvedRootURL(defaults: UserDefaults = .standard) -> URL {
        if let bookmarkData = defaults.data(forKey: AppPreferenceKey.historyAudioStorageRootBookmark),
           let bookmarkedURL = resolveSecurityScopedURL(from: bookmarkData, defaults: defaults) {
            return bookmarkedURL
        }

        if let path = defaults.string(forKey: AppPreferenceKey.historyAudioStorageRootPath), !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }

        return defaultRootURL
    }

    static func knownRootURLs(defaults: UserDefaults = .standard) -> [URL] {
        let current = resolvedRootURL(defaults: defaults)
        var roots = [current]
        let bookmarks = defaults.dictionary(forKey: "historyAudioPreviousRootBookmarks") as? [String: Data] ?? [:]
        for path in defaults.stringArray(forKey: "historyAudioPreviousRootPaths") ?? [] {
            if let bookmark = bookmarks[path], let resolved = resolveSecurityScopedURL(from: bookmark, defaults: defaults) {
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

        _ = resolveSecurityScopedURL(from: bookmark, defaults: defaults)
    }

    static func reauthorizeRootURL(_ url: URL, previousPath: String, defaults: UserDefaults = .standard) throws {
        let previous = URL(fileURLWithPath: previousPath, isDirectory: true).standardizedFileURL
        guard url.resolvingSymlinksInPath().standardizedFileURL == previous.resolvingSymlinksInPath() else {
            throw HistoryAudioQuotaError.unsafePath
        }
        let bookmark = try url.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
        if resolvedRootURL(defaults: defaults).standardizedFileURL == previous {
            defaults.set(bookmark, forKey: AppPreferenceKey.historyAudioStorageRootBookmark)
        } else {
            var bookmarks = defaults.dictionary(forKey: "historyAudioPreviousRootBookmarks") as? [String: Data] ?? [:]
            bookmarks[previousPath] = bookmark
            defaults.set(bookmarks, forKey: "historyAudioPreviousRootBookmarks")
        }
        _ = resolveSecurityScopedURL(from: bookmark, defaults: defaults)
    }

    @discardableResult
    static func stopManagingPreviousRoot(path: String, defaults: UserDefaults = .standard) -> Bool {
        let root = URL(fileURLWithPath: path, isDirectory: true).resolvingSymlinksInPath().standardizedFileURL
        guard root != resolvedRootURL(defaults: defaults).resolvingSymlinksInPath().standardizedFileURL else { return false }
        let previousPaths = defaults.stringArray(forKey: "historyAudioPreviousRootPaths") ?? []
        let removedPaths = previousPaths.filter {
            URL(fileURLWithPath: $0, isDirectory: true).resolvingSymlinksInPath().standardizedFileURL == root
        }
        guard !removedPaths.isEmpty else { return false }
        defaults.set(previousPaths.filter { !removedPaths.contains($0) }, forKey: "historyAudioPreviousRootPaths")
        var bookmarks = defaults.dictionary(forKey: "historyAudioPreviousRootBookmarks") as? [String: Data] ?? [:]
        for removedPath in removedPaths {
            bookmarks.removeValue(forKey: removedPath)
        }
        let scopedPaths = securityScopedURLs.keys.filter {
            URL(fileURLWithPath: $0, isDirectory: true).resolvingSymlinksInPath().standardizedFileURL == root
        }
        for scopedPath in scopedPaths { securityScopedURLs.removeValue(forKey: scopedPath)?.stopAccessingSecurityScopedResource() }
        defaults.set(bookmarks, forKey: "historyAudioPreviousRootBookmarks")
        return true
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

    private static func resolveSecurityScopedURL(from bookmarkData: Data, defaults: UserDefaults) -> URL? {
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
