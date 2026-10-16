// TranscriptionHistoryStore.swift
// Provides Transcription History Store for history storage and display support.

import Foundation
import Combine
import os

@MainActor
final class TranscriptionHistoryStore: ObservableObject {
    @Published private(set) var entries: [TranscriptionHistoryEntry] = []
    @Published private(set) var isLoading = false

    private var allEntries: [TranscriptionHistoryEntry] = []
    private var entriesByKind: [TranscriptionHistoryKind: [TranscriptionHistoryEntry]] = [:]
    private var loadedCount = 0
    private var totalEntryCount = 0
    private var reloadGeneration = 0
    private var nextPageRequestGeneration = 0
    private var isLoadingNextPage = false
    private let pageSize = 40
    private let retentionCleanupKinds: Set<TranscriptionHistoryKind> = [
        .normal,
        .translation,
        .rewrite
    ]

    private let fileManager = FileManager.default
    private let defaults: UserDefaults
    private var audioUseObserver: AnyCancellable?
    private var importedAudioUses: [String: HistoryAudioFileUse] = [:]
    @Published private(set) var databaseStorageBytes: Int64?
    @Published private(set) var diagnosticStorageBytes: Int64?
    @Published private(set) var diagnosticStorageIssue = false
    @Published private(set) var audioQuotaSnapshot = HistoryAudioQuotaSnapshot(measurementFailed: true)
    private static let audioQuotaQueue = DispatchQueue(label: "app.voxt.history-audio-quota", qos: .utility)
    private let requestedAudioQuotaLimitBytes = OSAllocatedUnfairLock<Int64?>(initialState: nil)
    private var audioQuotaRefreshInProgress = false
    private var audioQuotaRefreshRequested = false
    private let repository: HistoryRepositoryProtocol
    private let audioArchive: HistoryAudioArchiveManaging
    private let logFileStore: VoxtLogFileStore
    private let isolatedDefaultsSuite: String?
    private let isolatedHistoryFileURL: URL?

    init(
        repository: HistoryRepositoryProtocol? = nil,
        audioArchive: HistoryAudioArchiveManaging? = nil,
        defaults: UserDefaults? = nil,
        logFileStore: VoxtLogFileStore? = nil
    ) {
        let isolationID = UUID().uuidString
        let isolatedRoot = repository.map { _ in
            FileManager.default.temporaryDirectory.appendingPathComponent("voxt-history-store-\(isolationID)")
        }
        let suite = repository != nil && defaults == nil ? "VoxtHistoryStore-\(isolationID)" : nil
        self.isolatedDefaultsSuite = suite
        self.isolatedHistoryFileURL = isolatedRoot?.appendingPathComponent("transcription-history.json")
        if let defaults {
            self.defaults = defaults
        } else if let suite {
            guard let isolatedDefaults = UserDefaults(suiteName: suite) else {
                preconditionFailure("Cannot create isolated history preferences")
            }
            self.defaults = isolatedDefaults
        } else {
            self.defaults = .standard
        }
        self.repository = repository ?? HistoryRepository()
        self.audioArchive = audioArchive ?? HistoryAudioArchiveService(rootURL: isolatedRoot?.appendingPathComponent("audio"))
        self.logFileStore = logFileStore ?? isolatedRoot.map { VoxtLogFileStore(baseDirectoryURL: $0.appendingPathComponent("logs")) } ?? .shared
        HistoryAudioQuotaPreference.initialize(
            defaults: self.defaults,
            hasExistingData: ((try? self.repository.entryCount(kind: nil, query: "")) ?? 1) > 0
        )
        audioUseObserver = NotificationCenter.default.publisher(for: HistoryAudioFileUse.releasedNotification)
            .debounce(for: .milliseconds(100), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in self?.applyAudioStorageQuotaIfNeeded() }
        if repository == nil {
            reloadAsync()
        } else {
            // Injected repositories are primarily deterministic test/support stores.
            reload()
        }
    }

    deinit {
        if let isolatedDefaultsSuite { defaults.removePersistentDomain(forName: isolatedDefaultsSuite) }
    }

    var hasMore: Bool {
        loadedCount < totalEntryCount
    }

    func historyEntries(for kind: TranscriptionHistoryKind) -> [TranscriptionHistoryEntry] {
        entriesByKind[kind] ?? []
    }

    func entry(id: UUID) -> TranscriptionHistoryEntry? {
        if let cached = allEntries.first(where: { $0.id == id }) {
            return cached
        }
        return try? repository.entry(id: id)
    }

    func updateRetentionPolicy() {
        cleanupRetainedEntriesIfNeeded()
        reloadAsync()
    }

    func reload() {
        invalidatePendingLoads()
        do {
            let repositoryCount = try repository.entryCount(kind: nil, query: "")
            if repositoryCount > 0 || !legacyHistoryFileExists() {
                let firstPage = try repository.entries(kind: nil, query: "", limit: pageSize, offset: 0)
                applyLoadedEntries(firstPage, totalCount: repositoryCount, resetPagination: true)
                return
            }

            let url = try historyFileURL()
            guard fileManager.fileExists(atPath: url.path) else {
                applyReloadedEntries([], resetPagination: true)
                return
            }
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([TranscriptionHistoryEntry].self, from: data)
            applyReloadedEntries(decoded, resetPagination: true)
        } catch {
            applyReloadedEntries([], resetPagination: true)
        }
    }

    func reloadAsync() {
        invalidatePendingLoads()
        let generation = reloadGeneration
        let repository = repository
        isLoading = true

        let url: URL?
        do {
            url = try historyFileURL()
        } catch {
            isLoading = false
            applyReloadedEntries([], resetPagination: true)
            return
        }

        DispatchQueue.global(qos: .utility).async { [weak self, url] in
            let loadedEntries: [TranscriptionHistoryEntry]
            let totalCount: Int
            if let repositoryCount = try? repository.entryCount(kind: nil, query: ""),
               repositoryCount > 0 || url.map({ !FileManager.default.fileExists(atPath: $0.path) }) == true {
                totalCount = repositoryCount
                loadedEntries = (try? repository.entries(kind: nil, query: "", limit: 40, offset: 0)) ?? []
            } else if let url, FileManager.default.fileExists(atPath: url.path) {
                do {
                    let data = try Data(contentsOf: url)
                    loadedEntries = try JSONDecoder().decode([TranscriptionHistoryEntry].self, from: data)
                    totalCount = loadedEntries.reduce(into: 0) { count, entry in
                        if case .transcript = entry.kind {
                            return
                        }
                        count += 1
                    }
                } catch {
                    loadedEntries = []
                    totalCount = 0
                }
            } else {
                loadedEntries = []
                totalCount = 0
            }

            DispatchQueue.main.async {
                guard let self, generation == self.reloadGeneration else { return }
                self.isLoading = false
                self.applyLoadedEntries(loadedEntries, totalCount: totalCount, resetPagination: false)
            }
        }
    }

    func loadNextPage() {
        guard hasMore, !isLoadingNextPage else { return }
        isLoadingNextPage = true
        let offset = loadedCount
        let generation = reloadGeneration
        nextPageRequestGeneration += 1
        let requestGeneration = nextPageRequestGeneration
        let repository = repository
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let page = (try? repository.entries(kind: nil, query: "", limit: 40, offset: offset)) ?? []
            DispatchQueue.main.async {
                guard let self else { return }
                guard requestGeneration == self.nextPageRequestGeneration else { return }
                self.isLoadingNextPage = false
                guard generation == self.reloadGeneration,
                      offset == self.loadedCount
                else {
                    return
                }
                guard !page.isEmpty else {
                    self.totalEntryCount = self.loadedCount
                    self.publishVisibleEntries()
                    return
                }
                self.mergeLoadedEntries(page)
                self.loadedCount = min(self.loadedCount + page.count, self.totalEntryCount)
                self.refreshEntryIndexes()
                self.publishVisibleEntries()
            }
        }
    }

    func historyEntries(
        kind: TranscriptionHistoryKind?,
        query: String = "",
        limit: Int,
        offset: Int
    ) -> [TranscriptionHistoryEntry] {
        (try? repository.entries(kind: kind, query: query, limit: limit, offset: offset)) ?? []
    }

    func entryCount(kind: TranscriptionHistoryKind?, query: String = "") -> Int {
        (try? repository.entryCount(kind: kind, query: query)) ?? 0
    }

    func loadEntries(
        kind: TranscriptionHistoryKind?,
        query: String = "",
        limit: Int,
        offset: Int,
        completion: @escaping (Int, [TranscriptionHistoryEntry]) -> Void
    ) {
        let repository = repository
        DispatchQueue.global(qos: .userInitiated).async {
            let count = (try? repository.entryCount(kind: kind, query: query)) ?? 0
            let page = (try? repository.entries(kind: kind, query: query, limit: limit, offset: offset)) ?? []
            DispatchQueue.main.async {
                completion(count, page)
            }
        }
    }

    func loadListEntries(
        kind: TranscriptionHistoryKind?,
        query: String = "",
        limit: Int,
        offset: Int,
        completion: @escaping (Int, [TranscriptionHistoryListEntry]) -> Void
    ) {
        let repository = repository
        DispatchQueue.global(qos: .userInitiated).async {
            let count = (try? repository.entryCount(kind: kind, query: query)) ?? 0
            let page = (try? repository.listEntries(kind: kind, query: query, limit: limit, offset: offset)) ?? []
            DispatchQueue.main.async {
                completion(count, page)
            }
        }
    }

    /// Query each kind independently so unrelated history and UI pagination cannot
    /// crowd recent dictation/translation results out of the status menu.
    func loadRecentMenuEntries(
        limitPerKind: Int,
        completion: @escaping ([TranscriptionHistoryListEntry]) -> Void
    ) {
        let repository = repository
        DispatchQueue.global(qos: .utility).async {
            let candidates = [TranscriptionHistoryKind.normal, .translation].flatMap { kind in
                (try? repository.listEntries(kind: kind, query: "", limit: limitPerKind, offset: 0)) ?? []
            }
            DispatchQueue.main.async {
                completion(candidates)
            }
        }
    }

    func loadEntry(
        id: UUID,
        completion: @escaping (TranscriptionHistoryEntry?) -> Void
    ) {
        let repository = repository
        DispatchQueue.global(qos: .userInitiated).async {
            let entry = try? repository.entry(id: id)
            DispatchQueue.main.async {
                completion(entry ?? nil)
            }
        }
    }

    func allEntriesBatch(limit: Int, offset: Int) -> [TranscriptionHistoryEntry] {
        (try? repository.entries(kind: nil, query: "", limit: limit, offset: offset)) ?? []
    }

    func latestEntryText() -> String? {
        try? repository.latestEntryText()
    }

    func pendingDictionaryHistoryEntryCount(after checkpoint: DictionaryHistoryScanCheckpoint?) -> Int {
        (try? repository.pendingNormalEntryCount(after: checkpoint)) ?? 0
    }

    func pendingDictionaryHistoryEntries(after checkpoint: DictionaryHistoryScanCheckpoint?) -> [TranscriptionHistoryEntry] {
        (try? repository.pendingNormalEntries(after: checkpoint)) ?? []
    }

    func reportMetrics(
        dayStarts: [Date],
        branchStartDate: Date? = nil,
        completion: @escaping (HistoryReportMetrics?) -> Void
    ) {
        let repository = repository
        DispatchQueue.global(qos: .utility).async {
            let metrics = try? repository.reportMetrics(dayStarts: dayStarts, branchStartDate: branchStartDate)
            DispatchQueue.main.async {
                completion(metrics)
            }
        }
    }

    @discardableResult
    func append(
        entryID: UUID = UUID(),
        text: String,
        transcriptionEngine: String,
        transcriptionModel: String,
        enhancementMode: String,
        enhancementModel: String,
        kind: TranscriptionHistoryKind,
        isTranslation: Bool,
        audioDurationSeconds: TimeInterval?,
        transcriptionProcessingDurationSeconds: TimeInterval?,
        llmDurationSeconds: TimeInterval?,
        focusedAppName: String?,
        focusedAppBundleID: String?,
        browserURLHost: String? = nil,
        browserURLOrigin: String? = nil,
        matchedGroupID: UUID?,
        matchedGroupName: String?,
        matchedAppGroupName: String?,
        matchedURLGroupName: String?,
        remoteASRProvider: String?,
        remoteASRModel: String?,
        remoteASREndpoint: String?,
        remoteLLMProvider: String?,
        remoteLLMModel: String?,
        remoteLLMEndpoint: String?,
        audioRelativePath: String? = nil,
        whisperWordTimings: [WhisperHistoryWordTiming]?,
        senseVoiceMetadata: SenseVoiceTranscriptMetadata? = nil,
        transcriptSegments: [TranscriptSegment]? = nil,
        transcriptAudioRelativePath: String? = nil,
        meetingCaptureMode: MeetingCaptureMode? = nil,
        transcriptSummary: TranscriptSummarySnapshot? = nil,
        displayTitle: String? = nil,
        transcriptionChatMessages: [TranscriptSummaryChatMessage]? = nil,
        dictionaryHitTerms: [String],
        dictionaryCorrectedTerms: [String],
        dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot] = [],
        dictionarySuggestedTerms: [DictionarySuggestionSnapshot],
        allowEmptyTextWithAudio: Bool = false
    ) -> UUID? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || (allowEmptyTextWithAudio && audioRelativePath != nil) else { return nil }
        if (try? repository.entry(id: entryID)) != nil {
            return entryID
        }

        let entry = TranscriptionHistoryEntry(
            id: entryID,
            text: trimmed,
            createdAt: Date(),
            transcriptionEngine: transcriptionEngine,
            transcriptionModel: transcriptionModel,
            enhancementMode: enhancementMode,
            enhancementModel: enhancementModel,
            kind: kind,
            isTranslation: isTranslation,
            audioDurationSeconds: audioDurationSeconds,
            transcriptionProcessingDurationSeconds: transcriptionProcessingDurationSeconds,
            llmDurationSeconds: llmDurationSeconds,
            focusedAppName: focusedAppName,
            focusedAppBundleID: focusedAppBundleID,
            browserURLHost: browserURLHost,
            browserURLOrigin: browserURLOrigin,
            matchedGroupID: matchedGroupID,
            matchedGroupName: matchedGroupName,
            matchedAppGroupName: matchedAppGroupName,
            matchedURLGroupName: matchedURLGroupName,
            remoteASRProvider: remoteASRProvider,
            remoteASRModel: remoteASRModel,
            remoteASREndpoint: remoteASREndpoint,
            remoteLLMProvider: remoteLLMProvider,
            remoteLLMModel: remoteLLMModel,
            remoteLLMEndpoint: remoteLLMEndpoint,
            audioRelativePath: audioRelativePath,
            whisperWordTimings: whisperWordTimings,
            senseVoiceMetadata: senseVoiceMetadata,
            transcriptSegments: transcriptSegments,
            transcriptAudioRelativePath: transcriptAudioRelativePath,
            meetingCaptureMode: meetingCaptureMode,
            transcriptSummary: transcriptSummary,
            displayTitle: displayTitle,
            transcriptionChatMessages: transcriptionChatMessages,
            dictionaryHitTerms: dictionaryHitTerms,
            dictionaryCorrectedTerms: dictionaryCorrectedTerms,
            dictionaryCorrectionSnapshots: dictionaryCorrectionSnapshots,
            dictionarySuggestedTerms: dictionarySuggestedTerms
        )

        let importedPaths = [audioRelativePath, transcriptAudioRelativePath].compactMap { $0 }
        defer { importedPaths.forEach(releaseImportedAudio(path:)) }
        guard persistEntry(entry) else {
            removeUnreferencedAudio(paths: importedPaths)
            return nil
        }

        totalEntryCount += 1
        cacheUpdatedEntry(entry)
        loadedCount = min(max(loadedCount + 1, min(pageSize, allEntries.count)), totalEntryCount)
        refreshEntryIndexes()
        publishVisibleEntries()
        cleanupRetainedEntriesIfNeeded()
        applyAudioStorageQuotaIfNeeded()
        return entry.id
    }

    @discardableResult
    func delete(id: UUID) -> Bool {
        let wasCached = allEntries.contains { $0.id == id }
        let removed: TranscriptionHistoryEntry?
        do {
            removed = try repository.delete(id: id)
        } catch {
            return false
        }

        let interruptedReload = invalidatePendingLoads()
        removeCachedEntry(id: id)
        if removed != nil || wasCached {
            totalEntryCount = max(0, totalEntryCount - 1)
        }
        loadedCount = min(loadedCount, totalEntryCount)
        refreshEntryIndexes()
        publishVisibleEntries()
        if let removed { removeUnreferencedAudio(paths: [removed.audioRelativePath, removed.transcriptAudioRelativePath].compactMap { $0 }) }
        if interruptedReload {
            reloadAsync()
        }
        return true
    }

    func clearAll() {
        guard let audioPaths = try? repository.audioRelativePaths() else { return }
        do {
            try repository.clearAll()
        } catch {
            return
        }

        invalidatePendingLoads()
        removeUnreferencedAudio(paths: audioPaths)
        allEntries = []
        loadedCount = 0
        totalEntryCount = 0
        refreshEntryIndexes()
        publishVisibleEntries()
    }

    @discardableResult
    func clear(kind: TranscriptionHistoryKind) -> Bool {
        let removedEntries: [TranscriptionHistoryEntry]
        do {
            removedEntries = try repository.deleteEntries(kind: kind)
        } catch {
            return false
        }

        let interruptedReload = invalidatePendingLoads()
        removeUnreferencedAudio(paths: removedEntries.flatMap { [$0.audioRelativePath, $0.transcriptAudioRelativePath].compactMap { $0 } })
        allEntries.removeAll { $0.kind == kind }
        totalEntryCount = max(0, totalEntryCount - removedEntries.count)
        loadedCount = min(loadedCount, allEntries.count)
        refreshEntryIndexes()
        publishVisibleEntries()
        if interruptedReload {
            reloadAsync()
        }
        return true
    }

    func importAudioArchive(
        from sourceURL: URL,
        kind: TranscriptionHistoryKind,
        preferredFileName: String? = nil
    ) throws -> String {
        if let limitBytes = HistoryAudioQuotaPreference.limitBytes(defaults: defaults),
           let bytes = try sourceURL.resourceValues(forKeys: [.fileSizeKey]).fileSize,
           Int64(bytes) > limitBytes {
            throw HistoryAudioQuotaError.oversized
        }
        let path = try audioArchive.importArchive(from: sourceURL, kind: kind, preferredFileName: preferredFileName)
        importedAudioUses[path] = HistoryAudioFileUse(url: try audioArchive.rootURL().appendingPathComponent(path))
        return path
    }

    func removeImportedAudioArchive(relativePath: String) {
        releaseImportedAudio(path: relativePath)
        removeUnreferencedAudio(paths: [relativePath])
    }

    func replaceAudioArchive(for entryID: UUID, with sourceURL: URL) throws -> TranscriptionHistoryEntry? {
        guard let existingEntry = entry(id: entryID) else { return nil }

        let resolvedRelativePath = try importAudioArchive(from: sourceURL, kind: existingEntry.kind)
        let updatedEntry = existingEntry.updatingAudioRelativePath(resolvedRelativePath)
        defer { releaseImportedAudio(path: resolvedRelativePath) }
        guard persistEntry(updatedEntry) else {
            removeUnreferencedAudio(paths: [resolvedRelativePath])
            throw HistoryAudioQuotaError.filesystem
        }
        cacheUpdatedEntry(updatedEntry)
        applyAudioStorageQuotaIfNeeded()
        return updatedEntry
    }

    func audioURL(for entry: TranscriptionHistoryEntry) -> URL? {
        audioArchive.audioURL(for: entry)
    }

    func exportAllAudioArchives(to destinationDirectoryURL: URL) throws -> HistoryAudioExportSummary {
        try audioArchive.exportAllArchives(to: destinationDirectoryURL) { visitBatch in
            forEachHistoryBatch(visitBatch)
        }
    }

    func applyAudioStorageQuotaIfNeeded() {
        let limitBytes = HistoryAudioQuotaPreference.limitBytes(defaults: defaults)
        let requestedLimitBytes = requestedAudioQuotaLimitBytes
        requestedLimitBytes.withLock { $0 = limitBytes }
        audioQuotaRefreshRequested = true
        guard !audioQuotaRefreshInProgress else { return }
        audioQuotaRefreshRequested = false
        audioQuotaRefreshInProgress = true
        let roots: [URL]
        do {
            roots = try audioArchive.managedRootURLs()
        } catch {
            audioQuotaRefreshInProgress = false
            audioQuotaSnapshot = HistoryAudioQuotaSnapshot(measurementFailed: true)
            return
        }
        do {
            _ = try audioArchive.rootURL()
        } catch {
            audioQuotaSnapshot = HistoryAudioQuotaSnapshot(
                measurementFailed: true, unavailableRootPaths: roots.first.map { [$0.path] } ?? []
            )
        }
        let repository = repository
        let protectedPaths = Set(importedAudioUses.keys)
        let logFileStore = logFileStore
        Self.audioQuotaQueue.async { [weak self] in
            var snapshot: HistoryAudioQuotaSnapshot
            var clearedPaths = Set<String>()
            do {
                let paths = try repository.audioRelativePaths()
                snapshot = HistoryAudioQuota.reconcile(
                    roots: roots,
                    referencedPaths: paths,
                    requestedDeletions: Set(try repository.pendingAudioDeletionPaths()),
                    limitBytes: requestedLimitBytes.withLock { $0 },
                    protectedPaths: protectedPaths,
                    clearReferences: { path in
                        try repository.clearAudioReferences(path: path)
                        clearedPaths.insert(path)
                    }
                )
                snapshot.cleanupDeferred = !(try repository.pendingAudioDeletionPaths()).isEmpty
            } catch {
                snapshot = HistoryAudioQuotaSnapshot(measurementFailed: true)
            }
            let databaseBytes = try? (repository as? HistoryRepository)?.databaseStorageBytes()
            let logUsage = logFileStore.storageUsage()
            let clearedAudioPaths = clearedPaths
            DispatchQueue.main.async {
                guard let self else { return }
                if !self.audioQuotaRefreshRequested {
                    self.databaseStorageBytes = databaseBytes
                    self.diagnosticStorageBytes = logUsage.bytes
                    self.diagnosticStorageIssue = logUsage.cleanupFailed || logUsage.overLimit
                    self.audioQuotaSnapshot = snapshot
                }
                let updatedEntries = self.allEntries.map { entry in
                    var updatedEntry = entry
                    for path in [entry.audioRelativePath, entry.transcriptAudioRelativePath].compactMap({ $0 })
                    where clearedAudioPaths.contains(path) {
                        updatedEntry = updatedEntry.removingAudioArchiveReference(path: path)
                    }
                    return updatedEntry
                }
                if updatedEntries != self.allEntries {
                    self.allEntries = updatedEntries
                    self.refreshEntryIndexes()
                    self.publishVisibleEntries()
                }
                self.audioQuotaRefreshInProgress = false
                if self.audioQuotaRefreshRequested { self.applyAudioStorageQuotaIfNeeded() }
            }
        }
    }

    private func releaseImportedAudio(path: String) {
        importedAudioUses.removeValue(forKey: path)
        audioArchive.releaseImportedArchive(path: path)
    }

    private func removeUnreferencedAudio(paths: [String]) {
        guard audioArchive is HistoryAudioArchiveService else {
            paths.forEach(audioArchive.removeArchive(relativePath:))
            return
        }
        do {
            try repository.scheduleAudioDeletion(paths: paths)
            applyAudioStorageQuotaIfNeeded()
        } catch {
            audioQuotaSnapshot.cleanupFailed = true
        }
    }

    func currentAudioArchiveStorageStats() -> HistoryAudioStorageStats {
        HistoryAudioStorageStats(
            storedFileCount: audioQuotaSnapshot.fileCount,
            totalBytes: audioQuotaSnapshot.totalBytes,
            measurementFailed: audioQuotaSnapshot.measurementFailed
        )
    }

    func currentAudioArchiveStorageStats(completion: @escaping (HistoryAudioStorageStats) -> Void) {
        applyAudioStorageQuotaIfNeeded()
        completion(currentAudioArchiveStorageStats())
    }

    func applyDictionaryCorrectedTerms(_ correctedTermsByHistoryID: [UUID: [String]]) {
        guard !correctedTermsByHistoryID.isEmpty else { return }

        var didChange = false
        for (historyID, correctedTerms) in correctedTermsByHistoryID {
            guard let existingEntry = entry(id: historyID) else { continue }
            let merged = mergeUniqueTerms(
                existing: existingEntry.dictionaryCorrectedTerms,
                incoming: correctedTerms
            )
            guard merged != existingEntry.dictionaryCorrectedTerms else { continue }
            let updatedEntry = existingEntry.updatingDictionaryCorrectedTerms(merged)
            cacheUpdatedEntry(updatedEntry)
            persistEntry(updatedEntry)
            didChange = true
        }

        guard didChange else { return }
        refreshEntryIndexes()
        publishVisibleEntries()
    }

    func applyDictionaryCorrectionResult(
        historyID: UUID,
        updatedText: String,
        correctedTerms: [String],
        correctionSnapshots: [DictionaryCorrectionSnapshot]
    ) {
        guard let existingEntry = entry(id: historyID) else { return }

        let trimmedUpdatedText = updatedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let mergedTerms = mergeUniqueTerms(
            existing: existingEntry.dictionaryCorrectedTerms,
            incoming: correctedTerms
        )
        let mergedSnapshots = mergeCorrectionSnapshots(
            existing: existingEntry.dictionaryCorrectionSnapshots,
            incoming: correctionSnapshots
        )

        guard !trimmedUpdatedText.isEmpty else { return }
        let didTextChange = trimmedUpdatedText != existingEntry.text
        let didTermsChange = mergedTerms != existingEntry.dictionaryCorrectedTerms
        let didSnapshotsChange = mergedSnapshots != existingEntry.dictionaryCorrectionSnapshots
        guard didTextChange || didTermsChange || didSnapshotsChange else { return }

        let updatedEntry = existingEntry.updatingDictionaryCorrectionResult(
            text: trimmedUpdatedText,
            dictionaryCorrectedTerms: mergedTerms,
            dictionaryCorrectionSnapshots: mergedSnapshots
        )
        cacheUpdatedEntry(updatedEntry)

        refreshEntryIndexes()
        publishVisibleEntries()
        persistEntry(updatedEntry)
    }

    func replaceDictionaryCorrectionResult(
        historyID: UUID,
        updatedText: String,
        correctedTerms: [String],
        correctionSnapshots: [DictionaryCorrectionSnapshot]
    ) {
        guard let existingEntry = entry(id: historyID) else { return }

        let trimmedUpdatedText = updatedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let mergedTerms = mergeUniqueTerms(
            existing: existingEntry.dictionaryCorrectedTerms,
            incoming: correctedTerms
        )

        guard !trimmedUpdatedText.isEmpty else { return }
        let didTextChange = trimmedUpdatedText != existingEntry.text
        let didTermsChange = mergedTerms != existingEntry.dictionaryCorrectedTerms
        let didSnapshotsChange = correctionSnapshots != existingEntry.dictionaryCorrectionSnapshots
        guard didTextChange || didTermsChange || didSnapshotsChange else { return }

        let updatedEntry = existingEntry.updatingDictionaryCorrectionResult(
            text: trimmedUpdatedText,
            dictionaryCorrectedTerms: mergedTerms,
            dictionaryCorrectionSnapshots: correctionSnapshots
        )
        cacheUpdatedEntry(updatedEntry)

        refreshEntryIndexes()
        publishVisibleEntries()
        persistEntry(updatedEntry)
    }

    @discardableResult
    func updateTranscriptSummary(_ summary: TranscriptSummarySnapshot?, for entryID: UUID) -> TranscriptionHistoryEntry? {
        guard let existingEntry = entry(id: entryID) else { return nil }
        let updatedEntry = existingEntry.updatingTranscriptSummary(summary)
        cacheUpdatedEntry(updatedEntry)
        refreshEntryIndexes()
        publishVisibleEntries()
        persistEntry(updatedEntry)
        return updatedEntry
    }

    @discardableResult
    func updateTranscriptSummaryStale(_ isStale: Bool, for entryID: UUID) -> TranscriptionHistoryEntry? {
        guard let existingEntry = entry(id: entryID) else { return nil }
        let updatedEntry = existingEntry.updatingTranscriptSummaryStale(isStale)
        cacheUpdatedEntry(updatedEntry)
        refreshEntryIndexes()
        publishVisibleEntries()
        persistEntry(updatedEntry)
        return updatedEntry
    }

    @discardableResult
    func updateSummaryChatMessages(_ messages: [TranscriptSummaryChatMessage], for entryID: UUID) -> TranscriptionHistoryEntry? {
        guard let existingEntry = entry(id: entryID) else { return nil }
        let updatedEntry = existingEntry.updatingSummaryChatMessages(messages)
        cacheUpdatedEntry(updatedEntry)
        refreshEntryIndexes()
        publishVisibleEntries()
        persistEntry(updatedEntry)
        return updatedEntry
    }

    @discardableResult
    func updateTranscriptSegments(
        _ segments: [TranscriptSegment],
        for entryID: UUID,
        allowsEmptyText: Bool = false
    ) -> TranscriptionHistoryEntry? {
        guard let existingEntry = entry(id: entryID) else { return nil }
        let text = TranscriptFormatter.joinedText(for: segments)
        let canStoreEmptyMeetingTranscript = allowsEmptyText
            && existingEntry.meetingCaptureMode != nil
            && existingEntry.audioRelativePath != nil
        guard canStoreEmptyMeetingTranscript
            || !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        let updatedEntry = existingEntry.updatingTranscriptSegments(segments, text: text)
        cacheUpdatedEntry(updatedEntry)
        refreshEntryIndexes()
        publishVisibleEntries()
        persistEntry(updatedEntry)
        return updatedEntry
    }

    @discardableResult
    func updateTranscriptionChatMessages(_ messages: [TranscriptSummaryChatMessage], for entryID: UUID) -> TranscriptionHistoryEntry? {
        guard let existingEntry = entry(id: entryID) else { return nil }
        let updatedEntry = existingEntry.updatingTranscriptionChatMessages(messages)
        cacheUpdatedEntry(updatedEntry)
        refreshEntryIndexes()
        publishVisibleEntries()
        persistEntry(updatedEntry)
        return updatedEntry
    }

    @discardableResult
    func updateOutputDestination(
        for entryID: UUID,
        focusedAppName: String?,
        focusedAppBundleID: String?,
        browserURLHost: String?,
        browserURLOrigin: String?
    ) -> TranscriptionHistoryEntry? {
        guard let existingEntry = entry(id: entryID) else { return nil }
        let updatedEntry = existingEntry.updatingOutputDestination(
            focusedAppName: focusedAppName,
            focusedAppBundleID: focusedAppBundleID,
            browserURLHost: browserURLHost,
            browserURLOrigin: browserURLOrigin
        )
        cacheUpdatedEntry(updatedEntry)
        refreshEntryIndexes()
        publishVisibleEntries()
        persistEntry(updatedEntry)
        return updatedEntry
    }

    @discardableResult
    func updateTranscriptionEntry(
        _ entryID: UUID,
        text: String,
        createdAt: Date,
        audioDurationSeconds: TimeInterval?,
        transcriptionProcessingDurationSeconds: TimeInterval?,
        llmDurationSeconds: TimeInterval?,
        whisperWordTimings: [WhisperHistoryWordTiming]?,
        senseVoiceMetadata: SenseVoiceTranscriptMetadata?,
        transcriptionChatMessages: [TranscriptSummaryChatMessage],
        dictionaryHitTerms: [String],
        dictionaryCorrectedTerms: [String],
        dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot] = [],
        dictionarySuggestedTerms: [DictionarySuggestionSnapshot]
    ) -> TranscriptionHistoryEntry? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let existingEntry = entry(id: entryID)
        else {
            return nil
        }

        let updatedEntry = existingEntry.updatingTranscriptionEntry(
            text: trimmed,
            createdAt: createdAt,
            audioDurationSeconds: audioDurationSeconds,
            transcriptionProcessingDurationSeconds: transcriptionProcessingDurationSeconds,
            llmDurationSeconds: llmDurationSeconds,
            whisperWordTimings: whisperWordTimings,
            senseVoiceMetadata: senseVoiceMetadata,
            transcriptionChatMessages: transcriptionChatMessages,
            dictionaryHitTerms: dictionaryHitTerms,
            dictionaryCorrectedTerms: dictionaryCorrectedTerms,
            dictionaryCorrectionSnapshots: dictionaryCorrectionSnapshots,
            dictionarySuggestedTerms: dictionarySuggestedTerms
        )
        cacheUpdatedEntry(updatedEntry)
        refreshEntryIndexes()
        publishVisibleEntries()
        persistEntry(updatedEntry)
        return updatedEntry
    }

    @discardableResult
    private func persistEntry(_ entry: TranscriptionHistoryEntry) -> Bool {
        let interruptedReload = invalidatePendingLoads()
        var didPersist = false
        do {
            try repository.upsert(entry)
            didPersist = true
        } catch {
            VoxtLog.historyWarning("History entry persistence failed. entryID=\(entry.id.uuidString), error=\(error.localizedDescription)")
        }
        if interruptedReload, didPersist {
            reloadAsync()
        }
        return didPersist
    }

    private var historyCleanupEnabled: Bool {
        defaults.object(forKey: AppPreferenceKey.historyCleanupEnabled) as? Bool ?? true
    }

    private var historyRetentionPeriod: HistoryRetentionPeriod {
        let raw = defaults.string(forKey: AppPreferenceKey.historyRetentionPeriod)
        return HistoryRetentionPeriod(rawValue: raw ?? "") ?? .ninetyDays
    }

    private var historyRetentionCount: HistoryRetentionCount {
        let raw = defaults.string(forKey: AppPreferenceKey.historyRetentionCount)
        return HistoryRetentionCount(rawValue: raw ?? "") ?? .unlimited
    }

    private func applyReloadedEntries(
        _ decodedEntries: [TranscriptionHistoryEntry],
        resetPagination: Bool
    ) {
        allEntries = decodedEntries
            .filter { $0.kind != .transcript }
            .sorted { $0.createdAt > $1.createdAt }
        totalEntryCount = allEntries.count

        let targetLoadedCount: Int
        if resetPagination || loadedCount == 0 {
            targetLoadedCount = pageSize
        } else {
            targetLoadedCount = max(loadedCount, pageSize)
        }

        loadedCount = min(targetLoadedCount, allEntries.count)
        refreshEntryIndexes()
        publishVisibleEntries()
    }

    private func applyLoadedEntries(
        _ loadedEntries: [TranscriptionHistoryEntry],
        totalCount: Int,
        resetPagination: Bool
    ) {
        allEntries = uniqueSortedEntries(loadedEntries.filter { $0.kind != .transcript })
        totalEntryCount = max(0, totalCount)

        let targetLoadedCount: Int
        if resetPagination || loadedCount == 0 {
            targetLoadedCount = min(pageSize, allEntries.count)
        } else {
            targetLoadedCount = min(max(loadedCount, pageSize), allEntries.count)
        }

        loadedCount = targetLoadedCount
        refreshEntryIndexes()
        publishVisibleEntries()
        cleanupRetainedEntriesIfNeeded()
        applyAudioStorageQuotaIfNeeded()
    }

    private func cleanupRetainedEntriesIfNeeded(referenceDate: Date = Date()) {
        guard historyCleanupEnabled else { return }

        var removedEntries: [TranscriptionHistoryEntry] = []

        if let days = historyRetentionPeriod.days {
            let cutoff = Calendar.current.date(byAdding: .day, value: -days, to: referenceDate) ?? referenceDate
            let removedByAge = (
                try? repository.deleteEntries(
                    olderThan: cutoff,
                    kinds: retentionCleanupKinds
                )
            ) ?? []
            removedEntries.append(contentsOf: removedByAge)
        }

        if let keepCount = historyRetentionCount.count {
            for kind in retentionCleanupKinds.sorted(by: { $0.rawValue < $1.rawValue }) {
                let removedByCount = (
                    try? repository.deleteEntries(
                        keepingNewest: keepCount,
                        kind: kind
                    )
                ) ?? []
                removedEntries.append(contentsOf: removedByCount)
            }
        }

        guard !removedEntries.isEmpty else { return }
        let interruptedReload = invalidatePendingLoads()
        removeUnreferencedAudio(paths: removedEntries.flatMap { [$0.audioRelativePath, $0.transcriptAudioRelativePath].compactMap { $0 } })
        let removedIDs = Set(removedEntries.map(\.id))
        allEntries.removeAll { removedIDs.contains($0.id) }
        totalEntryCount = max(0, totalEntryCount - removedIDs.count)
        loadedCount = min(loadedCount, allEntries.count)
        refreshEntryIndexes()
        publishVisibleEntries()
        if interruptedReload {
            reloadAsync()
        }
    }

    private func refreshEntryIndexes() {
        entriesByKind = Dictionary(grouping: allEntries, by: \.kind)
    }

    private func publishVisibleEntries() {
        entries = Array(allEntries.prefix(loadedCount))
    }

    @discardableResult
    private func invalidatePendingLoads() -> Bool {
        let interruptedReload = isLoading
        reloadGeneration += 1
        nextPageRequestGeneration += 1
        isLoading = false
        isLoadingNextPage = false
        return interruptedReload
    }

    private func cacheUpdatedEntry(_ entry: TranscriptionHistoryEntry) {
        if let index = allEntries.firstIndex(where: { $0.id == entry.id }) {
            allEntries[index] = entry
        } else {
            allEntries.append(entry)
        }
        allEntries = uniqueSortedEntries(allEntries)
        if loadedCount == 0 {
            loadedCount = min(pageSize, allEntries.count)
        } else {
            loadedCount = min(max(loadedCount, min(pageSize, allEntries.count)), allEntries.count)
        }
    }

    private func mergeLoadedEntries(_ entries: [TranscriptionHistoryEntry]) {
        allEntries = uniqueSortedEntries(allEntries + entries.filter { $0.kind != .transcript })
    }

    private func removeCachedEntry(id: UUID) {
        allEntries.removeAll { $0.id == id }
    }

    private func uniqueSortedEntries(_ entries: [TranscriptionHistoryEntry]) -> [TranscriptionHistoryEntry] {
        var seen = Set<UUID>()
        return entries
            .sorted { $0.createdAt > $1.createdAt }
            .filter { seen.insert($0.id).inserted }
    }

    private func forEachHistoryBatch(_ body: ([TranscriptionHistoryEntry]) -> Void) {
        let batchSize = 500
        var offset = 0
        while true {
            let batch = allEntriesBatch(limit: batchSize, offset: offset)
            guard !batch.isEmpty else { break }
            body(batch)
            offset += batch.count
        }
    }

    private func historyFileURL() throws -> URL {
        if let isolatedHistoryFileURL { return isolatedHistoryFileURL }
        let appSupport = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("transcription-history.json")
    }

    private func legacyHistoryFileExists() -> Bool {
        (try? historyFileURL()).map { fileManager.fileExists(atPath: $0.path) } ?? false
    }

    private func mergeCorrectionSnapshots(
        existing: [DictionaryCorrectionSnapshot],
        incoming: [DictionaryCorrectionSnapshot]
    ) -> [DictionaryCorrectionSnapshot] {
        guard !incoming.isEmpty else { return existing }
        var merged = existing
        var seen = Set(existing)
        for snapshot in incoming where seen.insert(snapshot).inserted {
            merged.append(snapshot)
        }
        return merged.sorted { lhs, rhs in
            if lhs.finalLocation == rhs.finalLocation {
                return lhs.finalLength < rhs.finalLength
            }
            return lhs.finalLocation < rhs.finalLocation
        }
    }

    private func mergeUniqueTerms(existing: [String], incoming: [String]) -> [String] {
        var merged = existing
        let existingNormalized = Set(existing.map(DictionaryStore.normalizeTerm))
        var appended = Set<String>()

        for term in incoming {
            let normalized = DictionaryStore.normalizeTerm(term)
            guard !normalized.isEmpty else { continue }
            guard !existingNormalized.contains(normalized) else { continue }
            guard appended.insert(normalized).inserted else { continue }
            merged.append(term)
        }

        return merged
    }
}
