import AppKit
import SwiftUI

struct HistoryDiagnosticSampleSheet: View {
    @Environment(\.dismiss) private var dismiss
    let entry: TranscriptionHistoryEntry
    let audioRootURL: URL
    let resolvedAudioURL: URL?

    @StateObject private var playbackController = HistoryAudioPlaybackController(audioURL: nil)
    @State private var includeHistoryText = false
    @State private var includeAudio = false
    @State private var expectedTranscription = ""
    @State private var audioSource: HistoryDiagnosticAudioSource?
    @State private var isCheckingAudio = true
    @State private var isVisible = true
    @State private var isExporting = false
    @State private var isChoosingDestination = false
    @State private var exportTask: Task<URL, Error>?
    @State private var statusMessage: String?
    @State private var remainingPartialURL: URL?
    @State private var cleanupNeedsAttention = false

    private var request: HistoryDiagnosticSampleRequest {
        HistoryDiagnosticSampleRequest(
            historyText: includeHistoryText ? entry.text : nil,
            audio: audioSource,
            includeAudio: includeAudio,
            expectedTranscription: expectedTranscription.isEmpty ? nil : expectedTranscription,
            appVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(AppLocalization.localizedString("Export diagnostic sample"))
                .font(.title3.weight(.semibold))
            Text(AppLocalization.localizedString("Choose content from this record only. Audio, saved final text and expected transcription may contain confidential information. Nothing is uploaded. Share the exported folder only with people you trust."))
                .font(.callout)
                .foregroundStyle(.secondary)
            Toggle(AppLocalization.localizedString("Include saved final text"), isOn: $includeHistoryText)
                .disabled(isExporting)
            Text(AppLocalization.localizedString("Saved final text may include enhancement, translation or rewriting. It is not identified as the original recognizer output."))
                .font(.caption)
                .foregroundStyle(.secondary)
            ScrollView {
                Text(entry.text.isEmpty ? AppLocalization.localizedString("This record has no saved final text.") : entry.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 100)
            .padding(8)
            .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
            Toggle(AppLocalization.localizedString("Include saved audio"), isOn: $includeAudio)
                .disabled(audioSource == nil || isCheckingAudio || isExporting)
            if isCheckingAudio {
                ProgressView(AppLocalization.localizedString("Checking saved audio…"))
                    .controlSize(.small)
            } else if let audioSource {
                Text(AppLocalization.format("Saved audio: %@. Only this record's audio will be exported.", ByteCountFormatter.string(fromByteCount: audioSource.byteCount, countStyle: .file)))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(AppLocalization.localizedString("Saved audio is unavailable for this record. Exporting will not record audio or enable audio saving."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if playbackController.isAvailable {
                HistoryAudioPlayerView(controller: playbackController, compact: true)
            }
            Text(AppLocalization.localizedString("Expected transcription (optional, included if entered)"))
                .font(.callout)
            TextEditor(text: $expectedTranscription)
                .disabled(isExporting)
                .frame(height: 65)
                .border(Color.secondary.opacity(0.2))
                .accessibilityLabel(AppLocalization.localizedString("Expected transcription"))
            Text(AppLocalization.localizedString("Files to export:"))
                .font(.callout.weight(.medium))
            Text(request.fileNames.joined(separator: "\n"))
                .font(.caption.monospaced())
            Text(AppLocalization.localizedString("The manifest contains only a format version, numeric app version, selected filenames, audio byte count and final-text origin. It contains no settings, endpoints or diagnostic logs."))
                .font(.caption)
                .foregroundStyle(.secondary)
            if let statusMessage {
                Text(statusMessage).font(.callout).foregroundStyle(.secondary)
            }
            if let remainingPartialURL {
                Text(remainingPartialURL.path).font(.caption).textSelection(.enabled)
                Button(AppLocalization.localizedString("Show partial export in Finder")) {
                    NSWorkspace.shared.activateFileViewerSelecting([remainingPartialURL])
                }
            }
            HStack {
                if isExporting {
                    ProgressView().controlSize(.small)
                    Text(AppLocalization.localizedString("Exporting…"))
                }
                Spacer()
                Button(AppLocalization.localizedString("Cancel")) {
                    if isExporting { exportTask?.cancel() } else { dismiss() }
                }
                .disabled(isChoosingDestination)
                Button(AppLocalization.localizedString("Choose export folder…"), action: chooseDestination)
                    .disabled(isExporting || isChoosingDestination || cleanupNeedsAttention || (!includeHistoryText && !includeAudio))
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 580)
        .interactiveDismissDisabled(isExporting || isChoosingDestination)
        .task {
            let relativePath = entry.audioRelativePath ?? entry.transcriptAudioRelativePath
            let rootURL = audioRootURL
            guard let selectedAudioURL = resolvedAudioURL else {
                isCheckingAudio = false
                return
            }
            let accessURL = selectedAudioURL
            let didStartAccess = accessURL.startAccessingSecurityScopedResource()
            let preparedSource = await Task.detached(priority: .utility) {
                defer {
                    if didStartAccess { accessURL.stopAccessingSecurityScopedResource() }
                }
                return relativePath.flatMap {
                    try? HistoryDiagnosticAudioSource(rootURL: rootURL, relativePath: $0, resolvedAudioURL: selectedAudioURL)
                }
            }.value
            guard !Task.isCancelled else { return }
            audioSource = preparedSource
            playbackController.loadAudio(audioSource?.previewURL)
            isCheckingAudio = false
        }
        .onDisappear {
            isVisible = false
            exportTask?.cancel()
            playbackController.loadAudio(nil)
        }
    }

    private func chooseDestination() {
        let panel = NSOpenPanel()
        panel.message = AppLocalization.localizedString("Choose where to create the private diagnostic sample folder")
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        isChoosingDestination = true
        panel.begin { response in
            isChoosingDestination = false
            guard isVisible, response == .OK, let destinationURL = panel.url else { return }
            let didStartAccess = destinationURL.startAccessingSecurityScopedResource()
            let snapshot = request
            isExporting = true
            statusMessage = nil
            remainingPartialURL = nil
            cleanupNeedsAttention = false
            let worker = Task.detached(priority: .utility) {
                defer {
                    if didStartAccess { destinationURL.stopAccessingSecurityScopedResource() }
                }
                return try HistoryDiagnosticSampleExporter.export(snapshot, to: destinationURL)
            }
            exportTask = worker
            Task { @MainActor in
                do {
                    let exportedURL = try await worker.value
                    NSWorkspace.shared.activateFileViewerSelecting([exportedURL])
                    isExporting = false
                    exportTask = nil
                    dismiss()
                } catch {
                    isExporting = false
                    exportTask = nil
                    if case HistoryDiagnosticSampleError.cleanupFailed(let url) = error {
                        cleanupNeedsAttention = true
                        remainingPartialURL = url
                        let message = url == nil
                            ? AppLocalization.localizedString("Export failed and cleanup could not be confirmed. The sample folder's current location could not be determined; check the selected destination and any folders moved during export.")
                            : AppLocalization.localizedString("Export failed and a partial private sample could not be removed. Delete the folder shown below when you no longer need it.")
                        statusMessage = message
                        if !isVisible {
                            let alert = NSAlert()
                            alert.messageText = AppLocalization.localizedString("Partial diagnostic sample remains")
                            alert.informativeText = message + (url.map { "\n\n" + $0.path } ?? "")
                            if url != nil { alert.addButton(withTitle: AppLocalization.localizedString("Show partial export in Finder")) }
                            alert.addButton(withTitle: AppLocalization.localizedString("OK"))
                            if alert.runModal() == .alertFirstButtonReturn, let url {
                                NSWorkspace.shared.activateFileViewerSelecting([url])
                            }
                        }
                    } else if error is CancellationError {
                        statusMessage = AppLocalization.localizedString("Export cancelled. No sample was saved.")
                    } else if case HistoryDiagnosticSampleError.privatePermissionsUnavailable = error {
                        statusMessage = AppLocalization.localizedString("This destination cannot enforce private sample permissions. Choose a different local folder. No sample was saved.")
                    } else if case HistoryDiagnosticSampleError.destinationConflict = error {
                        statusMessage = AppLocalization.localizedString("The destination already exists. No existing files were changed. Choose another folder and try again.")
                    } else {
                        statusMessage = AppLocalization.localizedString("Could not export the sample. Check that the audio and destination are available and there is enough free space. No partial sample was saved.")
                    }
                }
            }
        }
    }
}
