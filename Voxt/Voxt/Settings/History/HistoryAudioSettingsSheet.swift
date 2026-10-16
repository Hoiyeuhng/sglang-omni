// HistoryAudioSettingsSheet.swift
// Provides History Audio Settings Sheet for history settings.

import SwiftUI

private func localizedHistoryAudioSettings(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct HistoryAudioSettingsSheet: View {
    @Binding var historyCleanupEnabled: Bool
    @Binding var historyRetentionPeriodRaw: String
    @Binding var historyRetentionCountRaw: String
    @Binding var historyAudioStorageLimitMB: Int
    @Binding var historyAudioQuotaMigrationPending: Bool
    @Binding var diagnosticLogStorageLimitMiB: Int
    @Binding var historyAudioStorageEnabled: Bool
    @Binding var historyAudioStorageDisplayPath: String
    @Binding var historyAudioStorageSelectionError: String?
    @Binding var historyAudioExportResultMessage: String?
    @Binding var isPresented: Bool

    let historyRetentionPeriod: HistoryRetentionPeriod
    let historyRetentionCount: HistoryRetentionCount
    let historyAudioStorageStatsSummary: String
    let diagnosticStorageSummary: String
    let databaseStorageSummary: String
    let diagnosticStorageIssue: Bool
    let unavailableAudioRootPaths: [String]
    let onOpenHistoryAudioStorageInFinder: () -> Void
    let onChooseHistoryAudioStorageDirectory: () -> Void
    let onExportAllHistoryAudio: () -> Void
    let onReconnectAudioDirectory: (String) -> Void
    let onStopManagingAudioDirectory: (String) -> Void

    private let dialogMaxHeight: CGFloat = 700
    private let contentMaxHeight: CGFloat = 620

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(localizedHistoryAudioSettings("History Settings"))
                        .font(.title3.weight(.semibold))

                    GeneralSettingsCard(titleText: localizedHistoryAudioSettings("Cleanup")) {
                        GeneralToggleRow(
                            titleText: localizedHistoryAudioSettings("History Cleanup"),
                            isOn: $historyCleanupEnabled
                        )

                        if historyCleanupEnabled {
                            HStack(alignment: .firstTextBaseline, spacing: 10) {
                                Text(localizedHistoryAudioSettings("Retention"))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                SettingsMenuPicker(
                                    selection: $historyRetentionPeriodRaw,
                                    options: HistoryRetentionPeriod.allCases.map { option in
                                        SettingsMenuOption(value: option.rawValue, title: option.title)
                                    },
                                    selectedTitle: historyRetentionPeriod.title,
                                    width: 160
                                )
                            }

                            HStack(alignment: .firstTextBaseline, spacing: 10) {
                                Text(localizedHistoryAudioSettings("Retention Count"))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                SettingsMenuPicker(
                                    selection: $historyRetentionCountRaw,
                                    options: HistoryRetentionCount.allCases.map { option in
                                        SettingsMenuOption(value: option.rawValue, title: option.title)
                                    },
                                    selectedTitle: historyRetentionCount.title,
                                    width: 160
                                )
                            }
                        } else {
                            Text(localizedHistoryAudioSettings("When disabled, Voxt keeps history entries until you delete them manually."))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    GeneralSettingsCard(titleText: localizedHistoryAudioSettings("Audio Storage")) {
                        GeneralToggleRow(
                            titleText: localizedHistoryAudioSettings("Save history audio"),
                            isOn: $historyAudioStorageEnabled
                        )

                        if historyAudioQuotaMigrationPending {
                            Text("Existing audio remains Unlimited until you choose a limit. A finite limit deletes the oldest saved audio, preserving text history.")
                                .font(.caption)
                            HStack {
                                Button("Use 100 MB and clean oldest audio") {
                                    historyAudioQuotaMigrationPending = false
                                    historyAudioStorageLimitMB = 100
                                }
                                Button("Keep Unlimited") {
                                    historyAudioQuotaMigrationPending = false
                                    historyAudioStorageLimitMB = 0
                                }
                            }
                        }
                        HStack {
                            Text("Audio storage limit")
                            Spacer()
                            Picker("Audio storage limit", selection: Binding(
                                get: { historyAudioStorageLimitMB },
                                set: { selectedLimit in
                                    historyAudioQuotaMigrationPending = false
                                    historyAudioStorageLimitMB = selectedLimit
                                }
                            )) {
                                ForEach(HistoryAudioQuotaPreference.choices, id: \.self) { limit in
                                    Text(limit == 0 ? localizedHistoryAudioSettings("Unlimited") : "\(limit) MB").tag(limit)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 160)
                        }
                        Text(historyAudioStorageStatsSummary)
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        if historyAudioStorageEnabled {
                            SettingsPathSelectionRow(
                                title: localizedHistoryAudioSettings("Storage Path"),
                                displayedPath: historyAudioStorageDisplayPath,
                                fallbackPath: HistoryAudioStorageDirectoryManager.defaultRootURL.path,
                                openButtonHelp: localizedHistoryAudioSettings("Open folder"),
                                chooseButtonTitle: localizedHistoryAudioSettings("Choose"),
                                onOpen: onOpenHistoryAudioStorageInFinder,
                                onChoose: onChooseHistoryAudioStorageDirectory
                            )

                            Text(localizedHistoryAudioSettings("New history audio is stored here. Previous locations remain managed; switching does not move files."))
                                .font(.caption)
                                .foregroundStyle(.secondary)

                        } else {
                            Text(localizedHistoryAudioSettings("When disabled, history items will not keep audio files."))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        if let historyAudioStorageSelectionError, !historyAudioStorageSelectionError.isEmpty {
                            Text(historyAudioStorageSelectionError)
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                    }

                    GeneralSettingsCard(titleText: localizedHistoryAudioSettings("Other Local Storage")) {
                        HStack {
                            Text("Diagnostic logs")
                            Spacer()
                            Text(diagnosticStorageSummary)
                            Picker("Diagnostic log limit", selection: $diagnosticLogStorageLimitMiB) {
                                ForEach(VoxtLogFileStore.quotaChoices, id: \.self) { limit in
                                    Text(limit == 0 ? localizedHistoryAudioSettings("Unlimited") : "\(limit) MiB").tag(limit)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 130)
                        }
                        Text("Includes current, rotated and legacy Voxt logs. Default: 12 MiB. Logs rotate independently of audio.")
                            .font(.caption)
                        if diagnosticStorageIssue {
                            Text("Log cleanup or writing failed; storage may remain over limit.").foregroundStyle(.red)
                        }
                        HStack {
                            Text("Database files (shared)")
                            Spacer()
                            Text(databaseStorageSummary)
                        }
                        Text("Includes SQLite, WAL and SHM for history, notes and dictionaries. Text and notes are not quota caches.")
                            .font(.caption)
                        Text("Audio imports and active files can temporarily exceed the limit. A recording larger than the limit is not saved; its text is retained.")
                            .font(.caption)
                    }

                    if !unavailableAudioRootPaths.isEmpty {
                        GeneralSettingsCard(titleText: localizedHistoryAudioSettings("Audio folders need attention")) {
                            Text("Total usage is unavailable. Automatic capacity cleanup is paused; explicit deletions in accessible folders can still finish.")
                                .font(.caption)
                            ForEach(unavailableAudioRootPaths, id: \.self) { path in
                                Text(path).font(.caption).textSelection(.enabled)
                                HStack {
                                    Button("Reconnect folder") { onReconnectAudioDirectory(path) }
                                    if URL(fileURLWithPath: path).standardizedFileURL.path != historyAudioStorageDisplayPath {
                                        Button("Stop managing folder") { onStopManagingAudioDirectory(path) }
                                    }
                                }
                            }
                            Text("Stopping management leaves files and text history unchanged. That folder is excluded from the quota and its audio cannot be played until you select it again.")
                                .font(.caption)
                        }
                    }

                    if historyAudioStorageEnabled {
                        GeneralSettingsCard(titleText: localizedHistoryAudioSettings("Export")) {
                            HStack(spacing: 10) {
                                Button(localizedHistoryAudioSettings("Export Audio")) {
                                    onExportAllHistoryAudio()
                                }
                                .buttonStyle(SettingsPillButtonStyle())

                                VStack(alignment: .leading, spacing: 4) {
                                    Text(historyAudioStorageStatsSummary)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)

                                    Text(localizedHistoryAudioSettings("Copies every saved history audio file into a folder you choose."))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }

                            if let historyAudioExportResultMessage, !historyAudioExportResultMessage.isEmpty {
                                Text(historyAudioExportResultMessage)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: contentMaxHeight)

            SettingsDialogActionRow {
                Button(localizedHistoryAudioSettings("Done")) {
                    isPresented = false
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(width: 560, maxHeight: dialogMaxHeight, onClose: { isPresented = false })
    }
}
