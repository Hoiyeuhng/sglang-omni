# History Settings

Settings UI for transcription history, audio retention, correction presentation, and detail sheets.

## Responsibilities

- Presents history-related settings and reusable history settings components.
- Configures audio archive behavior and correction display preferences.
- Supports detail sheet content used from settings and history workflows.

`HistorySettingsView` owns filters, selection and async list state; `HistorySettingsData` contains list/deletion values. `HistorySettingsComponents` holds shared history rows and toolbar presentation. Note filters/section controls live in `NoteHistoryControls`; `NoteHistoryRow` retains its own editing/focus state and private footer components.

## Private diagnostic samples

Use a history row's **Export diagnostic sample…** action to preview one record and
choose its transcription, saved audio, or both. Neither is selected initially.
Optional expected transcription is included only when entered. Audio preview does
not play automatically; unavailable audio never starts recording or changes the
history audio preference.

The export sheet lists the exact files. Its manifest allows only schema version,
numeric application version, selected filenames and selected audio byte count.
Custom model names, endpoints, configuration, other history and diagnostic logs
are excluded. Nothing is uploaded.

The worker holds a read descriptor for the selected regular archive, rejects
traversal and symlink archive paths, and streams to a private staging directory in
the chosen destination. Files use mode 0600 and directories 0700 where supported
by the destination filesystem. Publication is atomic and refuses existing targets.
Normal errors and cancellation remove staging files; a cleanup failure shows the
remaining private folder. A crash or forced process termination may leave a hidden
.voxt-sample-*.partial directory in that destination; it can be deleted manually.
The final exported folder belongs to the user and is outside application rotation.
This uses temporary space up to one selected audio copy plus selected text and
manifest; it does not keep an application sample cache.
