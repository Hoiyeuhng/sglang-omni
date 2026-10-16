# Logging

Voxt uses swift-log with Apple Unified Logging and the existing local rolling file store. Both sinks accept only canonical `RecordingDiagnosticEvent` JSON. Unknown fields, arbitrary messages and caller metadata never reach either sink. Existing raw lines are also excluded from viewing and export.

Use `VoxtLog.technical(event, category:)` for always-available safe lifecycle and failure records. Use `VoxtLog.hotkey(event)` or `VoxtLog.model(event)` for the independent developer debug switches. Use `RecordingDiagnostics.shared.record(event)` for opt-in correlated recording stages. The event schema admits only closed classifications, random correlation identifiers, Boolean flags and numeric metadata; never add a free-form string field for user/model/network content.

Legacy string APIs remain as migration compatibility: warning/error calls emit only a `legacyDiagnostic` category/severity placeholder without evaluating the message; info/debug/notice calls emit nothing. New technical information must use the typed APIs. The placeholder explicitly cannot distinguish individual legacy failures. Removed string-based hotkey/model debug APIs prevent accidentally adding apparently useful but discarded events. Regex redaction utilities are not the persistence privacy boundary.

The handler contract runner exercises actual swift-log dispatch into both sinks, including legacy warnings, untrusted metadata, and typed events. The recording runner exercises the recorder, rolling storage, viewer/export, runtime client, VAD and Remote ASR completion handling. See `docs/RecordingDiagnostics.md` for isolated commands and validation limits.
