import Foundation

nonisolated enum RecordingDiagnostics {
    static let shared = RecordingDiagnosticRecorder(defaults: .standard) { line in
        guard !VoxtRuntimeEnvironment.isRunningUnitTests else { return }
        VoxtLogFileStore.shared.append(line)
    }
}
