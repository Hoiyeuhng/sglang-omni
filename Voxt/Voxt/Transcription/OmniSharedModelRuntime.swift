import Foundation

/// A native runtime server shared by every caller of one model kind (Silero
/// VAD, Sortformer): the first lease starts it, and it stops when the last
/// lease is returned or the app terminates.
actor OmniSharedModelRuntime {
    nonisolated let kind: OmniASRModelKind
    private var runtime: OmniASRRuntime?
    private var modelDirectory: URL?
    private var leases = 0
    private let configuration: @Sendable () -> OmniBackendConfiguration?
    private let diagnostics: RecordingDiagnosticRecorder

    init(
        kind: OmniASRModelKind,
        configuration: (@Sendable () -> OmniBackendConfiguration?)? = nil,
        diagnostics: RecordingDiagnosticRecorder = RecordingDiagnostics.shared
    ) {
        self.kind = kind
        self.diagnostics = diagnostics
        self.configuration = configuration ?? { @Sendable in OmniASRBackend.configuration(for: kind) }
    }

    /// A ready endpoint, held until `release()`.
    func acquire(modelDirectory directory: URL, diagnosticSessionID: UUID? = nil) async throws -> OmniServerEndpoint {
        guard let configuration = configuration() else {
            throw OmniASRRuntimeError.launchFailed("The native runtime is not configured.")
        }
        leases += 1
        // Note (Jiaxin Deng): checking a runtime suspends and another acquire may replace it meanwhile,
        // so re-read it after every suspension and drop only the runtime that was checked.
        while let current = self.runtime {
            let usable = modelDirectory == directory
            if usable, await current.canServe {
                break
            } else if self.runtime === current {
                self.runtime = nil
                modelDirectory = nil
                await current.retire()
            } else {
                continue
            }
        }
        let runtime: OmniASRRuntime
        if let current = self.runtime {
            runtime = current
        } else {
            runtime = OmniASRRuntime(kind: kind, modelDirectory: directory, configuration: configuration, diagnostics: diagnostics)
            self.runtime = runtime
            modelDirectory = directory
        }
        if kind == .sileroVAD {
            diagnostics.record(.init(.vadRuntimeAcquiring, sessionID: diagnosticSessionID,
                runtimeID: runtime.diagnosticRuntimeID, backend: .silero))
        }
        do {
            var endpoint = try await runtime.prepare()
            endpoint.diagnosticRuntimeID = runtime.diagnosticRuntimeID
            if kind == .sileroVAD {
                diagnostics.record(.init(.vadRuntimeAcquired, sessionID: diagnosticSessionID,
                    runtimeID: runtime.diagnosticRuntimeID, backend: .silero, outcome: .success))
            }
            return endpoint
        } catch {
            if kind == .sileroVAD {
                let failure = RecordingDiagnosticEvent.Failure.classify(error)
                diagnostics.record(.init(.vadRuntimeAcquired, sessionID: diagnosticSessionID,
                    runtimeID: runtime.diagnosticRuntimeID, backend: .silero,
                    outcome: failure == .cancelled ? .cancelled : .failed,
                    failure: failure == .unknown ? .runtime : failure))
            }
            await release()
            throw error
        }
    }

    func release() async {
        leases = max(0, leases - 1)
        guard leases == 0, let runtime else { return }
        self.runtime = nil
        modelDirectory = nil
        await runtime.retire()
    }

    func shutdownForApplicationTermination() async {
        leases = 0
        guard let runtime else { return }
        self.runtime = nil
        await runtime.retire()
    }
}
