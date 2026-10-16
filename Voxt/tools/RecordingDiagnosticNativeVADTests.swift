import Foundation

extension RecordingDiagnosticRuntimeTests {
    static func verifyNativeVAD(
        server: URL, scratch: URL, recorder: RecordingDiagnosticRecorder, output: RuntimeTestOutput
    ) async throws {
        let firstSession = UUID()
        let secondSession = UUID()
        recorder.record(.init(.sessionStarted, sessionID: firstSession))
        recorder.record(.init(.sessionStarted, sessionID: secondSession))
        let configuration = OmniBackendConfiguration(runtimeExecutable: server, startupTimeoutSeconds: 5)
        let shared = OmniSharedModelRuntime(kind: .sileroVAD, configuration: { configuration }, diagnostics: recorder)
        async let firstAcquire = shared.acquire(modelDirectory: scratch, diagnosticSessionID: firstSession)
        async let secondAcquire = shared.acquire(modelDirectory: scratch, diagnosticSessionID: secondSession)
        let (firstEndpoint, secondEndpoint) = try await (firstAcquire, secondAcquire)
        precondition(firstEndpoint.diagnosticRuntimeID != nil)
        precondition(firstEndpoint.diagnosticRuntimeID == secondEndpoint.diagnosticRuntimeID)
        let runtimeID = firstEndpoint.diagnosticRuntimeID
        let first = OmniVoiceActivityStream(endpoint: firstEndpoint, diagnosticSessionID: firstSession, diagnostics: recorder)
        let second = OmniVoiceActivityStream(endpoint: secondEndpoint, diagnosticSessionID: secondSession, diagnostics: recorder)
        async let firstProbability = first.probability(samples16k: [Float](repeating: 0.1, count: 512))
        async let secondProbability = second.probability(samples16k: [Float](repeating: 0.2, count: 1024))
        let probabilities = try await [firstProbability, secondProbability]
        precondition(probabilities == [0.75, 0.75])
        _ = try await first.probability(samples16k: [Float](repeating: 0.3, count: 256))
        await first.close()
        await first.close()
        await second.close()
        let failed = OmniVoiceActivityStream(endpoint: firstEndpoint, diagnosticSessionID: firstSession, diagnostics: recorder)
        do {
            _ = try await failed.probability(samples16k: [-1])
            preconditionFailure("malformed native VAD reply must fail")
        } catch {}
        await failed.close()
        await failed.close()
        let cancelled = OmniVoiceActivityStream(endpoint: secondEndpoint, diagnosticSessionID: secondSession, diagnostics: recorder)
        let pending = Task { try await cancelled.probability(samples16k: [-2]) }
        let marker = scratch.appendingPathComponent("vad-stalled")
        let deadline = ContinuousClock.now + .seconds(5)
        while !FileManager.default.fileExists(atPath: marker.path), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        precondition(FileManager.default.fileExists(atPath: marker.path), "fake VAD server did not receive pending exchange")
        await cancelled.close()
        do { _ = try await pending.value; preconditionFailure("closed pending VAD socket must fail") } catch {}
        await cancelled.close()
        let empty = OmniVoiceActivityStream(endpoint: firstEndpoint, diagnosticSessionID: firstSession, diagnostics: recorder)
        await empty.close()
        let beforeDisabled = output.snapshot().count
        recorder.setEnabled(false)
        let disabled = OmniVoiceActivityStream(endpoint: firstEndpoint, diagnostics: recorder)
        _ = try await disabled.probability(samples16k: [0])
        await disabled.close()
        precondition(output.snapshot().count == beforeDisabled)
        recorder.setEnabled(true)
        await shared.release()
        await shared.release()
        let events = try output.snapshot().map { try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data($0.utf8)) }
        let vad = events.filter { $0.backend == .silero && $0.runtimeID == runtimeID }
        let starts = vad.filter { $0.kind == .requestStarted }
        let completions = vad.filter { $0.kind == .requestCompleted }
        precondition(starts.count == 5 && Set(starts.compactMap(\.requestID)).count == 5)
        precondition(vad.filter { $0.kind == .vadRuntimeAcquired }.count == 2)
        precondition(vad.filter { $0.kind == .vadStreamReady }.count == 2)
        precondition(completions.count == 5 && vad.filter { $0.kind == .requestCleanup }.count == 5)
        precondition(completions.filter { $0.outcome == .success }.count == 2)
        precondition(completions.contains { $0.outcome == .failed && $0.failure == .runtime })
        precondition(completions.contains { $0.outcome == .cancelled && $0.failure == .cancelled })
        precondition(completions.contains { $0.outcome == .empty && $0.count == 0 })
        precondition(completions.contains { $0.sessionID == firstSession && $0.count == 2 && $0.sampleCount == 768 })
        precondition(completions.contains { $0.sessionID == secondSession && $0.count == 1 && $0.sampleCount == 1024 })
        for start in starts {
            let related = vad.filter { $0.requestID == start.requestID }
            precondition(related.allSatisfy { $0.sessionID == start.sessionID })
            precondition(related.filter { $0.kind == .requestCompleted }.count == 1)
            precondition(related.filter { $0.kind == .requestCleanup }.count == 1)
        }
        let sharedEvents = events.filter { $0.runtimeID == runtimeID && $0.kind.rawValue.hasPrefix("runtime") }
        precondition(!sharedEvents.isEmpty && sharedEvents.allSatisfy { $0.sessionID == nil && $0.requestID == nil })
        precondition(!output.snapshot().joined().contains("private"))
        precondition(!output.snapshot().joined().contains("token=secret"))
        precondition(!output.snapshot().joined().contains(scratch.path))
        print("PASS native VAD socket diagnostics: two sessions, shared runtime, five request lifecycles, first reply, exact completed sample counts, malformed reply, cancellation, empty close, deduplication, privacy and OFF gate")
    }
}

nonisolated enum OmniASRBackend {
    static let launchSettings: Int? = nil
    static func configuration(for kind: OmniASRModelKind) -> OmniBackendConfiguration? { nil }
}
