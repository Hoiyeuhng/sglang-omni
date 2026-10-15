import AVFoundation
import Foundation

@main
struct RecordingDiagnosticRuntimeTests {
    static func main() async throws {
        let suiteName = "Voxt.Diagnostics.Runtime.Tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: scratch) }
        let output = RuntimeTestOutput()
        let recorder = RecordingDiagnosticRecorder(defaults: defaults, sink: { output.append($0) })
        recorder.setEnabled(true)
        let server = scratch.appendingPathComponent("fake-runtime")
        let script = """
        #!/usr/bin/python3
        import http.server
        import json
        import os
        import sys
        import threading
        import time
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *values):
                pass
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                if b'FAIL_REQUEST' in body:
                    self.send_response(422)
                    self.end_headers()
                    self.wfile.write(b'private transcript token=secret /private/model')
                    return
                if b'STALL_REQUEST' in body:
                    time.sleep(1)
                text = '' if b'EMPTY_REQUEST' in body else 'synthetic private transcript'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for event in [{'type': 'transcript.text.delta', 'delta': text}, {'type': 'transcript.text.done', 'text': text}]:
                    self.wfile.write(('data: ' + json.dumps(event) + '\\n\\n').encode())
        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        print(json.dumps({'event': 'ready', 'host': '127.0.0.1', 'port': server.server_port, 'model_name': 'private/custom-model', 'server_pid': os.getpid()}), flush=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        sys.stdin.readline()
        server.shutdown()
        """
        try script.write(to: server, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: server.path)
        let runtime = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch,
            configuration: .init(runtimeExecutable: server, startupTimeoutSeconds: 5), diagnostics: recorder)
        let sessionID = UUID()
        recorder.record(.init(.sessionStarted, sessionID: sessionID))
        _ = try await runtime.prepare()
        let fixture = try AVAudioFile(forReading: URL(fileURLWithPath: CommandLine.arguments[1]))
        let buffer = AVAudioPCMBuffer(pcmFormat: fixture.processingFormat, frameCapacity: AVAudioFrameCount(fixture.length))!
        try fixture.read(into: buffer)
        let samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
        precondition(!samples.isEmpty, "fixture must decode actual samples")
        var request = OmniTranscriptionRequest(samples: samples, sampleRate: Int(fixture.processingFormat.sampleRate),
            language: nil, prompt: "private prompt https://private.example/?token=secret", maxNewTokens: 32,
            stopAtEndOfText: true, stopOnTokenLoop: true)
        let result = try await RecordingDiagnosticContext.$sessionID.withValue(sessionID) {
            try await runtime.transcribe(request)
        }
        precondition(result.text == "synthetic private transcript")
        request.prompt = "EMPTY_REQUEST"
        let empty = try await RecordingDiagnosticContext.$sessionID.withValue(sessionID) { try await runtime.transcribe(request) }
        precondition(empty.text.isEmpty)
        request.prompt = "FAIL_REQUEST"
        do {
            _ = try await RecordingDiagnosticContext.$sessionID.withValue(sessionID) { try await runtime.transcribe(request) }
            preconditionFailure("fake request must fail")
        } catch let error as OmniTranscriptionError {
            precondition(error.errorDescription?.contains("422") == true)
        }
        request.prompt = "STALL_REQUEST"
        let cancelledRequest = request
        let operation = Task {
            try await RecordingDiagnosticContext.$sessionID.withValue(sessionID) { try await runtime.transcribe(cancelledRequest) }
        }
        try await Task.sleep(for: .milliseconds(80))
        operation.cancel()
        do { _ = try await operation.value; preconditionFailure("cancelled request must not succeed") } catch {}
        await runtime.retire()
        let timeout = scratch.appendingPathComponent("stalled-runtime")
        try "#!/bin/sh\nexec /bin/sleep 30\n".write(to: timeout, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: timeout.path)
        let stalled = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch,
            configuration: .init(runtimeExecutable: timeout, startupTimeoutSeconds: 0.1), diagnostics: recorder)
        do { _ = try await stalled.prepare(); preconditionFailure("stalled runtime must time out") } catch {}
        await stalled.retire()
        let failedServer = scratch.appendingPathComponent("failed-runtime")
        try "#!/bin/sh\nprintf '%s\\n' 'private transcript token=secret /Users/test/private-model' >&2\nexit 7\n".write(to: failedServer, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: failedServer.path)
        let failed = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch,
            configuration: .init(runtimeExecutable: failedServer, startupTimeoutSeconds: 1), diagnostics: recorder)
        do { _ = try await failed.prepare(); preconditionFailure("failed process must not become ready") } catch {}
        await failed.retire()
        let lines = output.snapshot()
        let events = try lines.map { try JSONDecoder().decode(RecordingDiagnosticEvent.self, from: Data($0.utf8)) }
        precondition(events.filter { $0.kind == .requestStarted }.count == 4)
        precondition(events.filter { $0.kind == .requestCleanup }.count == 4)
        precondition(events.filter { $0.kind == .requestCompleted }.map(\.outcome) == [.success, .empty, .failed, .cancelled])
        precondition(events.filter { $0.kind == .requestStarted }.allSatisfy { $0.sessionID == sessionID && $0.runtimeID == runtime.diagnosticRuntimeID })
        precondition(events.filter { $0.kind == .runtimeStarted || $0.kind == .runtimeReady }.allSatisfy { $0.sessionID == nil })
        precondition(events.contains { $0.kind == .runtimeFailed && $0.failure == .timeout })
        precondition(events.filter { $0.kind == .runtimeReleased }.count == 3)
        precondition(events.filter { $0.kind == .runtimeExited }.count == 3)
        precondition(events.contains { $0.kind == .runtimeExited && $0.exitStatusCode == 7 && $0.outcome == .failed })
        precondition(events.contains { $0.kind == .runtimeExited && $0.exitStatusCode == 0 && $0.outcome == .success })
        precondition(events.contains { $0.kind == .requestCompleted && $0.httpStatusCode == 422 })
        precondition(!lines.joined().contains("private"))
        precondition(!lines.joined().contains("secret"))
        precondition(!lines.joined().contains(scratch.path))
        print("PASS 19 runtime integration assertions; fixture samples=\(samples.count); fake runtime only")
    }
}

nonisolated final class RuntimeTestOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func append(_ line: String) { lock.lock(); lines.append(line); lock.unlock() }
    func snapshot() -> [String] { lock.lock(); defer { lock.unlock() }; return lines }
}
