#!/bin/bash
set -euo pipefail

voxt_directory="$(cd "$(dirname "$0")/.." && pwd)"
build_directory="$(mktemp -d "${TMPDIR:-/tmp}/voxt-sample-contracts.XXXXXX")"
trap 'rm -rf "$build_directory"' EXIT

cat > "$build_directory/SampleContractMain.swift" <<'SWIFT'
import Foundation

@main struct SampleContractMain {
    static func main() async throws {
        var passed = 0
        for (name, test) in HistoryDiagnosticSampleChecks.cases {
            try test()
            print("PASS \(name)")
            passed += 1
        }
        try await HistoryDiagnosticSampleChecks.taskCancellation()
        print("PASS task-cancellation")
        print("RESULT: \(passed + 1) passed, 0 failed")
    }
}
SWIFT

sources=(
    "$voxt_directory/Voxt/Core/History/HistoryDiagnosticSampleExport.swift"
    "$voxt_directory/Voxt/Core/History/HistoryAudioArchiveSupport.swift"
    "$voxt_directory/VoxtTests/TestSupport/HistoryDiagnosticSampleChecks.swift"
    "$voxt_directory/VoxtTests/TestSupport/HistoryDiagnosticSamplePermissionChecks.swift"
    "$build_directory/SampleContractMain.swift"
)
# Include the shared audio-use guard when validating with the storage quota PR.
if [[ -f "$voxt_directory/Voxt/Core/History/HistoryAudioQuota.swift" ]]; then
    sources+=("$voxt_directory/Voxt/Core/History/HistoryAudioQuota.swift")
fi

swiftc -swift-version 5 -default-isolation MainActor -D VOXT_STANDALONE_TESTS \
    -module-cache-path "$build_directory/module-cache" \
    "${sources[@]}" \
    -o "$build_directory/sample-contracts"
"$build_directory/sample-contracts"
