#!/bin/bash
set -euo pipefail
project_directory="$(cd "$(dirname "$0")/.." && pwd)"
swift_log_directory="${1:?Pass the swift-log checkout pinned by Package.resolved}"
scratch_directory="$(mktemp -d "${TMPDIR:-/tmp}/voxt-diagnostic-handler-tests.XXXXXX")"
trap 'rm -rf "$scratch_directory"' EXIT
swift_log_sources=()
while IFS= read -r filename; do swift_log_sources+=("$filename"); done < <(find "$swift_log_directory/Sources/Logging" -name '*.swift' -type f)
swiftc -swift-version 6 -module-cache-path "$scratch_directory/module-cache" \
  -package-name swift_log -emit-module -emit-library -module-name Logging "${swift_log_sources[@]}" \
  -emit-module-path "$scratch_directory/Logging.swiftmodule" -o "$scratch_directory/libLogging.dylib"
swiftc -swift-version 5 -default-isolation MainActor -module-cache-path "$scratch_directory/module-cache" \
  -I "$scratch_directory" -L "$scratch_directory" -lLogging -Xlinker -rpath -Xlinker "$scratch_directory" \
  "$project_directory/Voxt/Core/Logging/RecordingDiagnosticRecorder.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtLogFileStore.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtRuntimeEnvironment.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtLogCategory.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtFileLogHandler.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtOSLogHandler.swift" \
  "$project_directory/tools/RecordingDiagnosticHandlerTests.swift" -o "$scratch_directory/handler-tests"
"$scratch_directory/handler-tests"
