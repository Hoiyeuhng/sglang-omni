#!/bin/bash
set -euo pipefail
project_directory="$(cd "$(dirname "$0")/.." && pwd)"
scratch_directory="$(mktemp -d "${TMPDIR:-/tmp}/voxt-diagnostics-tests.XXXXXX")"
trap 'rm -rf "$scratch_directory"' EXIT
swiftc -swift-version 5 -default-isolation MainActor -module-cache-path "$scratch_directory/module-cache" \
  "$project_directory/Voxt/Core/Logging/RecordingDiagnosticRecorder.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtLogFileStore.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtLogExportStore.swift" \
  "$project_directory/Voxt/Core/Utilities/MainActorSync.swift" \
  "$project_directory/tools/RecordingDiagnosticContractTests.swift" \
  -o "$scratch_directory/diagnostics-tests"
"$scratch_directory/diagnostics-tests"
swiftc -swift-version 5 -default-isolation MainActor -module-cache-path "$scratch_directory/module-cache" \
  "$project_directory/Voxt/Core/Logging/RecordingDiagnosticRecorder.swift" \
  "$project_directory/Voxt/Core/Logging/RecordingDiagnostics.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtLogFileStore.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtRuntimeEnvironment.swift" \
  "$project_directory/Voxt/Transcription/OmniASRRuntime.swift" \
  "$project_directory/Voxt/Transcription/OmniTranscriptionRequest.swift" \
  "$project_directory/tools/RecordingDiagnosticRuntimeTests.swift" \
  -o "$scratch_directory/runtime-tests"
"$scratch_directory/runtime-tests" "$project_directory/VoxtTests/Fixtures/Audio/qwen-official/qwen_audio_short_en.flac"
swiftc -swift-version 5 -default-isolation MainActor -module-cache-path "$scratch_directory/module-cache" \
  "$project_directory/Voxt/Core/Logging/RecordingDiagnosticRecorder.swift" \
  "$project_directory/Voxt/Core/Logging/RecordingDiagnostics.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtLogFileStore.swift" \
  "$project_directory/Voxt/Core/Logging/VoxtRuntimeEnvironment.swift" \
  "$project_directory/Voxt/Core/Transcription/ASRVoiceActivityPlanning.swift" \
  "$project_directory/Voxt/App/Recording/RecordingVoiceActivitySupport.swift" \
  "$project_directory/tools/RecordingDiagnosticVADTests.swift" \
  -o "$scratch_directory/vad-tests"
"$scratch_directory/vad-tests"
swiftc -swift-version 5 -default-isolation MainActor -module-cache-path "$scratch_directory/module-cache" \
  "$project_directory/Voxt/Core/Logging/RecordingDiagnosticRecorder.swift" \
  "$project_directory/Voxt/Core/Utilities/TrackedTaskStore.swift" \
  "$project_directory/Voxt/Transcription/RemoteASR/RemoteASRPreviewController.swift" \
  "$project_directory/Voxt/Transcription/RemoteASR/RemoteASRTextSupport.swift" \
  "$project_directory/tools/RecordingDiagnosticRemoteASRTests.swift" \
  -o "$scratch_directory/remote-asr-tests"
"$scratch_directory/remote-asr-tests"
