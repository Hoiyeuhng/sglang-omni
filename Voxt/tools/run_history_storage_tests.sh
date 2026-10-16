#!/bin/bash
set -euo pipefail
voxt_root="$(cd "$(dirname "$0")/.." && pwd)"
validation_build="$(mktemp -d "${TMPDIR:-/tmp}/voxt-storage-contract.XXXXXX")"
trap 'rm -rf "$validation_build"' EXIT
swiftc -swift-version 5 -default-isolation MainActor -D VOXT_STORAGE_CONTRACT_TESTS \
  -module-cache-path "$validation_build/module-cache" \
  "$voxt_root/Voxt/Core/History/HistoryAudioQuota.swift" \
  "$voxt_root/Voxt/Core/History/HistoryAudioQuotaReconciliation.swift" \
  "$voxt_root/Voxt/Core/Logging/VoxtLogFileStore.swift" \
  "$voxt_root/VoxtTests/HistoryAudioQuotaTests.swift" \
  "$voxt_root/VoxtTests/TestSupport/HistoryAudioQuotaConcurrencyChecks.swift" \
  -o "$validation_build/storage-contract"
"$validation_build/storage-contract"
