#!/usr/bin/env bash
# Stand-in for the upstream cleanup script. Prints which version is running.
set -euo pipefail
echo "SCRIPT_VERSION=PR (modified by the PR author)"
echo "would remove: $1"
