#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "$APP_ROOT/.." && pwd)"

if [[ $# != 2 ]]; then
  echo 'Usage: package.sh OUTPUT_DIRECTORY DOWNLOAD_URL' >&2
  echo 'Create a macOS arm64 app archive and a local Homebrew tap.' >&2
  exit 1
fi
OUTPUT_DIRECTORY="$1"
DOWNLOAD_URL="$2"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Apple Silicon macOS is required.' >&2; exit 1; }
[[ "$DOWNLOAD_URL" =~ ^https?://[a-zA-Z0-9/:._%?=\&+-]+$ ]] || { echo 'Use an HTTP(S) archive URL without credentials or fragments.' >&2; exit 1; }
[[ ! -e "$OUTPUT_DIRECTORY" ]] || { echo 'The output directory must not exist.' >&2; exit 1; }
command -v uv >/dev/null
brew list --versions ffmpeg@7 >/dev/null
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP_ROOT/Resources/Info.plist")"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'Use a three-part numeric app version.' >&2; exit 1; }
BUILD_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/omnityper-package.XXXXXX")"
trap 'rm -rf "$BUILD_DIRECTORY"' EXIT

uv python install 3.12 --install-dir "$BUILD_DIRECTORY/python" --no-bin
PYTHON_EXECUTABLE="$(UV_PYTHON_INSTALL_DIR="$BUILD_DIRECTORY/python" uv python find --managed-python 3.12)"
RUNTIME_DIRECTORY="$(cd -- "$(dirname -- "$PYTHON_EXECUTABLE")/.." && pwd -P)"
# Note (Codex): This standalone prefix is private to the build, never the host Python.
export UV_BREAK_SYSTEM_PACKAGES=1
export PYTHONDONTWRITEBYTECODE=1
UV_SYSTEM_PYTHON=1 SGLANG_OMNI_VENV="$RUNTIME_DIRECTORY" bash "$REPO_ROOT/install.sh" --non-interactive
uv pip install --python "$PYTHON_EXECUTABLE" --system -r "$APP_ROOT/backend/requirements.txt"
uv pip install --python "$PYTHON_EXECUTABLE" --system --no-deps --reinstall "$REPO_ROOT"
uv pip check --python "$PYTHON_EXECUTABLE"
uv pip freeze --python "$PYTHON_EXECUTABLE" > "$BUILD_DIRECTORY/runtime-requirements.txt"
if grep -q '^-e ' "$BUILD_DIRECTORY/runtime-requirements.txt"; then
  echo 'The runtime must contain a non-editable SGLang-Omni installation.' >&2
  exit 1
fi

OMNITYPER_RUNTIME="$RUNTIME_DIRECTORY" OMNITYPER_DIST_DIR="$BUILD_DIRECTORY/dist" bash "$APP_ROOT/scripts/build.sh"
APP_BUNDLE="$BUILD_DIRECTORY/dist/OmniTyper.app"
cp "$BUILD_DIRECTORY/runtime-requirements.txt" "$APP_BUNDLE/Contents/Resources/"
git -C "$REPO_ROOT" rev-parse HEAD > "$APP_BUNDLE/Contents/Resources/source-revision.txt"

# Note (Codex): Sign nested native code before sealing the outer app bundle.
SIGNING_ARGUMENTS=(--force --sign "${CODE_SIGN_IDENTITY:--}" --options runtime)
if [[ "${CODE_SIGN_IDENTITY:--}" != - ]]; then
  SIGNING_ARGUMENTS+=(--timestamp)
fi
while IFS= read -r -d '' NATIVE_FILE; do
  if /usr/bin/file -b "$NATIVE_FILE" | /usr/bin/grep -q 'Mach-O'; then
    codesign "${SIGNING_ARGUMENTS[@]}" --entitlements "$APP_ROOT/Resources/RuntimeEntitlements.plist" "$NATIVE_FILE"
  fi
done < <(find "$APP_BUNDLE/Contents/Resources/runtime" -type f \( -name '*.so' -o -name '*.dylib' -o -perm -111 \) -print0)
codesign "${SIGNING_ARGUMENTS[@]}" --entitlements "$APP_ROOT/Resources/Entitlements.plist" "$APP_BUNDLE"
codesign --verify --deep --strict "$APP_BUNDLE"

mkdir -p "$OUTPUT_DIRECTORY/Casks"
OUTPUT_DIRECTORY="$(cd -- "$OUTPUT_DIRECTORY" && pwd)"
ARCHIVE="$OUTPUT_DIRECTORY/OmniTyper-$VERSION-arm64.zip"
if [[ -n "${OMNITYPER_NOTARY_PROFILE:-}" ]]; then
  [[ "${CODE_SIGN_IDENTITY:--}" != - ]] || { echo 'Notarization requires Developer ID signing.' >&2; exit 1; }
  ditto -c -k --keepParent "$APP_BUNDLE" "$BUILD_DIRECTORY/notarize.zip"
  xcrun notarytool submit "$BUILD_DIRECTORY/notarize.zip" --keychain-profile "$OMNITYPER_NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP_BUNDLE"
  spctl --assess --type execute --verbose "$APP_BUNDLE"
fi
ditto -c -k --keepParent "$APP_BUNDLE" "$ARCHIVE"
ARCHIVE_SHA256="$(shasum -a 256 "$ARCHIVE" | cut -d ' ' -f 1)"
cat > "$OUTPUT_DIRECTORY/Casks/omnityper.rb" <<EOF
cask "omnityper" do
  version "$VERSION"
  sha256 "$ARCHIVE_SHA256"

  url "$DOWNLOAD_URL"
  name "OmniTyper"
  desc "Local voice typing powered by SGLang-Omni"
  homepage "https://github.com/sgl-project/sglang-omni/tree/main/OmniTyper"

  depends_on arch: :arm64
  depends_on macos: ">= :sonoma"
  depends_on formula: "ffmpeg@7"

  app "OmniTyper.app"
end
EOF
echo "Archive: $ARCHIVE"
echo "Cask: $OUTPUT_DIRECTORY/Casks/omnityper.rb"
echo 'Unsigned builds are for local testing. Public releases require Developer ID signing and notarization.'
