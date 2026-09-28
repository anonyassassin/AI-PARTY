#!/usr/bin/env bash
#
# AI Party — setup script for macOS and Linux.
#
# Auto-detects OS, CPU architecture, and NVIDIA GPU/CUDA support, then
# downloads the correct prebuilt llama.cpp binaries from GitHub.
#
# Installs the full dependency set. A machine can act as an API host,
# a worker node, or both — the role is chosen at runtime.
#
set -euo pipefail

# ---------- config ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${REPO_ROOT}/bin"
MODELS_DIR="${REPO_ROOT}/models"
VENV_DIR="${REPO_ROOT}/.venv"

LLAMA_REPO="ggml-org/llama.cpp"
GITHUB_API="https://api.github.com/repos/${LLAMA_REPO}/releases"
GITHUB_DL="https://github.com/${LLAMA_REPO}/releases/download"

# ---------- pretty output ----------
c_reset=$'\033[0m'
c_bold=$'\033[1m'
c_amber=$'\033[38;5;214m'
c_green=$'\033[38;5;114m'
c_red=$'\033[38;5;203m'

say() { printf '%s\n' "${c_amber}${c_bold}▸${c_reset} $*"; }
ok() { printf '%s\n' "${c_green}✓${c_reset} $*"; }
warn() { printf '%s\n' "${c_amber}!${c_reset} $*"; }
die() {
  printf '%s\n' "${c_red}✗${c_reset} $*" >&2
  exit 1
}

# ---------- OS and architecture detection ----------
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
Darwin)
  PLATFORM="macos"
  [[ "$ARCH" == "arm64" ]] && ARCH_TAG="arm64" || ARCH_TAG="x64"
  GPU_TAG=""
  ;;
Linux)
  PLATFORM="ubuntu"
  [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]] && ARCH_TAG="arm64" || ARCH_TAG="x64"

  GPU_TAG=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    say "NVIDIA GPU detected, checking CUDA driver version…"
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
    if [[ -n "$DRIVER_VER" ]]; then
      MAJOR=${DRIVER_VER%%.*}
      if ((MAJOR >= 570)); then
        GPU_TAG="cuda-12.8"
        ok "CUDA 12.8 capable (driver ${DRIVER_VER})"
      elif ((MAJOR >= 550)); then
        GPU_TAG="cuda-12.4"
        ok "CUDA 12.4 capable (driver ${DRIVER_VER})"
      else
        warn "Driver ${DRIVER_VER} too old for prebuilt CUDA archives; using CPU."
      fi
    fi
  fi
  ;;
*)
  die "Unsupported OS: $OS. Use setup.ps1 on Windows."
  ;;
esac

say "Detected: ${PLATFORM} / ${ARCH_TAG} ${GPU_TAG:+(GPU: $GPU_TAG)}"

# ---------- Python ----------
say "Checking Python…"
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  die "Python not found. Install Python 3.10+ and re-run."
fi

PY_VER="$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_MAJOR="${PY_VER%%.*}"
PY_MINOR="${PY_VER##*.}"
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
  die "Python 3.10+ required, found $PY_VER"
fi
ok "Python $PY_VER at $(command -v $PY)"

# ---------- Resolve a usable release tag ----------
# llama.cpp publishes rolling bNNNNN pre-releases. Older ones can be
# unpublished or marked as drafts, in which case their assets 404.
# We walk the recent releases and pick the newest one that:
#   - has at least one downloadable asset
#   - has an asset matching our selected platform/variant
# This avoids the "404 Not Found" that happens when ?per_page=1 points
# at a tag whose assets aren't published yet.
say "Resolving llama.cpp release…"

RELEASES_JSON=$(curl -sL "${GITHUB_API}?per_page=20")
[[ -n "$RELEASES_JSON" ]] || die "Could not reach GitHub API."

# Build the asset name pattern list for this machine.
# Each pattern uses a glob that gets matched against the asset names
# in the release JSON.
if [[ "$PLATFORM" == "macos" ]]; then
  PATTERNS=(
    "llama-*-bin-macos-${ARCH_TAG}.tar.gz"
    "llama-*-bin-macos-${ARCH_TAG}.zip"
  )
elif [[ "$GPU_TAG" == cuda-* && "$ARCH_TAG" == "x64" ]]; then
  PATTERNS=(
    "llama-*-bin-ubuntu-${GPU_TAG}-${ARCH_TAG}.tar.gz"
    "llama-*-bin-ubuntu-${GPU_TAG}-${ARCH_TAG}.zip"
    "llama-*-bin-ubuntu-cuda-${ARCH_TAG}.tar.gz"
    "llama-*-bin-ubuntu-cuda-${ARCH_TAG}.zip"
  )
else
  PATTERNS=(
    "llama-*-bin-ubuntu-${ARCH_TAG}.tar.gz"
    "llama-*-bin-ubuntu-${ARCH_TAG}.zip"
  )
fi

# Walk releases in order, find the first one whose assets match.
TAG=""
ASSET=""
RELEASE_IDX=0
while [[ $RELEASE_IDX -lt 20 ]]; do
  # Extract this release's tag_name
  CAND_TAG=$(echo "$RELEASES_JSON" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    idx = $RELEASE_IDX
    if idx < len(data):
        print(data[idx].get('tag_name',''))
except Exception:
    pass
" 2>/dev/null || echo "")

  [[ -n "$CAND_TAG" ]] || break

  # Extract this release's asset names, one per line
  CAND_ASSETS=$(echo "$RELEASES_JSON" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    idx = $RELEASE_IDX
    if idx < len(data):
        for a in data[idx].get('assets', []):
            print(a.get('name',''))
except Exception:
    pass
" 2>/dev/null || echo "")

  # Skip releases with no assets at all (drafts, empty)
  if [[ -z "$CAND_ASSETS" ]]; then
    RELEASE_IDX=$((RELEASE_IDX + 1))
    continue
  fi

  # Try each pattern against the asset list
  for PATTERN in "${PATTERNS[@]}"; do
    MATCH=$(echo "$CAND_ASSETS" | grep -F -x "$(echo "$PATTERN" | sed 's/\*/.*/g' | sed 's/\?/./g' | sed 's/^/^/;s/$/$/')" 2>/dev/null | head -n1 || true)
    # Fallback: use shell globbing against the asset list
    if [[ -z "$MATCH" ]]; then
      while IFS= read -r ASSET_NAME; do
        # shellcheck disable=SC2053
        if [[ "$ASSET_NAME" == $PATTERN ]]; then
          MATCH="$ASSET_NAME"
          break
        fi
      done <<<"$CAND_ASSETS"
    fi

    if [[ -n "$MATCH" ]]; then
      TAG="$CAND_TAG"
      ASSET="$MATCH"
      break
    fi
  done

  [[ -n "$TAG" ]] && break
  RELEASE_IDX=$((RELEASE_IDX + 1))
done

if [[ -z "$TAG" || -z "$ASSET" ]]; then
  warn "No prebuilt asset found in the last 20 releases for ${PLATFORM}/${ARCH_TAG}."
  warn "Falling back to the newest release with any asset."
  TAG=$(echo "$RELEASES_JSON" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for r in data:
        if r.get('assets'):
            print(r['tag_name']); break
except Exception:
    pass
" 2>/dev/null || echo "")
  [[ -n "$TAG" ]] || die "Could not find any usable release."

  if [[ "$PLATFORM" == "macos" ]]; then
    ASSET="llama-${TAG}-bin-macos-${ARCH_TAG}.tar.gz"
  elif [[ "$GPU_TAG" == cuda-* && "$ARCH_TAG" == "x64" ]]; then
    ASSET="llama-${TAG}-bin-ubuntu-${GPU_TAG}-${ARCH_TAG}.tar.gz"
  else
    ASSET="llama-${TAG}-bin-ubuntu-${ARCH_TAG}.tar.gz"
  fi
fi

ok "Release: $TAG"
say "Selected asset: $ASSET"

# ---------- Download and extract ----------
URL="${GITHUB_DL}/${TAG}/${ASSET}"
TMP_FILE="/tmp/${ASSET}"

say "Downloading ${ASSET} …"
if ! curl -fL --progress-bar -o "$TMP_FILE" "$URL"; then
  die "Could not download $URL

Check the release page for the exact asset name:
  https://github.com/${LLAMA_REPO}/releases/tag/${TAG}
"
fi

say "Extracting into ${BIN_DIR} …"
mkdir -p "$BIN_DIR"
case "$TMP_FILE" in
*.zip)
  if command -v unzip >/dev/null 2>&1; then
    unzip -oq "$TMP_FILE" -d "$BIN_DIR"
  else
    die "unzip is required to extract $TMP_FILE. Install it (apt install unzip / brew install unzip)."
  fi
  # Flatten any versioned subdirectory
  SUBDIR=$(find "$BIN_DIR" -maxdepth 1 -type d -name "llama-*" | head -1)
  if [[ -n "$SUBDIR" ]]; then
    mv "$SUBDIR"/* "$BIN_DIR"/ 2>/dev/null || true
    rmdir "$SUBDIR" 2>/dev/null || true
  fi
  ;;
*.tar.gz)
  tar -xzf "$TMP_FILE" -C "$BIN_DIR" --strip-components=1
  ;;
*)
  die "Unknown archive format: $TMP_FILE"
  ;;
esac
rm -f "$TMP_FILE"

# ---------- macOS quarantine fix ----------
if [[ "$PLATFORM" == "macos" ]]; then
  xattr -dr com.apple.quarantine "$BIN_DIR" 2>/dev/null || true
  ok "Gatekeeper quarantine removed"
fi

# ---------- Verify binaries ----------
[[ -x "${BIN_DIR}/llama-server" ]] || die "llama-server not found in ${BIN_DIR}."
[[ -x "${BIN_DIR}/ggml-rpc-server" ]] || warn "ggml-rpc-server not found; RPC backend may be missing from this build."
ok "Binaries present"

# ---------- Python deps ----------
say "Installing Python dependencies…"
REQ_FILE="${REPO_ROOT}/requirements.txt"
[[ -f "$REQ_FILE" ]] || die "Missing $REQ_FILE"

if [[ ! -d "$VENV_DIR" ]]; then
  say "Creating virtual environment at ${VENV_DIR}"
  "$PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip >/dev/null
pip install -r "$REQ_FILE"
ok "Dependencies installed"

# ---------- Models directory ----------
if [[ ! -d "$MODELS_DIR" ]]; then
  mkdir -p "$MODELS_DIR"
  cat >"$MODELS_DIR/README.txt" <<'EOF'
Drop your .gguf files in this folder.
Recommended starters:
  - Qwen2.5-0.5B-Instruct Q4_K_M   (~350 MB)
  - Qwen2.5-1.5B-Instruct Q4_K_M   (~1 GB)
  - Llama-3.2-3B-Instruct Q4_K_M   (~2 GB)
EOF
  ok "Created ${MODELS_DIR}"
fi

# ---------- Frontend check ----------
if [[ -d "${REPO_ROOT}/frontend" && -f "${REPO_ROOT}/frontend/index.html" ]]; then
  ok "Frontend present at ${REPO_ROOT}/frontend"
else
  warn "No prebuilt frontend at ${REPO_ROOT}/frontend."
  warn "The API will run, but the chat UI won't be served."
fi

# ---------- Summary ----------
cat <<EOF

${c_green}${c_bold}Setup complete.${c_reset}

  Binaries : ${BIN_DIR}
  Models   : ${MODELS_DIR}
  Venv     : ${VENV_DIR}
  Build    : ${PLATFORM}/${ARCH_TAG}${GPU_TAG:+ (${GPU_TAG})}
  Release  : ${TAG}

You can now run this machine as a host, a worker, or both.

  Start as host (serves the API + chat UI):
    ./scripts/run-host.sh

  Start as worker (joins another machine's cluster):
    ./scripts/run-worker.sh --api-url http://<host-ip>:8000

  Launch the desktop GUI client:
    ./scripts/run-gui.sh

To activate the virtual environment in this shell:

    source .venv/bin/activate

Once activated, your prompt shows '(.venv)' and 'python' resolves
to the venv interpreter. Run 'deactivate' to leave it.

EOF
