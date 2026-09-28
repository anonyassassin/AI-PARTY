#!/usr/bin/env bash
#
# AI Party — setup script for macOS and Linux.
#
# Auto-detects OS, CPU architecture, and NVIDIA GPU/CUDA support, then
# downloads the correct prebuilt llama.cpp binaries from GitHub.
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

# ---------- argument parsing ----------
MODE="host"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --worker)
    MODE="worker"
    shift
    ;;
  --host)
    MODE="host"
    shift
    ;;
  -h | --help)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *) die "Unknown argument: $1" ;;
  esac
done

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

# ---------- Fetch latest release tag (bNNNNN) ----------
# Note: /releases/latest returns the most recent *stable* release (vX.Y.Z),
# which may not include the newest binaries. /releases?per_page=1 returns
# the newest entry regardless of pre-release status, which is what we want
# since llama.cpp ships rolling bNNNNN builds.
say "Resolving latest llama.cpp release…"
RELEASE_JSON=$(curl -sL "${GITHUB_API}?per_page=1")
TAG=$(echo "$RELEASE_JSON" | grep -o '"tag_name": *"[^"]*"' | head -n1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')

if [[ -z "$TAG" ]]; then
  # Fallback in case the API shape changes
  TAG=$(curl -sL "${GITHUB_API}/latest" | grep -o '"tag_name": *"[^"]*"' | head -n1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
fi

[[ -n "$TAG" ]] || die "Could not resolve latest release tag from GitHub."
ok "Latest release: $TAG"

# ---------- Build asset name ----------
# Examples:
#   macOS arm64:  llama-b10809-bin-macos-arm64.tar.gz
#   Ubuntu x64:   llama-b10809-bin-ubuntu-x64.tar.gz
#   Windows x64:  llama-b10809-bin-win-cpu-x64.zip
if [[ "$PLATFORM" == "macos" ]]; then
  ASSET="llama-${TAG}-bin-macos-${ARCH_TAG}.tar.gz"
elif [[ "$GPU_TAG" == cuda-* && "$ARCH_TAG" == "x64" ]]; then
  # Official CUDA Linux builds are x64 only.
  ASSET="llama-${TAG}-bin-ubuntu-${GPU_TAG}-${ARCH_TAG}.tar.gz"
else
  ASSET="llama-${TAG}-bin-ubuntu-${ARCH_TAG}.tar.gz"
fi

say "Selected asset: $ASSET"

# ---------- Download and extract ----------
URL="${GITHUB_DL}/${TAG}/${ASSET}"
TMP_FILE="/tmp/${ASSET}"

say "Downloading ${ASSET} …"
if ! curl -fL --progress-bar -o "$TMP_FILE" "$URL"; then
  if [[ "$GPU_TAG" == cuda-* ]]; then
    FALLBACK="llama-${TAG}-bin-ubuntu-cuda-${ARCH_TAG}.tar.gz"
    warn "CUDA asset not found; trying fallback: ${FALLBACK}"
    curl -fL --progress-bar -o "$TMP_FILE" "${GITHUB_DL}/${TAG}/${FALLBACK}" ||
      die "Could not download llama.cpp binaries for ${PLATFORM}/${ARCH_TAG} (${GPU_TAG:-CPU})."
  else
    die "Could not download from ${URL}"
  fi
fi

say "Extracting into ${BIN_DIR} …"
mkdir -p "$BIN_DIR"
tar -xzf "$TMP_FILE" -C "$BIN_DIR" --strip-components=1
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
[[ "$MODE" == "worker" ]] && REQ_FILE="${REPO_ROOT}/requirements-worker.txt"
[[ -f "$REQ_FILE" ]] || die "Missing $REQ_FILE"

if [[ ! -d "$VENV_DIR" ]]; then
  say "Creating virtual environment…"
  "$PY" -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip >/dev/null
pip install -r "$REQ_FILE"
ok "Dependencies installed"

# ---------- Models directory ----------
if [[ "$MODE" == "host" && ! -d "$MODELS_DIR" ]]; then
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

# ---------- Summary ----------
cat <<EOF

${c_green}${c_bold}Setup complete.${c_reset}

  Binaries : ${BIN_DIR}
  Models   : ${MODELS_DIR}
  Venv     : ${VENV_DIR}
  Build    : ${PLATFORM}/${ARCH_TAG}${GPU_TAG:+ (${GPU_TAG})}

$(
  [[ "$MODE" == "host" ]] && cat <<'INNER'
Next steps (host):
  1. Add a .gguf file to ./models/
  2. Start: ./scripts/run-host.sh
INNER
)

$(
  [[ "$MODE" == "worker" ]] && cat <<'INNER'
Next steps (worker):
  ./scripts/run-worker.sh --api-url http://<host>:8000
INNER
)

EOF
