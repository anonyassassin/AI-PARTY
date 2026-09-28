#!/usr/bin/env bash
#
# AI Party — start the API host (api.py).
#
# Usage:
#   ./scripts/run-host.sh                              # defaults
#   ./scripts/run-host.sh --port 9000                  # override port
#   ./scripts/run-host.sh --llama-dir /other/bin       # override llama dir
#   ./scripts/run-host.sh --model-dir /data/gguf       # override models dir
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- pick Python: prefer the repo's venv if present ----
PY=""
for candidate in \
  "${REPO_ROOT}/.venv/bin/python" \
  "${REPO_ROOT}/myvenv/bin/python" \
  "${REPO_ROOT}/venv/bin/python"; do
  if [[ -x "$candidate" ]]; then
    PY="$candidate"
    break
  fi
done

if [[ -z "$PY" ]]; then
  PY="$(command -v python3 || command -v python)" || {
    echo "Python not found. Run ./scripts/setup.sh first." >&2
    exit 1
  }
fi

# ---- defaults (overridable by flags you pass) ----
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_LLAMA_DIR="${REPO_ROOT}/bin"
DEFAULT_MODEL_DIR="${REPO_ROOT}/models"

# ---- scan caller args for the flags we provide defaults for ----
HAS_LLAMA_DIR=0
HAS_MODEL_DIR=0
HAS_HOST=0
HAS_PORT=0

for arg in "$@"; do
  case "$arg" in
  --llama-dir) HAS_LLAMA_DIR=1 ;;
  --model-dir) HAS_MODEL_DIR=1 ;;
  --host) HAS_HOST=1 ;;
  --port) HAS_PORT=1 ;;
  esac
done

# ---- build the argument list ----
ARGS=()

if [[ "$HAS_HOST" -eq 0 ]]; then
  ARGS+=(--host "$DEFAULT_HOST")
fi
if [[ "$HAS_PORT" -eq 0 ]]; then
  ARGS+=(--port "$DEFAULT_PORT")
fi
if [[ "$HAS_LLAMA_DIR" -eq 0 ]]; then
  ARGS+=(--llama-dir "$DEFAULT_LLAMA_DIR")
fi
if [[ "$HAS_MODEL_DIR" -eq 0 ]]; then
  ARGS+=(--model-dir "$DEFAULT_MODEL_DIR")
fi

exec "$PY" "${REPO_ROOT}/api.py" "${ARGS[@]}" "$@"
