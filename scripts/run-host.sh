#!/usr/bin/env bash
#
# AI Party — start the API host (api.py).
#
# Usage:
#   ./scripts/run-host.sh                        # defaults: 0.0.0.0:8000, ./bin, ./models
#   ./scripts/run-host.sh --port 9000            # override any api.py flag
#   ./scripts/run-host.sh --model-dir /data/gguf # custom models folder
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use the venv's Python if present, otherwise fall back to system python3.
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)" || {
    echo "Python not found. Run ./scripts/setup.sh first." >&2
    exit 1
  }
fi

# Default paths, overridable by the caller passing the same flags again.
LLAMA_DIR="${REPO_ROOT}/bin"
MODEL_DIR="${REPO_ROOT}/models"

exec "$PY" "${REPO_ROOT}/api.py" \
  --host 0.0.0.0 \
  --port 8000 \
  --llama-dir "$LLAMA_DIR" \
  --model-dir "$MODEL_DIR" \
  "$@"
