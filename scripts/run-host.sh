#!/usr/bin/env bash
#
# AI Party — start a worker node (client.py).
#
# Usage:
#   ./scripts/run-worker.sh --api-url http://192.168.1.6:8000
#   ./scripts/run-worker.sh --api-url http://192.168.1.6:8000 --node-ip 192.168.1.14
#   ./scripts/run-worker.sh --api-url http://192.168.1.6:8000 --port 50053
#   ./scripts/run-worker.sh --api-url http://192.168.1.6:8000 --llama-dir /other/bin
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- pick Python: prefer the repo's venv if present ----------
PY=""
for candidate in \
  "${REPO_ROOT}/myvenv/bin/python" \
  "${REPO_ROOT}/.venv/bin/python" \
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

# ---------- parse the flags we handle specially ----------
API_URL=""
NODE_IP=""
HAS_LLAMA_DIR=0
REST=()

while [[ $# -gt 0 ]]; do
  case "$1" in
  --api-url)
    [[ $# -ge 2 ]] || {
      echo "Missing value for --api-url" >&2
      exit 1
    }
    API_URL="$2"
    shift 2
    ;;
  --node-ip)
    [[ $# -ge 2 ]] || {
      echo "Missing value for --node-ip" >&2
      exit 1
    }
    NODE_IP="$2"
    shift 2
    ;;
  --llama-dir)
    [[ $# -ge 2 ]] || {
      echo "Missing value for --llama-dir" >&2
      exit 1
    }
    HAS_LLAMA_DIR=1
    REST+=("$1" "$2")
    shift 2
    ;;
  --help | -h)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    REST+=("$1")
    shift
    ;;
  esac
done

if [[ -z "$API_URL" ]]; then
  cat >&2 <<'EOF'
Missing --api-url.

Usage:
  ./scripts/run-worker.sh --api-url http://<host-ip>:8000

Example:
  ./scripts/run-worker.sh --api-url http://192.168.1.6:8000 --node-ip 192.168.1.14
EOF
  exit 1
fi

# ---------- auto-detect node IP if not given ----------
if [[ -z "$NODE_IP" ]]; then
  case "$(uname -s)" in
  Darwin)
    # Try the common interfaces. en0 is WiFi on Apple Silicon,
    # en1 is used on some Intel Macs.
    NODE_IP="$(ipconfig getifaddr en0 2>/dev/null ||
      ipconfig getifaddr en1 2>/dev/null ||
      true)"
    ;;
  Linux)
    # Take the first address from hostname -I (usually the primary NIC).
    NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
    ;;
  esac

  if [[ -z "$NODE_IP" ]]; then
    cat >&2 <<'EOF'
Could not auto-detect a local IP.

Pass --node-ip explicitly. This must be the address the API host uses
to reach THIS machine, not the address this machine uses to reach the
host.

Examples:
  --node-ip 192.168.1.14      (LAN)
  --node-ip 100.99.40.83      (Tailscale)
  --node-ip 10.0.0.5          (VPN)
EOF
    exit 1
  fi
  echo "Auto-detected node IP: $NODE_IP"
fi

# ---------- build the argument list ----------
ARGS=(--api-url "$API_URL" --node-ip "$NODE_IP")

# Only add --llama-dir if the user didn't already supply one.
if [[ "$HAS_LLAMA_DIR" -eq 0 ]]; then
  ARGS+=(--llama-dir "${REPO_ROOT}/bin")
fi

exec "$PY" "${REPO_ROOT}/client.py" "${ARGS[@]}" "${REST[@]}"
