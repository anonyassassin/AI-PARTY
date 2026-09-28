<div align="center">

# AI Party

**Run large language models by pooling RAM across the machines you already own.**

A distributed [llama.cpp](https://github.com/ggerganov/llama.cpp) cluster with a web chat UI, an operations dashboard, and a desktop client for worker nodes.

[Quickstart](#quickstart) · [How it works](#how-it-works) · [Configuration](#configuration) · [Troubleshooting](#troubleshooting)

</div>

---

## What it does

You have a laptop, a phone, an old desktop. Each one has a few gigabytes of RAM sitting idle. AI Party turns them into one machine that can run models none of them could run alone.

- **One host machine** runs the API and holds the model files. It also runs `llama-server`, the process that serves inference.
- **Any number of worker machines** join the cluster, run a lightweight RPC server, and lend their RAM to hold a slice of the model.
- **One browser tab** gives you a chat interface, a live model catalog, and a dashboard showing every machine in the cluster and its share of the split.

No cloud. No GPU required. No accounts, no telemetry, no data leaving your network.

---

## Why it's interesting

llama.cpp has had an RPC backend for a while — it can split a model across machines over the network. What it doesn't have is the glue: a way to discover workers, decide who gets how many layers, launch the server with the right flags, show the user what's happening during a multi-minute load, and recover when a node drops off the network.

AI Party is that glue. It adds:

- **Automatic tensor split** — weighted by each node's *usable* RAM (live available RAM minus a safety reserve), not raw total RAM. A machine with less headroom gets fewer layers, so it doesn't OOM.
- **Live load progress** — during a load, the frontend shows aggregate progress, per-worker byte counts, current network throughput, an ETA, and a per-node breakdown so you can see which machine is the bottleneck.
- **Verified models** — once a model loads successfully, it's marked verified and never shown as "locked" again, even if the RAM estimate says it won't fit. The estimate is conservative; the proof is definitive.
- **Self-healing workers** — if the host restarts, its in-memory node registry is wiped. Workers detect this and re-register automatically. No manual restart needed.
- **Same-origin frontend** — the Next.js app is a static export served by FastAPI on the same port as the API. No CORS, no Node runtime for users, no hardcoded IPs.

---

## Quickstart

### Requirements

- **Python 3.10+** on every machine
- **llama.cpp binaries** — installed automatically by the setup script (downloaded from the official GitHub releases)
- A local network where all machines can reach the host, and the host can reach each worker

You do **not** need Node.js, a C++ toolchain, or any prior llama.cpp install.

### 1. Clone the repo

```bash
git clone https://github.com/anonyassassin/ai-party.git
cd ai-party
```

### 2. Run the setup script

The setup script detects your OS, CPU architecture, and (on Linux/Windows) whether you have an NVIDIA GPU. It downloads the matching prebuilt llama.cpp binaries, creates a Python virtual environment, installs dependencies, and prepares the `models/` folder.

**macOS / Linux:**

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

**Windows (PowerShell):**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1
```

On a worker-only machine (no API host), pass `--worker` on macOS/Linux or `-Worker` on Windows to skip the FastAPI stack:

```bash
./scripts/setup.sh --worker
```

### 3. Add a model on the host

Drop one or more `.gguf` files into `./models/`. Any GGUF that `llama-server` supports will work. Recommended starters:

| Model | Size | Notes |
|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M | ~350 MB | Smallest. Boots in under 3 seconds. |
| Qwen2.5-1.5B-Instruct Q4_K_M | ~1 GB | Good balance of speed and capability. |
| Llama-3.2-3B-Instruct Q4_K_M | ~2 GB | Capable on a two- or three-node cluster. |

Sources: [Hugging Face GGUF models](https://huggingface.co/models?search=gguf)

### 4. Start the host

```bash
./scripts/run-host.sh          # macOS / Linux
.\scripts\run-host.ps1         # Windows
```

You'll see the host boot with the **smallest** model in `./models`, then start the HTTP API on `http://0.0.0.0:8000`.

### 5. Open the chat UI

Open `http://localhost:8000` in a browser. You should see the landing page. Click **Try it out now**, or navigate to `/chat` directly.

That's it — one URL, one port, one process serving both the frontend and the API.

### 6. Add a worker (optional)

On any other machine on the network, after running `setup.sh`:

```bash
./scripts/run-worker.sh --api-url http://<host-ip>:8000
```

`--node-ip` is auto-detected from your network interfaces. If the detection picks the wrong one (common with Docker, VPNs, or Tailscale), pass it explicitly:

```bash
./scripts/run-worker.sh --api-url http://192.168.1.6:8000 --node-ip 192.168.1.14
```

Watch the host log — you should see:

```
NODE JOINED: <uuid>
TOTAL USABLE CLUSTER CAPACITY: X.XX GB
```

The host recalculates the tensor split, restarts `llama-server` with `--rpc` pointing at the new worker, and the model reloads with its layers distributed. The dashboard at `/dashboard` shows every node and its share.

---

## How it works

### The three components

| File | Role | Runs on |
|---|---|---|
| `api.py` | HTTP API server (FastAPI). Serves the frontend, exposes `/api/*` and `/nodes`, spawns `llama-server`, coordinates the cluster. | Host only |
| `server.py` | Cluster intelligence. Model discovery, RAM accounting, tensor split calculation, load progress tracking, process launching. Imported by `api.py`. | Host only |
| `client.py` | Worker node. Spawns `ggml-rpc-server`, registers with the host, heartbeats, re-registers if the host forgets it. | Every worker |
| `app.py` | Optional desktop GUI wrapper around `client.py` for non-technical users. | Any worker |

### Data flow

```
              ┌──────────────────────────────────────────┐
              │             API HOST                     │
              │                                          │
              │  ┌────────────┐    ┌────────────────┐    │
   browser ───┼─▶│  FastAPI   │───▶│  llama-server  │    │
              │  │  api.py    │    │  (spawned)     │    │
              │  └────────────┘    └────────┬───────┘    │
              │                             │            │
              │  ┌────────────┐             │ tensor     │
              │  │  models/   │             │ slices     │
              │  │  *.gguf    │─────────────┘            │
              │  └────────────┘                          │
              └──────────────┬───────────────────────────┘
                             │  --rpc <ip:port>
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ WORKER 1 │   │ WORKER 2 │   │ WORKER 3 │
        │          │   │          │   │          │
        │ ggml-    │   │ ggml-    │   │ ggml-    │
        │ rpc-     │   │ rpc-     │   │ rpc-     │
        │ server   │   │ server   │   │ server   │
        │          │   │          │   │          │
        │ holds    │   │ holds    │   │ holds    │
        │ layers   │   │ layers   │   │ layers   │
        │ 0..N/3   │   │ N/3..2N/3│   │ 2N/3..N  │
        └──────────┘   └──────────┘   └──────────┘
```

The host reads the `.gguf` file from local disk. Each worker receives its slice of the model's tensors over the network at load time — it never sees the model file itself. During inference, activations flow between the host and workers over the same RPC channels.

### The tensor split

When a worker joins, `calculate_tensor_split` in `server.py` builds the `--tensor-split` argument from each node's **usable** RAM:

```
usable_ram_gb_i = max(available_ram_gb_i - safety_reserve, 0)
weight_i        = max(usable_ram_gb_i, 0.1)
split_i         = round(weight_i / sum(weights) * 100)
```

Weighting by usable (not total) RAM is what prevents OOMs. A machine with 8 GB total but 7 GB already in use has 1 GB usable, and gets a small share even though its "spec sheet" number looks bigger than a phone with 2 GB free.

Nodes below `MIN_USABLE_GB_TO_PARTICIPATE` (default 0.15 GB) are dropped from the split entirely rather than given a token sliver of layers.

### Load progress

The frontend polls `/api/load-status` once per second during a load. The backend tracks:

- **Aggregate progress** — total bytes uploaded divided by total bytes planned
- **Per-worker progress** — how much each worker has received out of its share
- **Network rate** — exponential moving average of the host's upload throughput
- **ETA** — derived from three sources in priority order: llama.cpp's own progress dots (authoritative), per-worker RPC buffer sizes parsed from the log, and predicted totals from the tensor split (fallback)

When a worker finishes receiving its slice, its row in the loading overlay flips to a green "✓ received" state. When all workers have received, the card shows a green banner — "Data received successfully, waiting for other nodes" — while the host finalizes. The overlay disappears when llama-server reports healthy.

### Model selection and "verified"

`/api/models` returns the full catalog with three flags per model:

- **`active`** — currently loaded and serving
- **`verified`** — has loaded successfully at least once, ever
- **`locked`** — not verified, and the RAM estimate says it won't fit

Verified models are **never** shown as locked, even if the current estimate says they won't fit. The estimate (`file_size × 1.15 + KV allowance`) is approximate and errs conservative; once a model has proven it fits by actually running, the estimate no longer overrides that.

Verified models are persisted to `verified_models.json` so the state survives host restarts.

### Cold start and persistence

On first launch, the cluster loads the **smallest** model on disk regardless of how much RAM is available. This guarantees the cluster comes up. A file called `last_model.json` records the last successfully-running model, so on subsequent starts the cluster resumes that model instead of falling back to the smallest.

The point is: startup is fast and predictable, and once you've picked a model, that choice survives restarts.

### Self-healing workers

`client.py` registers with the host via `POST /connect` and then heartbeats via `GET /alive` every few seconds. If the host restarts, its `NODE_REGISTRY` is wiped (it's in-memory only). The next heartbeat returns `Node not registered`, and the client immediately re-registers. From the user's perspective, the node disappears for a few seconds and then comes back on its own.

---

## Configuration

### Host (`api.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `0.0.0.0` | Interface the HTTP API binds to |
| `--port` | `8000` | Port the HTTP API binds to |
| `--llama-dir` | `./bin` | Folder containing `llama-server` and its shared libraries |
| `--model-dir` | `./models` | Folder scanned for `.gguf` files |

Environment overrides: `LLAMA_SERVER_PORT`, `LLAMA_SERVER_HOST`, `CONTEXT_SIZE`, `MODEL_DIR`, `LLAMA_DIR`, `HEARTBEAT_TIMEOUT`.

### Worker (`client.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--api-url` | `http://127.0.0.1:8000` | Base URL of the API host |
| `--node-ip` | `127.0.0.1` | This machine's IP **as seen by the API host** |
| `--port` | `50052` | Port the RPC server binds to |
| `--host` | `0.0.0.0` | Interface the RPC server binds to |
| `--interval` | `3` | Heartbeat interval in seconds |
| `--llama-dir` | `./bin` | Folder containing `ggml-rpc-server` |

Environment overrides: `API_URL`, `NODE_IP`, `RPC_PORT`, `RPC_HOST`, `LLAMA_DIR`.

### The `--node-ip` rule

`--node-ip` is the address the **API host** uses to reach this worker. Not the address the worker uses to reach the host. If you're on Tailscale, use the worker's Tailscale IP. If you're on a LAN, use the worker's LAN IP.

If a node appears in `/nodes` but never gets included in the tensor split, `--node-ip` is almost always the cause.

### Advanced env vars

| Variable | Default | Purpose |
|---|---|---|
| `CONTEXT_SIZE` | `8192` | Context window for every model launch |
| `MODEL_OVERHEAD_MULTIPLIER` | `1.15` | RAM overhead multiplier for RAM estimation |
| `MODEL_KV_OVERHEAD_GB` | `0.2` | KV cache allowance for RAM estimation |
| `MIN_USABLE_GB_TO_PARTICIPATE` | `0.15` | Nodes below this are excluded from the split |
| `HOST_OVERHEAD_RESERVE_GB` | `0.5` | Extra reserve on the host (runs KV cache) |
| `RAM_SAFETY_RESERVE_GB` | `0.3` | Reserve subtracted from every node's available RAM |
| `LOAD_STALL_TIMEOUT_S` | `300` | Kill a load with no progress after this many seconds |
| `HEARTBEAT_TIMEOUT` | `10` | Evict a node after this many seconds without a heartbeat |

---

## Prebuilt binaries

The setup script downloads prebuilt llama.cpp binaries from the official [`ggml-org/llama.cpp` releases](https://github.com/ggml-org/llama.cpp/releases). It picks the right archive automatically:

| Platform | Asset |
|---|---|
| macOS arm64 (Apple Silicon) | `llama-<tag>-bin-macos-arm64.tar.gz` |
| macOS x64 (Intel) | `llama-<tag>-bin-macos-x64.tar.gz` |
| Linux x64 (CPU) | `llama-<tag>-bin-ubuntu-x64.tar.gz` |
| Linux x64 (NVIDIA) | `llama-<tag>-bin-ubuntu-cuda-12.4-x64.tar.gz` (or `cuda-12.8`) |
| Linux arm64 | `llama-<tag>-bin-ubuntu-arm64.tar.gz` |
| Windows x64 (CPU) | `llama-<tag>-bin-win-cpu-x64.zip` |
| Windows x64 (NVIDIA) | `llama-<tag>-bin-win-cuda-12.4-x64.zip` (or `cuda-12.8`) |

The script checks `nvidia-smi` on Linux and Windows to decide whether to grab a CUDA build. macOS always uses the Metal-accelerated CPU build (no separate CUDA variant exists).

### Manual install

If the automatic download fails, grab the binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases) manually and extract them into `./bin/`:

```
bin/
├── llama-server          (llama-server.exe on Windows)
├── ggml-rpc-server       (ggml-rpc-server.exe on Windows)
└── *.dylib / *.so / *.dll
```

Then rerun `setup.sh`. The script will skip the download if `bin/llama-server` already exists.

---

## Troubleshooting

### The frontend shows "can't reach cluster"

Open the browser devtools → Network tab and look at which URL the requests are hitting. They should be relative (`/api/models`, `/nodes`) resolved against the page's own origin. If you see an absolute IP like `192.168.1.6:8000`, the frontend build is stale.

Fix in the Next.js source: set `const API_BASE = '';` at the top of both the chat page and the dashboard page, rebuild with `npm run build`, and copy `out/` to the Python repo's `frontend/` folder.

### A worker joins but doesn't appear in the split

Almost always `--node-ip`. Check `/nodes` on the host:

```bash
curl http://localhost:8000/nodes | python3 -m json.tool
```

Look at the worker's `ip_address`. If it says `127.0.0.1` but the worker isn't on the same machine, that's the bug. Kill the worker and restart with an explicit `--node-ip`:

```bash
./scripts/run-worker.sh --api-url http://192.168.1.6:8000 --node-ip 192.168.1.14
```

### Model load fails with exit code -6

`-6` is `SIGABRT`. On multi-node loads, this is almost always an OOM on one of the workers, or a mismatch between the host and worker llama.cpp builds. Two things to check:

1. **Same build on every machine.** Run `bin/llama-server --version` on the host and `bin/ggml-rpc-server --help` on each worker. The commit hashes should match.
2. **Worker RAM floor.** Raise `MIN_USABLE_GB_TO_PARTICIPATE` so workers with very little free RAM are excluded from the split:

   ```bash
   export MIN_USABLE_GB_TO_PARTICIPATE=2.5
   ./scripts/run-host.sh
   ```

### llama-server crashes immediately from `api.py` but works from the shell

On macOS, the two most common causes are a missing `TMPDIR`/`HOME` in the inherited environment and a quarantine xattr on the downloaded binary. Both are handled automatically by the current `server.py` (see `_build_subprocess_env` and `_strip_quarantine`). If you're on an older version, upgrade.

### `myvenv` shows up in `git status`

Add it to `.gitignore`:

```
myvenv/
.venv/
venv/
```

If it's already tracked:

```bash
git rm -r --cached myvenv
git commit -m "Stop tracking virtualenv"
```

### Frontend build fails with `Cannot find module '../server/require-hook'`

You're on an odd-numbered Node version (25, 23, 21), which Next.js doesn't officially support. Install Node 20 or 22 LTS:

```bash
nvm install --lts
nvm use --lts
rm -rf node_modules package-lock.json
npm install
npm run build
```

---

## Development

### Rebuilding the frontend

The Next.js source lives in a separate repo (or a `frontend-src/` folder if you've merged them). After editing:

```bash
cd /path/to/nextjs-repo
npm run build
rm -rf /path/to/ai-party/frontend
cp -r out /path/to/ai-party/frontend
```

No need to restart `api.py` — FastAPI's `StaticFiles` reads from disk on every request.

### Running with auto-reload

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Auto-reload only watches Python files. Frontend changes require a rebuild and a browser refresh.

---

## Architecture notes

### Why same-origin?

Serving the frontend and API from the same FastAPI process (same host, same port) eliminates:

- **CORS setup** — no `Access-Control-Allow-Origin` headers, no preflight
- **Hardcoded URLs** — relative fetches work on any device, any network
- **Node.js on the client** — the frontend is a static export, no runtime

The only trade-off is that the frontend can't be hosted separately from the API. For a self-hosted cluster tool, that's not a constraint.

### Why in-memory node registry?

`NODE_REGISTRY` is a plain dict, wiped on host restart. This is deliberate: a persistent registry would keep stale nodes around after they've shut down or dropped off the network, and the split would try to include machines that are no longer reachable. The in-memory registry stays honest — a node is in it only if it's actively heartbeating. Self-healing clients handle the rest.

### Why verify on load instead of on RAM estimate?

The RAM estimate is `file_size × 1.15 + 0.2 GB`. It's a heuristic, and it errs conservative — it prefers to say "won't fit" when a model would have worked. Verification is the escape hatch: once a model has actually loaded and served a request, the estimate is irrelevant. The `verified_models.json` file is the record of that proof.

---

## License

MIT

---

## Credits

Built on top of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov and contributors. The RPC backend, the tensor split mechanism, and `llama-server` are all upstream. AI Party adds the orchestration layer, the cluster manager, the frontend, and the worker client.

<div align="center">

**Your laptop, your phone, your old desktop — together.**

</div>
