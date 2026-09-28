<div align="center">

# AI Party

**Turn your idle devices into a personal supercomputer.**

A distributed [llama.cpp](https://github.com/ggerganov/llama.cpp) cluster with a web chat UI, an operations dashboard, and a desktop GUI for worker nodes.

</div>

---

## What it does

You have a laptop, a phone, an old desktop. Each one has a few gigabytes of RAM sitting idle. AI Party pools them into one virtual machine — a personal supercomputer that can run models none of your devices could run alone.

- **One host machine** runs the API and holds the model files. It also runs `llama-server`, the process that serves inference.
- **Any number of worker machines** join the cluster, run a lightweight RPC server, and lend their RAM to hold a slice of the model.
- **Two frontends**: a web chat + dashboard served by the host, and an optional desktop GUI that makes joining the cluster a one-click operation on any machine.

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
- **Desktop GUI for workers** — a Tkinter app that lets anyone join the cluster by filling in a form and clicking **Connect**. No terminal, no flags to memorize.

---

## Quickstart

### Requirements

- **Python 3.10+** on every machine
- **llama.cpp binaries** — installed automatically by the setup script
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

On a worker-only machine (no API host), pass `--worker` on macOS/Linux or `-Worker` on Windows:

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

The host boots with the **smallest** model in `./models`, then starts the HTTP API on `http://0.0.0.0:8000`.

### 5. Open the web UI

Open `http://localhost:8000` in a browser. Click **Try it out now** to open the chat, or **View dashboard** to see the cluster.

### 6. Add a worker

You can add a worker in two ways. Pick whichever you prefer.

#### Option A — Desktop GUI (recommended for non-technical users)

On any other machine, after running `setup.sh`, launch the GUI:

```bash
python app.py
```

The window has fields for every flag `client.py` accepts:

- **API host URL** — where the host is running
- **Node IP** — this machine's IP as seen by the host, with a **Detect** button that fills it in automatically
- **RPC port** and **bind host** — defaults are fine
- **Heartbeat interval** — how often to ping the host
- **llama.cpp folder** — with a native **Browse** button

Click **Connect**. The GUI spawns `client.py` as a subprocess and streams its output into a live log pane on the right side of the window. A green status indicator appears once the worker is registered and heartbeating.

Config is saved to `~/.aiparty_client_gui.json` on every change, so the fields are pre-filled next time you open it. There's also a **Detect** button for the local IP.

#### Option B — Command line

```bash
./scripts/run-worker.sh --api-url http://<host-ip>:8000
```

`--node-ip` is auto-detected. If the detection picks the wrong interface (common with Docker, VPNs, or Tailscale), pass it explicitly:

```bash
./scripts/run-worker.sh --api-url http://192.168.1.6:8000 --node-ip 192.168.1.14
```

Both options launch the same `client.py` process under the hood — the GUI is a form-based wrapper. Whichever you use, the host recalculates the tensor split, restarts `llama-server` with `--rpc` pointing at the new worker, and the model reloads with its layers distributed. The dashboard at `/dashboard` shows every node and its share.

---

## How it works

### The components

| File | Role | Runs on |
|---|---|---|
| `api.py` | HTTP API server (FastAPI). Serves the frontend, exposes `/api/*` and `/nodes`, spawns `llama-server`, coordinates the cluster. | Host only |
| `server.py` | Cluster intelligence. Model discovery, RAM accounting, tensor split calculation, load progress tracking, process launching. Imported by `api.py`. | Host only |
| `client.py` | Worker node. Spawns `ggml-rpc-server`, registers with the host, heartbeats, re-registers if the host forgets it. | Every worker |
| `app.py` | Desktop GUI wrapper around `client.py`. Tkinter + CustomTkinter. Optional but recommended for non-technical users. | Any worker |

The host reads the `.gguf` file from local disk. Each worker receives its slice of the model's tensors over the network at load time — it never sees the model file itself. During inference, activations flow between the host and workers over the same RPC channels.

### The tensor split

`calculate_tensor_split` builds the `--tensor-split` argument from each node's **usable** RAM:

```
usable_ram_gb_i = max(available_ram_gb_i - safety_reserve, 0)
weight_i        = max(usable_ram_gb_i, 0.1)
split_i         = round(weight_i / sum(weights) * 100)
```

Weighting by usable (not total) RAM is what prevents OOMs. A machine with 8 GB total but 7 GB in use has 1 GB usable, and gets a small share even though its "spec sheet" number looks bigger than a phone with 2 GB free.

Nodes below `MIN_USABLE_GB_TO_PARTICIPATE` (default 0.15 GB) are dropped from the split entirely.

### The desktop GUI

`app.py` is a single-file Tkinter application. When you click **Connect**:

1. It reads the form fields and builds the exact `python client.py ...` argument list.
2. It spawns `client.py` as a subprocess with `stdout` piped, using `-u` so output arrives unbuffered.
3. A background thread reads the pipe line by line and pushes lines into a queue.
4. The main thread drains the queue every 80 ms and appends to a scrolling text widget.

The GUI doesn't reimplement any cluster logic. It's purely a form + log viewer around the same `client.py` that the CLI wrappers call. If you update `client.py`, the GUI picks up the change automatically.

On Windows the subprocess is spawned with `CREATE_NO_WINDOW` so no console flashes. On macOS and Linux, `os.setsid` puts the child in its own process group so terminating the GUI also terminates `ggml-rpc-server`.

---

## Configuration

### Host (`api.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `0.0.0.0` | Interface the HTTP API binds to |
| `--port` | `8000` | Port the HTTP API binds to |
| `--llama-dir` | `./bin` | Folder containing `llama-server` |
| `--model-dir` | `./models` | Folder scanned for `.gguf` files |

### Worker (`client.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--api-url` | `http://127.0.0.1:8000` | Base URL of the API host |
| `--node-ip` | `127.0.0.1` | This machine's IP **as seen by the API host** |
| `--port` | `50052` | Port the RPC server binds to |
| `--host` | `0.0.0.0` | Interface the RPC server binds to |
| `--interval` | `3` | Heartbeat interval in seconds |
| `--llama-dir` | `./bin` | Folder containing `ggml-rpc-server` |

Every one of these flags is also exposed as a form field in the desktop GUI.

### Advanced env vars

| Variable | Default | Purpose |
|---|---|---|
| `CONTEXT_SIZE` | `8192` | Context window for every model launch |
| `MIN_USABLE_GB_TO_PARTICIPATE` | `0.15` | Nodes below this are excluded from the split |
| `RAM_SAFETY_RESERVE_GB` | `0.3` | Reserve subtracted from every node's available RAM |
| `HEARTBEAT_TIMEOUT` | `10` | Evict a node after this many seconds without a heartbeat |

---

## Credits

Built on top of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov and contributors. The RPC backend, the tensor split mechanism, and `llama-server` are all upstream. AI Party adds the orchestration layer, the cluster manager, the web frontend, the desktop GUI, and the worker client.

<div align="center">

**Your laptop, your phone, your old desktop — together.**

</div>
