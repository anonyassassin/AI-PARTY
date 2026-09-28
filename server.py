import os
import re
import json
import time
import codecs
import subprocess
import threading
import platform
import collections
import httpx
import psutil
from typing import Optional, Dict, Any, List

# Executable extension based on OS
EXE_EXT = ".exe" if platform.system() == "Windows" else ""


def resolve_llama_bin(bin_name: str, custom_dir: Optional[str] = None) -> str:
    target_name = f"{bin_name}{EXE_EXT}"

    if custom_dir:
        candidate = os.path.join(custom_dir, target_name)
        if os.path.exists(candidate):
            return candidate

    env_llama_dir = os.getenv("LLAMA_DIR")
    if env_llama_dir:
        candidate = os.path.join(env_llama_dir, target_name)
        if os.path.exists(candidate):
            return candidate

    project_root = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(project_root, "llama.cpp", "build", "bin"),
        os.path.join(project_root, "llama.cpp", "build", "bin", "Release"),
        os.path.join(project_root, "bin"),
        os.path.join(project_root, "llama.cpp"),
    ]

    for path in search_paths:
        candidate = os.path.join(path, target_name)
        if os.path.exists(candidate):
            return candidate

    return target_name


def get_models_dir(custom_model_dir: Optional[str] = None) -> str:
    """
    Resolve the models directory. Search order:
      1. Explicit custom_model_dir argument
      2. MODEL_DIR env var
      3. ./models next to server.py
      4. ./models next to the current working directory
      5. ../models relative to server.py
    """
    candidates = []
    if custom_model_dir:
        candidates.append(custom_model_dir)

    env_model_dir = os.getenv("MODEL_DIR")
    if env_model_dir:
        candidates.append(env_model_dir)

    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "models"))
    candidates.append(os.path.join(os.getcwd(), "models"))
    candidates.append(os.path.join(os.path.dirname(here), "models"))

    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)

    return os.path.abspath(candidates[2] if len(candidates) > 2 else "models")


def resolve_model_path(filename: str, custom_model_dir: Optional[str] = None) -> str:
    return os.path.join(get_models_dir(custom_model_dir), filename)


# Dynamic Defaults
LLAMA_SERVER_PORT = os.getenv("LLAMA_SERVER_PORT", "8080")
LLAMA_SERVER_HOST = os.getenv("LLAMA_SERVER_HOST", "0.0.0.0")

# Single source of truth for the context window llama-server is launched with.
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "8192"))

# ------------------------------------------------------------------
# RAM safety configuration
# ------------------------------------------------------------------
MODEL_OVERHEAD_MULTIPLIER = float(os.getenv("MODEL_OVERHEAD_MULTIPLIER", "1.15"))
MODEL_KV_OVERHEAD_GB = float(os.getenv("MODEL_KV_OVERHEAD_GB", "0.2"))
MIN_USABLE_GB_TO_PARTICIPATE = float(os.getenv("MIN_USABLE_GB_TO_PARTICIPATE", "0.15"))

CURRENT_SERVER_PROCESS: Optional[subprocess.Popen] = None
CURRENT_MODEL_FILENAME: Optional[str] = None

LOAD_STALL_TIMEOUT_S = float(os.getenv("LOAD_STALL_TIMEOUT_S", "300"))

VERIFIED_MODELS_FILE = os.getenv(
    "VERIFIED_MODELS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "verified_models.json"),
)

LAST_MODEL_FILE = os.getenv(
    "LAST_MODEL_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_model.json"),
)

# When set to 1/true/yes, cold start resumes the persisted last-running
# model instead of falling back to the smallest model on disk. Default is
# off so a bad manual selection doesn't persist across restarts.
RESUME_LAST_MODEL = os.getenv("RESUME_LAST_MODEL", "0").strip().lower() in ("1", "true", "yes")

_LAUNCH_LOCK = threading.RLock()
_STATE_LOCK = threading.Lock()
_LOAD_GENERATION = 0
_LAST_GOOD_MODEL: Optional[str] = None
_LAST_LAUNCH_CONFIG: Dict[str, Any] = {"active_nodes": {}, "llama_dir": None, "model_dir": None}

LOAD_STATE: Dict[str, Any] = {
    "state": "idle",        # idle | loading | ready | error
    "reason": None,         # startup | switch | topology | revert
    "message": None,
    "model": None,
    "model_name": None,
    "size_gb": 0.0,
    "phase": None,          # starting | uploading | loading | finalizing
    "progress_pct": None,
    "loaded_gb": None,
    "remote_nodes": 0,
    "network_gb": 0.0,
    "net_sent_gb": 0.0,
    "net_rate_mbps": None,
    "eta_s": None,
    "workers": [],
    "bottleneck_worker": None,
    "progress_source": None,
    "started_at": None,
    "ready_at": None,
    "last_error": None,
}


def _load_verified() -> Dict[str, float]:
    try:
        with open(VERIFIED_MODELS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[SERVER] Couldn't read {VERIFIED_MODELS_FILE} ({e}); starting with no verified models.")
    return {}


_VERIFIED_MODELS: Dict[str, float] = _load_verified()


def is_model_verified(model: Dict[str, Any]) -> bool:
    with _STATE_LOCK:
        size = _VERIFIED_MODELS.get(model["filename"])
    return size is not None and abs(size - float(model["size_gb"])) < 0.01


def _mark_verified(filename: str, size_gb: float) -> None:
    with _STATE_LOCK:
        _VERIFIED_MODELS[filename] = float(size_gb)
        snapshot = dict(_VERIFIED_MODELS)
    try:
        tmp = VERIFIED_MODELS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        os.replace(tmp, VERIFIED_MODELS_FILE)
    except Exception as e:
        print(f"[SERVER] Couldn't persist verified models to {VERIFIED_MODELS_FILE}: {e}")


def get_verified_models() -> Dict[str, float]:
    with _STATE_LOCK:
        return dict(_VERIFIED_MODELS)


def get_last_model() -> Optional[str]:
    """Returns the filename of the model that was last successfully running, or None."""
    try:
        with open(LAST_MODEL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("filename"):
            return str(data["filename"])
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[SERVER] Couldn't read {LAST_MODEL_FILE}: {e}")
    return None


def _persist_last_model(filename: str) -> None:
    try:
        tmp = LAST_MODEL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"filename": filename, "at": time.time()}, f, indent=2)
        os.replace(tmp, LAST_MODEL_FILE)
    except Exception as e:
        print(f"[SERVER] Couldn't persist last model to {LAST_MODEL_FILE}: {e}")


def _clear_last_model() -> None:
    try:
        if os.path.exists(LAST_MODEL_FILE):
            os.remove(LAST_MODEL_FILE)
    except Exception as e:
        print(f"[SERVER] Couldn't remove {LAST_MODEL_FILE}: {e}")


def get_load_status() -> Dict[str, Any]:
    with _STATE_LOCK:
        status = dict(LOAD_STATE)
        if status["last_error"]:
            status["last_error"] = dict(status["last_error"])
        status["workers"] = [dict(w) for w in status.get("workers", [])]
    if status["started_at"]:
        end = status["ready_at"] if status["state"] != "loading" and status["ready_at"] else time.time()
        status["elapsed_s"] = round(max(end - status["started_at"], 0.0), 1)
    else:
        status["elapsed_s"] = None
    return status


def is_loading() -> bool:
    with _STATE_LOCK:
        return LOAD_STATE["state"] == "loading"


def _begin_load_locked(model: Dict[str, Any], reason: str, message: Optional[str]) -> int:
    global _LOAD_GENERATION, CURRENT_MODEL_FILENAME
    _LOAD_GENERATION += 1
    CURRENT_MODEL_FILENAME = model["filename"]
    LOAD_STATE.update({
        "state": "loading",
        "reason": reason,
        "message": message,
        "model": model["filename"],
        "model_name": model["name"],
        "size_gb": model["size_gb"],
        "phase": "starting",
        "progress_pct": None,
        "loaded_gb": None,
        "remote_nodes": 0,
        "network_gb": 0.0,
        "net_sent_gb": 0.0,
        "net_rate_mbps": None,
        "eta_s": None,
        "workers": [],
        "bottleneck_worker": None,
        "progress_source": None,
        "started_at": time.time(),
        "ready_at": None,
    })
    return _LOAD_GENERATION


def begin_manual_switch(model: Dict[str, Any]) -> Optional[int]:
    with _STATE_LOCK:
        if LOAD_STATE["state"] == "loading":
            return None
        LOAD_STATE["last_error"] = None
        return _begin_load_locked(model, "switch", f"Switching to {model['name']}")


def begin_topology_reload(model: Dict[str, Any], message: Optional[str] = None) -> Optional[int]:
    with _STATE_LOCK:
        if LOAD_STATE["state"] == "loading" and LOAD_STATE["reason"] == "switch":
            return None
        LOAD_STATE["last_error"] = None
        return _begin_load_locked(model, "topology", message or "Cluster topology changed — reloading")


def _update_load(gen: int, **fields) -> bool:
    with _STATE_LOCK:
        if gen != _LOAD_GENERATION or LOAD_STATE["state"] != "loading":
            return False
        LOAD_STATE.update(fields)
        return True


def _mark_ready(gen: int) -> None:
    global _LAST_GOOD_MODEL
    with _STATE_LOCK:
        if gen != _LOAD_GENERATION or LOAD_STATE["state"] != "loading":
            return
        LOAD_STATE.update({
            "state": "ready",
            "phase": None,
            "progress_pct": 100.0,
            "loaded_gb": LOAD_STATE["size_gb"],
            "eta_s": 0,
            "ready_at": time.time(),
        })
        filename, size_gb, name = LOAD_STATE["model"], LOAD_STATE["size_gb"], LOAD_STATE["model_name"]
        started = LOAD_STATE["started_at"] or time.time()
        _LAST_GOOD_MODEL = filename
    _mark_verified(filename, size_gb)
    _persist_last_model(filename)
    print(f"[SERVER] '{name}' is loaded and serving (took {time.time() - started:.1f}s).")


def _fail_load(gen: int, message: str, allow_revert: bool = True) -> None:
    global CURRENT_MODEL_FILENAME
    with _STATE_LOCK:
        if gen != _LOAD_GENERATION or LOAD_STATE["state"] != "loading":
            return
        failed_model = LOAD_STATE["model"]
        failed_name = LOAD_STATE["model_name"]
        was_revert = LOAD_STATE["reason"] == "revert"
        last_good = _LAST_GOOD_MODEL
        LOAD_STATE.update({"state": "error", "phase": None, "eta_s": None, "ready_at": time.time()})
        LOAD_STATE["last_error"] = {
            "model": failed_model,
            "model_name": failed_name,
            "message": message,
            "at": time.time(),
            "reverted_to": None,
        }
        CURRENT_MODEL_FILENAME = None
    print(f"[SERVER] Load of '{failed_name}' failed: {message}")

    if not allow_revert or was_revert:
        return

    cfg = dict(_LAST_LAUNCH_CONFIG)
    models = discover_models(cfg["model_dir"])
    by_name = {m["filename"]: m for m in models}
    target = None
    if last_good and last_good != failed_model and last_good in by_name:
        target = by_name[last_good]
    elif models and models[0]["filename"] != failed_model:
        target = models[0]
    if target is None:
        return

    with _STATE_LOCK:
        if LOAD_STATE["last_error"] and LOAD_STATE["last_error"]["model"] == failed_model:
            LOAD_STATE["last_error"]["reverted_to"] = target["name"]
    print(f"[SERVER] Reverting to '{target['name']}'.")
    start_or_restart_server(
        cfg["active_nodes"], cfg["llama_dir"], cfg["model_dir"], target["filename"],
        reason="revert",
        message=f"{failed_name} failed to load — reverting to {target['name']}",
    )


# ------------------------------------------------------------------
# llama-server output parsing / load progress
# ------------------------------------------------------------------
_DOTS_RE = re.compile(r"^\.+$")
_RPC_BUFFER_RE = re.compile(r"RPC\[([^\]]+)\]\s+model buffer size\s*=\s*([\d.]+)\s*MiB")
_RPC_TENSOR_RE = re.compile(r"RPC\[([^\]]+)\][^\n]*?([\d.]+)\s*MiB")
_ERROR_HINT_RE = re.compile(r"error|failed|unable|cannot|abort|exception", re.IGNORECASE)
_OOM_RE = re.compile(r"out of memory|failed to allocate|cannot allocate|bad_alloc|OOM", re.IGNORECASE)


class _LoadMonitor:
    def __init__(self):
        self.dots = 0
        self.dots_done = False
        self.rpc_bytes_by_endpoint: Dict[str, float] = {}
        self.rpc_buffer_bytes_unattributed = 0.0
        self.tail = collections.deque(maxlen=40)

    def total_rpc_bytes(self) -> float:
        return sum(self.rpc_bytes_by_endpoint.values()) + self.rpc_buffer_bytes_unattributed


def _handle_line(line: str, monitor: _LoadMonitor, complete: bool) -> None:
    stripped = line.strip()
    if _DOTS_RE.match(stripped):
        monitor.dots = max(monitor.dots, len(stripped))
        if complete:
            monitor.dots_done = True
        return
    if not complete:
        return

    m = _RPC_BUFFER_RE.search(stripped)
    if m:
        endpoint = m.group(1).strip()
        try:
            b = float(m.group(2)) * 1024 * 1024
            monitor.rpc_bytes_by_endpoint[endpoint] = monitor.rpc_bytes_by_endpoint.get(endpoint, 0.0) + b
        except ValueError:
            pass
        return

    m = _RPC_TENSOR_RE.search(stripped)
    if m:
        endpoint = m.group(1).strip()
        try:
            b = float(m.group(2)) * 1024 * 1024
            monitor.rpc_bytes_by_endpoint[endpoint] = monitor.rpc_bytes_by_endpoint.get(endpoint, 0.0) + b
        except ValueError:
            pass
        return

    if stripped.upper().startswith("RPC") and "MiB" in stripped:
        m2 = re.search(r"([\d.]+)\s*MiB", stripped)
        if m2:
            try:
                monitor.rpc_buffer_bytes_unattributed += float(m2.group(1)) * 1024 * 1024
            except ValueError:
                pass


def _log_reader(proc: subprocess.Popen, monitor: _LoadMonitor) -> None:
    pipe = proc.stdout
    if pipe is None:
        return
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    partial = ""
    try:
        while True:
            chunk = pipe.read1(4096) if hasattr(pipe, "read1") else pipe.read(4096)
            if not chunk:
                break
            text = partial + decoder.decode(chunk).replace("\r", "\n")
            lines = text.split("\n")
            partial = lines.pop()
            for line in lines:
                print(f"[LLAMA] {line}", flush=True)
                if line.strip():
                    monitor.tail.append(line)
                _handle_line(line, monitor, complete=True)
            _handle_line(partial, monitor, complete=False)
    except Exception:
        pass
    finally:
        if partial.strip():
            print(f"[LLAMA] {partial}", flush=True)
            monitor.tail.append(partial)
        try:
            pipe.close()
        except Exception:
            pass


def _net_bytes_sent() -> int:
    try:
        return int(psutil.net_io_counters().bytes_sent)
    except Exception:
        return 0


def _failure_message(name: str, returncode: Optional[int], monitor: _LoadMonitor) -> str:
    tail = list(monitor.tail)
    if any(_OOM_RE.search(l) for l in tail):
        msg = f"{name} ran out of memory while loading."
    else:
        msg = f"llama-server exited (code {returncode}) while loading {name}."
    detail = next((l.strip() for l in reversed(tail) if _ERROR_HINT_RE.search(l)), None)
    if detail:
        msg += f" Last error: {detail[:240]}"
    return msg


RATE_WARMUP_S = float(os.getenv("LOAD_ETA_WARMUP_S", "6"))
RATE_MIN_SAMPLES = int(os.getenv("LOAD_ETA_MIN_SAMPLES", "10"))


class _RateTracker:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.rate: Optional[float] = None
        self.samples = 0
        self._last_t: Optional[float] = None
        self._last_v: float = 0.0

    def update(self, value: float, now: float) -> None:
        if self._last_t is None:
            self._last_t, self._last_v = now, value
            return
        dt = now - self._last_t
        if dt <= 0:
            return
        inst = max(value - self._last_v, 0.0) / dt
        self.rate = inst if self.rate is None else (self.alpha * inst + (1 - self.alpha) * self.rate)
        self._last_t, self._last_v = now, value
        self.samples += 1

    def stable(self) -> Optional[float]:
        if self.samples < RATE_MIN_SAMPLES or self.rate is None or self.rate <= 0:
            return None
        return self.rate


def _watch_load(proc: subprocess.Popen, gen: int, monitor: _LoadMonitor,
                model: Dict[str, Any], planned_targets: Dict[str, Dict[str, Any]],
                remote_nodes: int, started_at: float) -> None:
    health_url = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health"
    net_start = _net_bytes_sent()
    total_bytes = float(model["size_gb"]) * (1024 ** 3)

    agg_rate = _RateTracker(alpha=0.3)

    first_dots_point: Optional[tuple] = None
    first_byte_t: Optional[float] = None

    last_progress_t = time.time()
    last_progress_mark = (0, 0)

    with httpx.Client(timeout=1.0) as client:
        while True:
            time.sleep(0.5)
            with _STATE_LOCK:
                if gen != _LOAD_GENERATION:
                    return

            rc = proc.poll()
            if rc is not None:
                time.sleep(0.5)
                tail = list(monitor.tail)
                if tail:
                    print("[SERVER] --- llama-server output tail before exit ---")
                    for line in tail[-15:]:
                        print(f"[SERVER]   {line}")
                    print("[SERVER] --- end of tail ---")
                else:
                    print("[SERVER] llama-server exited with no captured output. "
                          "This usually means an env/cwd/binary problem, not a model problem.")
                _fail_load(gen, _failure_message(model["name"], rc, monitor))
                return

            try:
                if client.get(health_url).status_code == 200:
                    with _STATE_LOCK:
                        for w in LOAD_STATE.get("workers", []):
                            w["sent_gb"] = w["target_gb"]
                            w["pct"] = 100.0
                            w["eta_s"] = 0
                            w["done"] = True
                    _mark_ready(gen)
                    return
            except httpx.HTTPError:
                pass

            now = time.time()

            net_sent = max(_net_bytes_sent() - net_start, 0)
            agg_rate.update(float(net_sent), now)
            agg_rate_val = agg_rate.stable()

            if net_sent > 0 and first_byte_t is None:
                first_byte_t = now

            real_by_endpoint = monitor.rpc_bytes_by_endpoint
            workers_snapshot: List[Dict[str, Any]] = []

            if real_by_endpoint:
                endpoints = list(real_by_endpoint.keys())
                total_planned_bytes = sum(real_by_endpoint.values())
                progress_source = "rpc_buffers"
            else:
                endpoints = list(planned_targets.keys())
                total_planned_bytes = sum(
                    t["planned_gb"] * (1024 ** 3) for t in planned_targets.values()
                ) if planned_targets else 0.0
                progress_source = "network" if total_planned_bytes > 0 else None

            if total_planned_bytes > 0 and endpoints:
                sent_bytes = min(net_sent, total_planned_bytes)
                for ep in endpoints:
                    if real_by_endpoint:
                        target_bytes = real_by_endpoint[ep]
                    else:
                        target_bytes = planned_targets[ep]["planned_gb"] * (1024 ** 3)
                    fraction = target_bytes / total_planned_bytes if total_planned_bytes > 0 else 0.0
                    worker_sent = sent_bytes * fraction
                    worker_pct = min(99.0, (worker_sent / target_bytes * 100.0)) if target_bytes > 0 else 0.0

                    worker_eta = None
                    if agg_rate_val and target_bytes > 0:
                        remaining = max(target_bytes - worker_sent, 0.0)
                        worker_eta = remaining / agg_rate_val

                    info = planned_targets.get(ep, {})
                    workers_snapshot.append({
                        "endpoint": ep,
                        "hostname": info.get("hostname", ep),
                        "target_gb": round(target_bytes / (1024 ** 3), 2),
                        "sent_gb": round(worker_sent / (1024 ** 3), 2),
                        "pct": round(worker_pct, 1),
                        "eta_s": round(worker_eta) if worker_eta is not None else None,
                        "done": False,
                    })

            if monitor.dots > 0:
                progress_source = "dots"
                if first_dots_point is None:
                    first_dots_point = (now, monitor.dots)
                if monitor.dots_done:
                    pct: Optional[float] = 100.0
                else:
                    pct = float(min(monitor.dots, 99))
            elif total_planned_bytes > 0 and net_sent > 0:
                pct = min(99.0, net_sent / total_planned_bytes * 100.0)
            else:
                pct = None

            eta: Optional[float] = None

            if (monitor.dots > 0 and not monitor.dots_done
                    and first_dots_point is not None
                    and now - first_dots_point[0] >= RATE_WARMUP_S):
                dt = now - first_dots_point[0]
                dp = monitor.dots - first_dots_point[1]
                if dp >= 2 and dt > 0:
                    eta = (100 - monitor.dots) * dt / dp

            if eta is None and agg_rate_val and total_planned_bytes > 0 and first_byte_t is not None:
                if now - first_byte_t >= RATE_WARMUP_S:
                    remaining = max(total_planned_bytes - net_sent, 0.0)
                    eta = remaining / agg_rate_val

            bottleneck = None
            if workers_snapshot:
                incomplete = [w for w in workers_snapshot if not w["done"]]
                if incomplete:
                    bottleneck = max(incomplete, key=lambda w: w["target_gb"] - w["sent_gb"])["endpoint"]

            mark = (monitor.dots, net_sent // (256 * 1024))
            if mark != last_progress_mark:
                last_progress_mark = mark
                last_progress_t = now
            elif now - last_progress_t > LOAD_STALL_TIMEOUT_S:
                print(f"[SERVER] No load progress for {LOAD_STALL_TIMEOUT_S:.0f}s — treating as hung.")
                _terminate(proc)
                _fail_load(gen, f"Loading {model['name']} stalled (no progress for "
                                f"{LOAD_STALL_TIMEOUT_S:.0f}s). Check the worker nodes and network.")
                return

            if monitor.dots_done:
                phase = "finalizing"
            elif monitor.dots > 0:
                phase = "loading"
            elif net_sent > 0:
                phase = "uploading"
            else:
                phase = "starting"

            if not _update_load(
                gen,
                phase=phase,
                progress_pct=round(pct, 1) if pct is not None else None,
                loaded_gb=round(total_bytes * pct / 100 / (1024 ** 3), 2) if pct is not None else None,
                remote_nodes=remote_nodes,
                network_gb=round(total_planned_bytes / (1024 ** 3), 2) if total_planned_bytes > 0 else 0.0,
                net_sent_gb=round(min(net_sent, total_planned_bytes) / (1024 ** 3), 2) if total_planned_bytes > 0 else 0.0,
                net_rate_mbps=round(agg_rate_val / (1024 ** 2), 1) if agg_rate_val else None,
                eta_s=round(eta) if eta is not None else None,
                workers=workers_snapshot,
                bottleneck_worker=bottleneck,
                progress_source=progress_source,
            ):
                return


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def get_host_system_details() -> dict:
    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024 ** 3), 2)
    available_ram_gb = round(vm.available / (1024 ** 3), 2)
    cpu_cores = os.cpu_count()
    return {
        "system_details": {
            "system_name": platform.system(),
            "hostname": f"HOST-{platform.node()}",
            "total_ram_gb": float(total_ram_gb),
            "available_ram_gb": float(available_ram_gb),
            "cpu_count": int(cpu_cores) if cpu_cores is not None else 4,
        }
    }


HOST_OVERHEAD_RESERVE_GB = float(os.getenv("HOST_OVERHEAD_RESERVE_GB", "0.5"))
RAM_SAFETY_RESERVE_GB = float(os.getenv("RAM_SAFETY_RESERVE_GB", "0.3"))


def _node_usable_gb(specs: dict, is_host: bool = False) -> float:
    available = float(specs.get("available_ram_gb", specs.get("total_ram_gb", 0)))
    reserve = RAM_SAFETY_RESERVE_GB + (HOST_OVERHEAD_RESERVE_GB if is_host else 0.0)
    return round(max(available - reserve, 0.0), 2)


def calculate_cluster_capacity(active_nodes: dict) -> Dict[str, Any]:
    host_profile = get_host_system_details()
    participants = [("HOST_MACHINE", host_profile["system_details"], True)]

    for node_id, info in active_nodes.items():
        participants.append((node_id, info.get("system_details", {}), False))

    per_node = []
    total_raw = 0.0
    total_usable = 0.0
    for pid, specs, is_host in participants:
        usable = _node_usable_gb(specs, is_host=is_host)
        total_raw += float(specs.get("total_ram_gb", 0.0))
        total_usable += usable
        per_node.append({
            "id": pid,
            "hostname": specs.get("hostname", pid),
            "is_host": is_host,
            "total_ram_gb": float(specs.get("total_ram_gb", 0.0)),
            "available_ram_gb": float(specs.get("available_ram_gb", specs.get("total_ram_gb", 0.0))),
            "usable_ram_gb": usable,
            "excluded": (not is_host) and usable < MIN_USABLE_GB_TO_PARTICIPATE,
            "cpu_count": int(specs.get("cpu_count", 0)),
        })

    return {
        "total_raw_gb": round(total_raw, 2),
        "total_usable_gb": round(total_usable, 2),
        "nodes": per_node,
    }


def discover_models(model_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    directory = get_models_dir(model_dir)
    models: List[Dict[str, Any]] = []
    if os.path.isdir(directory):
        for fname in sorted(os.listdir(directory)):
            if not fname.lower().endswith(".gguf"):
                continue
            full_path = os.path.join(directory, fname)
            try:
                size_gb = round(os.path.getsize(full_path) / (1024 ** 3), 2)
            except OSError:
                continue
            min_ram_gb = round(size_gb * MODEL_OVERHEAD_MULTIPLIER + MODEL_KV_OVERHEAD_GB, 2)
            pretty = os.path.splitext(fname)[0].replace('-', ' ').replace('_', ' ')
            models.append({
                "filename": fname,
                "name": pretty.title(),
                "size_gb": size_gb,
                "min_ram_gb": min_ram_gb,
            })
    models.sort(key=lambda m: m["size_gb"])
    return models


def pick_best_model(model_dir: Optional[str], usable_gb: float, manual_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
    models = discover_models(model_dir)
    if not models:
        return None

    if manual_filename:
        return next((m for m in models if m["filename"] == manual_filename), None)

    fitting = [m for m in models if m["min_ram_gb"] <= usable_gb]
    if fitting:
        return max(fitting, key=lambda m: m["size_gb"])
    return min(models, key=lambda m: m["size_gb"])


def resolve_startup_model(model_dir: Optional[str], requested: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Startup / topology-reload model selection:
      1. Explicit `requested` filename, if it exists on disk.
      2. Persisted last running model, if RESUME_LAST_MODEL is enabled and
         it still exists on disk.
      3. Smallest model on disk.

    RESUME_LAST_MODEL defaults to off so a bad manual selection (e.g. a
    7B that's too slow for the current cluster) doesn't persist across
    restarts. Set RESUME_LAST_MODEL=1 in the environment to opt back in.
    """
    models = discover_models(model_dir)
    if not models:
        return None
    by_name = {m["filename"]: m for m in models}

    if requested and requested in by_name:
        return by_name[requested]

    if RESUME_LAST_MODEL:
        last = get_last_model()
        if last and last in by_name:
            print(f"[SERVER] RESUME_LAST_MODEL is on — resuming '{by_name[last]['name']}'.")
            return by_name[last]

    smallest = min(models, key=lambda m: m["size_gb"])
    print(f"[SERVER] Cold start: choosing smallest available model '{smallest['name']}'.")
    return smallest


def calculate_tensor_split(active_nodes: dict) -> Dict[str, Any]:
    capacity = calculate_cluster_capacity(active_nodes)
    host_node = next(n for n in capacity["nodes"] if n["is_host"])
    worker_nodes = [n for n in capacity["nodes"] if not n["is_host"] and not n["excluded"]]
    excluded_nodes = [n for n in capacity["nodes"] if not n["is_host"] and n["excluded"]]

    weights = [max(host_node["usable_ram_gb"], 0.1)]
    endpoints = []

    for n in worker_nodes:
        info = active_nodes.get(n["id"])
        if not info:
            continue
        ep = f"{info['ip_address']}:{info['port']}"
        endpoints.append((ep, n["id"], n["hostname"]))
        weights.append(max(n["usable_ram_gb"], 0.1))

    total_weight = sum(weights)
    if total_weight > 0:
        split_values = [max(1, round((w / total_weight) * 100)) for w in weights]
    else:
        split_values = [1] * len(weights)

    return {
        "tensor_split": ",".join(str(v) for v in split_values),
        "tensor_split_values": split_values,
        "rpc_string": ",".join(ep for ep, _, _ in endpoints),
        "total_usable_gb": capacity["total_usable_gb"],
        "excluded_nodes": excluded_nodes,
        "workers": [(ep, hostname, w) for (ep, _, hostname), w in zip(endpoints, weights[1:])],
        "host_weight": weights[0],
    }


def _build_planned_targets(split_info: Dict[str, Any], model_size_gb: float) -> Dict[str, Dict[str, Any]]:
    workers = split_info.get("workers", [])
    split_values = split_info.get("tensor_split_values", [])
    if not workers or not split_values or len(split_values) < 2:
        return {}

    total_weight = sum(split_values)
    if total_weight <= 0:
        return {}

    model_bytes = model_size_gb * (1024 ** 3)
    planned: Dict[str, Dict[str, Any]] = {}
    for (ep, hostname, weight), split_val in zip(workers, split_values[1:]):
        share = split_val / total_weight
        planned[ep] = {
            "hostname": hostname,
            "weight": weight,
            "planned_gb": round(model_bytes * share / (1024 ** 3), 3),
        }
    return planned


def get_current_launch_info() -> Dict[str, Any]:
    cfg = dict(_LAST_LAUNCH_CONFIG)
    info = calculate_tensor_split(cfg.get("active_nodes") or {})
    info["active_model"] = CURRENT_MODEL_FILENAME
    info["llama_dir"] = cfg.get("llama_dir")
    info["model_dir"] = cfg.get("model_dir")
    return info


def _build_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HOME", os.path.expanduser("~"))
    if not env.get("TMPDIR"):
        env["TMPDIR"] = "/tmp"
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    return env


def _strip_quarantine(binary_path: str) -> None:
    if platform.system() != "Darwin":
        return
    try:
        subprocess.run(
            ["xattr", "-dr", "com.apple.quarantine", binary_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception:
        pass


def start_or_restart_server(
    active_nodes: dict,
    llama_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    model_filename: Optional[str] = None,
    reason: str = "startup",
    message: Optional[str] = None,
):
    global CURRENT_SERVER_PROCESS, CURRENT_MODEL_FILENAME

    split_info = calculate_tensor_split(active_nodes)
    usable_gb = split_info["total_usable_gb"]

    if model_filename is not None:
        chosen = pick_best_model(model_dir, usable_gb, manual_filename=model_filename)
        if chosen is None:
            print(f"[SERVER] Requested model '{model_filename}' not found on disk — falling back to startup resolution.")
            chosen = resolve_startup_model(model_dir, None)
    else:
        chosen = resolve_startup_model(model_dir, None)

    if chosen is None:
        print(f"[SERVER] No .gguf models found in '{get_models_dir(model_dir)}' — nothing to launch.")
        CURRENT_SERVER_PROCESS = None
        CURRENT_MODEL_FILENAME = None
        return

    if model_filename is None and chosen["min_ram_gb"] > usable_gb and not is_model_verified(chosen):
        print(f"[SERVER] '{chosen['name']}' needs ~{chosen['min_ram_gb']}GB usable RAM but the "
              f"cluster estimate is only ~{usable_gb}GB right now.")
        smallest = min(discover_models(model_dir), key=lambda m: m["size_gb"])
        if smallest["filename"] != chosen["filename"]:
            print(f"[SERVER] Falling back to the smallest available model instead: '{smallest['name']}'.")
            chosen = smallest

    if chosen["min_ram_gb"] > usable_gb and not is_model_verified(chosen):
        print(f"[SERVER] WARNING: launching '{chosen['name']}' anyway even though the estimate "
              f"(~{chosen['min_ram_gb']}GB) exceeds usable RAM (~{usable_gb}GB). Watch for OOM.")

    if split_info["excluded_nodes"]:
        names = ", ".join(n["hostname"] for n in split_info["excluded_nodes"])
        print(f"[SERVER] Excluding low-memory node(s) from the split: {names}")

    with _LAUNCH_LOCK:
        if CURRENT_SERVER_PROCESS is not None:
            _terminate(CURRENT_SERVER_PROCESS)

        model_full_path = resolve_model_path(chosen["filename"], custom_model_dir=model_dir)
        llama_server_bin = resolve_llama_bin("llama-server", custom_dir=llama_dir)

        _strip_quarantine(llama_server_bin)

        cmd = [
            llama_server_bin,
            "-m", model_full_path,
            "-c", str(CONTEXT_SIZE),
        ]

        if split_info["rpc_string"]:
            cmd.extend(["--rpc", split_info["rpc_string"]])

        cmd.extend([
            "--tensor-split", split_info["tensor_split"],
            "-b", "256",
            "-ub", "256",
            "-t", "8",
            "--port", LLAMA_SERVER_PORT,
            "--host", LLAMA_SERVER_HOST
        ])

        _LAST_LAUNCH_CONFIG.update({
            "active_nodes": active_nodes,
            "llama_dir": llama_dir,
            "model_dir": model_dir,
        })

        with _STATE_LOCK:
            already_loading_same = (
                LOAD_STATE["state"] == "loading"
                and LOAD_STATE["model"] == chosen["filename"]
            )
            if already_loading_same:
                gen = _LOAD_GENERATION
            else:
                gen = _begin_load_locked(chosen, reason, message)

        print(f"[SERVER] Spawning: {cmd!r}")
        child_cwd = os.path.dirname(llama_server_bin) or None
        print(f"[SERVER] Spawn env: HOME={os.environ.get('HOME')!r} TMPDIR={os.environ.get('TMPDIR')!r} "
              f"cwd={child_cwd!r}")

        child_env = _build_subprocess_env()

        try:
            CURRENT_SERVER_PROCESS = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=child_env,
                cwd=child_cwd,
            )
        except FileNotFoundError:
            print(f"[SERVER] llama-server binary not found at {llama_server_bin!r}. "
                  f"Pass --llama-dir or set LLAMA_DIR.")
            _fail_load(gen, f"llama-server binary not found at {llama_server_bin}", allow_revert=False)
            return
        except PermissionError:
            print(f"[SERVER] llama-server at {llama_server_bin!r} is not executable. "
                  f"Run: chmod +x {llama_server_bin}")
            _fail_load(gen, f"llama-server not executable at {llama_server_bin}", allow_revert=False)
            return
        except OSError as e:
            print(f"[SERVER] Failed to spawn llama-server: {e}")
            _fail_load(gen, f"Failed to spawn llama-server: {e}", allow_revert=False)
            return

        CURRENT_MODEL_FILENAME = chosen["filename"]

        monitor = _LoadMonitor()
        reader_thread = threading.Thread(
            target=_log_reader,
            args=(CURRENT_SERVER_PROCESS, monitor),
            daemon=True,
        )
        reader_thread.start()

        planned_targets = _build_planned_targets(split_info, float(chosen["size_gb"]))
        remote_nodes = len(planned_targets)
        watcher = threading.Thread(
            target=_watch_load,
            args=(
                CURRENT_SERVER_PROCESS,
                gen,
                monitor,
                chosen,
                planned_targets,
                remote_nodes,
                time.time(),
            ),
            daemon=True,
        )
        watcher.start()

        planned_total = sum(t["planned_gb"] for t in planned_targets.values())
        print(f"[SERVER] Launched '{chosen['name']}' ({chosen['filename']}) using '{llama_server_bin}'. "
              f"PID {CURRENT_SERVER_PROCESS.pid} | tensor-split {split_info['tensor_split']} | "
              f"usable RAM {usable_gb}GB | planned network transfer {planned_total:.2f}GB "
              f"across {remote_nodes} worker(s)")


def stop_server():
    global CURRENT_SERVER_PROCESS, CURRENT_MODEL_FILENAME
    with _LAUNCH_LOCK:
        if CURRENT_SERVER_PROCESS is not None:
            _terminate(CURRENT_SERVER_PROCESS)
        CURRENT_SERVER_PROCESS = None
        CURRENT_MODEL_FILENAME = None
