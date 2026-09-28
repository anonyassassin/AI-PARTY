import os
import subprocess
import requests
import time
import threading
import uuid
import platform
import argparse
import psutil

NODE_ID = str(uuid.uuid4())

# Executable extension based on OS. Duplicated from server.py so this file
# has no local imports and can be deployed on a client machine on its own.
EXE_EXT = ".exe" if platform.system() == "Windows" else ""


def resolve_llama_bin(bin_name: str, custom_dir: str = None) -> str:
    """
    Locate a llama.cpp binary. Search order:
      1. Explicit custom_dir argument
      2. LLAMA_DIR environment variable
      3. Common project-relative locations next to this script
      4. Bare name (relies on PATH)
    """
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

    here = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(here, "llama.cpp", "build", "bin"),
        os.path.join(here, "llama.cpp", "build", "bin", "Release"),
        os.path.join(here, "bin"),
        os.path.join(here, "llama.cpp"),
        os.getcwd(),
    ]

    for path in search_paths:
        candidate = os.path.join(path, target_name)
        if os.path.exists(candidate):
            return candidate

    return target_name


def get_system_details():
    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024 ** 3), 2)
    available_ram_gb = round(vm.available / (1024 ** 3), 2)
    return {
        "system_name": platform.system(),
        "os_release": platform.release(),
        "hostname": platform.node(),
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "cpu_count": os.cpu_count() or 0,
    }


def register_with_api(api_url, node_ip, rpc_port, system_specs, interval):
    """
    Register this node with the API host, retrying forever until it
    succeeds.

    This is critical: if the initial POST fails (API still starting up,
    network blip), or the API host restarts and wipes its in-memory
    NODE_REGISTRY, we must NOT just keep heartbeating an unregistered
    node ID — we must retry the registration until the host accepts us.
    """
    connect_url = f"{api_url.rstrip('/')}/connect"
    payload = {
        "node_id": NODE_ID,
        "ip_address": node_ip,
        "port": rpc_port,
        "system_details": system_specs,
    }
    while True:
        try:
            res = requests.post(
                connect_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if res.status_code == 200:
                print(f"[+] Registered with API host ({res.status_code})", flush=True)
                return
            print(f"[-] Registration rejected ({res.status_code}): {res.text[:200]}", flush=True)
        except Exception as e:
            print(f"[-] Registration failed, retrying in {interval}s: {e}", flush=True)
        time.sleep(interval)


def poll_connect_endpoint(api_url, node_ip, rpc_port, interval):
    print("[+] Collecting system performance specs...")
    system_specs = get_system_details()

    alive_url = f"{api_url.rstrip('/')}/alive"

    # Never proceed to heartbeating until we are actually registered.
    register_with_api(api_url, node_ip, rpc_port, system_specs, interval)

    while True:
        try:
            available_ram_gb = round(psutil.virtual_memory().available / (1024 ** 3), 2)
            params = {"node_id": NODE_ID, "available_ram_gb": available_ram_gb}
            response = requests.get(alive_url, params=params, timeout=5)
            data = response.json()
            status = data.get("status")

            if status == "ok":
                print(f"[+] Heartbeat ok | free RAM: {available_ram_gb}GB", flush=True)
            else:
                # The API host doesn't know us anymore — most likely it was
                # restarted and its in-memory NODE_REGISTRY was wiped.
                # Re-register instead of heartbeating into the void.
                print(
                    f"[-] API says '{data.get('message', status)}' — re-registering...",
                    flush=True,
                )
                register_with_api(api_url, node_ip, rpc_port, get_system_details(), interval)
        except Exception as e:
            print(f"[-] Heartbeat failed: {e}", flush=True)

        time.sleep(interval)


def run_and_poll(args):
    rpc_bin = resolve_llama_bin("ggml-rpc-server", custom_dir=args.llama_dir)

    command = [
        rpc_bin,
        "-H", args.host,
        "-p", str(args.port),
    ]

    # Give the child a sane env. A missing HOME/TMPDIR is a common cause
    # of early aborts on some setups — same reasoning as the server-side
    # spawn fix.
    env = os.environ.copy()
    env.setdefault("HOME", os.path.expanduser("~"))
    if not env.get("TMPDIR"):
        env["TMPDIR"] = "/tmp"

    print(f"[*] Resolved RPC binary: {rpc_bin}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except FileNotFoundError:
        print(
            f"[-] Could not find '{rpc_bin}'. Install llama.cpp's "
            f"ggml-rpc-server or pass --llama-dir pointing at the "
            f"folder containing it."
        )
        return
    except PermissionError:
        print(f"[-] '{rpc_bin}' is not executable. Run: chmod +x {rpc_bin}")
        return
    except OSError as e:
        print(f"[-] Failed to launch RPC server: {e}")
        return

    print(f"[*] Launched RPC server binary '{rpc_bin}' (PID: {process.pid}) on {args.host}:{args.port}")

    poll_thread = threading.Thread(
        target=poll_connect_endpoint,
        args=(args.api_url, args.node_ip, args.port, args.interval),
        daemon=True,
    )
    poll_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Shutting down client...")
        process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Party RPC Client Node")
    parser.add_argument("--api-url", type=str, default=os.getenv("API_URL", "http://127.0.0.1:8000"),
                        help="API server base URL")
    parser.add_argument("--node-ip", type=str, default=os.getenv("NODE_IP", "127.0.0.1"),
                        help="IP address advertised to cluster manager")
    parser.add_argument("--port", type=int, default=int(os.getenv("RPC_PORT", "50052")),
                        help="Port to run ggml-rpc-server")
    parser.add_argument("--host", type=str, default=os.getenv("RPC_HOST", "0.0.0.0"),
                        help="Host address to bind ggml-rpc-server")
    parser.add_argument("--interval", type=int, default=3,
                        help="Alive heartbeat interval in seconds")
    parser.add_argument("--llama-dir", type=str, default=None,
                        help="Custom directory containing llama.cpp binaries")

    args = parser.parse_args()
    run_and_poll(args)