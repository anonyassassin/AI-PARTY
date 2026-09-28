import os
import argparse
import asyncio
import time
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from fastapi.staticfiles import StaticFiles

import server as server_module
from server import (
    start_or_restart_server,
    calculate_cluster_capacity,
    discover_models,
    pick_best_model,
    resolve_startup_model,
    begin_manual_switch,
    begin_topology_reload,
    get_load_status,
    is_loading,
    is_model_verified,
    get_verified_models,
    get_last_model,
    CONTEXT_SIZE,
)

app = FastAPI(title="AI PARTY")
"""
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://100.118.45.78:3000",
    "http://100.99.40.83:3000",
    "http://100.95.91.4:3000",
    "http://100.104.33.40:3000",
    "http://192.168.1.6:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
async_client = httpx.AsyncClient(base_url=LLAMA_SERVER_URL)

NODE_REGISTRY: Dict[str, Dict[str, Any]] = {}
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "10"))

CONFIG = {
    "llama_dir": None,
    "model_dir": None,
}

MANUAL_MODEL_OVERRIDE: Optional[str] = None


def evaluate_cluster_level_change(event_type: str, node_id: str) -> None:
    """Logs current cluster usable capacity on node join/leave."""
    capacity = calculate_cluster_capacity(NODE_REGISTRY)

    print("\n" + "-" * 60)
    if event_type == "join":
        print(f"NODE JOINED: {node_id}")
    else:
        print(f"NODE LEFT: {node_id}")
    print(f"TOTAL USABLE CLUSTER CAPACITY: {capacity['total_usable_gb']:.2f} GB "
          f"(raw pooled RAM: {capacity['total_raw_gb']:.2f} GB)")
    print("-" * 60 + "\n")


class SystemDetailsSchema(BaseModel):
    system_name: str = Field(default="Unknown")
    os_release: str = Field(default="Unknown")
    hostname: str = Field(default="Unknown")
    total_ram_gb: float = Field(default=8.0)
    available_ram_gb: float = Field(default=8.0)
    cpu_count: int = Field(default=4)


class ConnectRequest(BaseModel):
    node_id: str
    ip_address: str
    port: int
    system_details: SystemDetailsSchema


class SelectModelRequest(BaseModel):
    filename: str
    force: bool = False


class ContextMessage(BaseModel):
    role: str
    content: str = ""


class ContextCheckRequest(BaseModel):
    messages: List[ContextMessage]


@app.post("/api/context")
async def check_context(payload: ContextCheckRequest):
    total_tokens = 0
    try:
        for m in payload.messages:
            if not m.content:
                continue
            res = await async_client.post("/tokenize", json={"content": m.content}, timeout=5.0)
            res.raise_for_status()
            tokens = res.json().get("tokens", [])
            total_tokens += len(tokens) + 4
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Cluster communication issue: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"llama-server tokenize call failed: {e}")

    percent = min(100, round((total_tokens / CONTEXT_SIZE) * 100)) if CONTEXT_SIZE else 0
    return {
        "used_tokens": total_tokens,
        "max_tokens": CONTEXT_SIZE,
        "percent": percent,
    }


@app.get("/api/capacity")
async def get_capacity():
    return calculate_cluster_capacity(NODE_REGISTRY)


@app.get("/api/load-status")
async def load_status():
    """Live load progress for the frontend's blocking loading overlay."""
    return get_load_status()


@app.get("/api/models")
async def get_models():
    models = discover_models(CONFIG["model_dir"])
    capacity = calculate_cluster_capacity(NODE_REGISTRY)
    current_filename = server_module.CURRENT_MODEL_FILENAME

    enriched = []
    for m in models:
        verified = is_model_verified(m)
        is_active = m["filename"] == current_filename
        # Verified models stay permanently unlocked — they've proven they
        # fit and run on this cluster. Unverified models are locked only
        # when the live usable-RAM estimate says they won't fit.
        locked = (not verified) and (not is_active) and (m["min_ram_gb"] > capacity["total_usable_gb"])
        enriched.append({
            **m,
            "locked": locked,
            "active": is_active,
            "verified": verified,
        })

    return {
        "models": enriched,
        "usable_ram_gb": capacity["total_usable_gb"],
        "raw_ram_gb": capacity["total_raw_gb"],
        "manual_override": MANUAL_MODEL_OVERRIDE,
        "active_model": current_filename,
        "loading": is_loading(),
    }


@app.post("/api/models/select")
async def select_model(payload: SelectModelRequest, background_tasks: BackgroundTasks):
    global MANUAL_MODEL_OVERRIDE

    # Hard-block stacking a second switch on top of an in-flight one.
    if is_loading():
        raise HTTPException(
            status_code=409,
            detail="A model is already loading. Wait for it to finish before switching.",
        )

    models = discover_models(CONFIG["model_dir"])
    match = next((m for m in models if m["filename"] == payload.filename), None)
    if not match:
        raise HTTPException(status_code=404, detail="Model not found in the models directory.")

    capacity = calculate_cluster_capacity(NODE_REGISTRY)
    verified = is_model_verified(match)

    if (not verified) and match["min_ram_gb"] > capacity["total_usable_gb"] and not payload.force:
        raise HTTPException(
            status_code=409,
            detail=(f"'{match['name']}' needs ~{match['min_ram_gb']}GB usable RAM, but the cluster only "
                    f"has ~{capacity['total_usable_gb']}GB usable right now. Retry with force=true to attempt anyway.")
        )

    # Atomically claim the loader. Returns None if another load slipped in
    # between the is_loading() check above and here.
    gen = begin_manual_switch(match)
    if gen is None:
        raise HTTPException(status_code=409, detail="A model load is already in progress.")

    MANUAL_MODEL_OVERRIDE = payload.filename
    print(f"[CLUSTER] Manual model switch requested -> {match['name']} ({match['filename']})")

    background_tasks.add_task(
        start_or_restart_server,
        NODE_REGISTRY,
        CONFIG["llama_dir"],
        CONFIG["model_dir"],
        payload.filename,
        reason="switch",
        message=f"Switching to {match['name']}",
    )

    return {"status": "switching", "model": match}


@app.get("/status")
async def get_cluster_status():
    try:
        res = await async_client.get("/health", timeout=0.5)
        if res.status_code == 200:
            return {"status": "ready"}
    except Exception as e:
        print(f"[STATUS] Llama-server not ready yet: {e}")
    return {"status": "loading"}


async def clean_dead_nodes():
    while True:
        await asyncio.sleep(5)
        current_time = time.time()
        registry_changed = False

        for node_id in list(NODE_REGISTRY.keys()):
            info = NODE_REGISTRY[node_id]
            last_seen = float(info["last_seen"])

            if current_time - last_seen > HEARTBEAT_TIMEOUT:
                print(f"[REMOVED] Node {node_id} ({info['system_details']['hostname']}) missed heartbeat timeout.")
                if node_id in NODE_REGISTRY:
                    del NODE_REGISTRY[node_id]
                    registry_changed = True
                    evaluate_cluster_level_change(event_type="leave", node_id=node_id)

        if registry_changed:
            print("[INFO] Re-configuring cluster layout...")
            chosen = resolve_startup_model(
                CONFIG["model_dir"],
                MANUAL_MODEL_OVERRIDE or server_module.CURRENT_MODEL_FILENAME,
            )
            if chosen is not None:
                msg = "Node left — reloading cluster with new topology"
                gen = begin_topology_reload(chosen, message=msg)
                if gen is not None:
                    start_or_restart_server(
                        NODE_REGISTRY,
                        CONFIG["llama_dir"],
                        CONFIG["model_dir"],
                        chosen["filename"],
                        reason="topology",
                        message=msg,
                    )


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(clean_dead_nodes())
    print("[INIT] Dead node cleanup task started.")

    # Cold start: bring llama-server up immediately using either the last
    # successfully running model (persisted to disk, survives tab close /
    # server restart) or, if none is on record, the smallest model on disk —
    # regardless of how much pooled RAM is available.
    chosen = resolve_startup_model(CONFIG["model_dir"], None)
    if chosen is not None:
        msg = f"Starting cluster with {chosen['name']}"
        gen = begin_topology_reload(chosen, message=msg)
        if gen is not None:
            asyncio.create_task(asyncio.to_thread(
                start_or_restart_server,
                NODE_REGISTRY,
                CONFIG["llama_dir"],
                CONFIG["model_dir"],
                chosen["filename"],
                reason="startup",
                message=msg,
            ))
    else:
        print("[INIT] No models on disk — llama-server not started.")


@app.on_event("shutdown")
def shutdown_event():
    if server_module.CURRENT_SERVER_PROCESS is not None:
        print("[SHUTDOWN] Terminating active cluster processes...")
        try:
            server_module.CURRENT_SERVER_PROCESS.terminate()
        except Exception:
            pass


@app.get("/alive")
async def alive(node_id: str, available_ram_gb: Optional[float] = None):
    if node_id in NODE_REGISTRY:
        NODE_REGISTRY[node_id]["last_seen"] = time.time()
        if available_ram_gb is not None:
            NODE_REGISTRY[node_id]["system_details"]["available_ram_gb"] = available_ram_gb
        return {"status": "ok"}
    else:
        return {"status": "error", "message": "Node not registered"}


@app.post("/connect")
async def connect_post(payload: ConnectRequest, background_tasks: BackgroundTasks):
    NODE_REGISTRY[payload.node_id] = {
        "ip_address": payload.ip_address,
        "port": payload.port,
        "last_seen": time.time(),
        "system_details": payload.system_details.model_dump(),
    }

    evaluate_cluster_level_change(event_type="join", node_id=payload.node_id)

    chosen = resolve_startup_model(
        CONFIG["model_dir"],
        MANUAL_MODEL_OVERRIDE or server_module.CURRENT_MODEL_FILENAME,
    )
    if chosen is not None:
        msg = f"Node {payload.node_id} joined — reloading cluster"
        gen = begin_topology_reload(chosen, message=msg)
        if gen is not None:
            background_tasks.add_task(
                start_or_restart_server,
                NODE_REGISTRY,
                CONFIG["llama_dir"],
                CONFIG["model_dir"],
                chosen["filename"],
                reason="topology",
                message=msg,
            )

    return {
        "status": "Success",
        "connected_node": payload.node_id,
        "message": f"Node {payload.node_id} integrated successfully.",
    }


@app.post("/api/chat")
async def chat_proxy(payload: dict):
    try:
        req = async_client.build_request("POST", "/v1/chat/completions", json=payload)
        response = await async_client.send(req, stream=True)
        if response.status_code != 200:
            await response.aread()
            raise HTTPException(status_code=response.status_code, detail="Cluster layer initializing or down.")
        return StreamingResponse(response.aiter_raw(), status_code=response.status_code, headers=dict(response.headers))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Cluster communication issue: {e}")


@app.get("/nodes")
async def get_active_nodes():
    return {"active_nodes": NODE_REGISTRY}


_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"[INIT] No frontend folder at {_FRONTEND_DIR} — API-only mode. "
          f"Build the Next.js app and copy out/ to frontend/.")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="AI Party Cluster API Host")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run API server")
    parser.add_argument("--llama-dir", type=str, default=None, help="Custom folder containing llama.cpp executables")
    parser.add_argument("--model-dir", type=str, default=None, help="Custom folder containing GGUF model files")

    args = parser.parse_args()
    if args.llama_dir:
        CONFIG["llama_dir"] = args.llama_dir
    if args.model_dir:
        CONFIG["model_dir"] = args.model_dir

    uvicorn.run(app, host=args.host, port=args.port)
