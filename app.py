import os
import sys
import json
import shutil
import signal
import subprocess
import threading
import queue
import platform
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# ------------- constants -------------
APP_NAME = "AI Party · Client"
CONFIG_PATH = Path.home() / ".aiparty_client_gui.json"
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

# Where to look for client.py, in order of preference:
#   1. Next to this file (running from source)
#   2. Inside the PyInstaller bundle (sys._MEIPASS)
#   3. In the current working directory
def locate_client_script() -> Path | None:
    candidates = []
    here = Path(__file__).resolve().parent
    candidates.append(here / "client.py")
    if getattr(sys, "_MEIPASS", None):
        candidates.append(Path(sys._MEIPASS) / "client.py")
    candidates.append(Path.cwd() / "client.py")
    for c in candidates:
        if c.exists():
            return c
    return None


# ------------- config persistence -------------

DEFAULTS = {
    "api_url": "http://127.0.0.1:8000",
    "node_ip": "",
    "port": "50052",
    "host": "0.0.0.0",
    "interval": "3",
    "llama_dir": "",
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                if k in cfg and isinstance(v, str):
                    cfg[k] = v
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return cfg


def save_config(cfg: dict) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ------------- main app -------------

class ClientGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_NAME)
        self.geometry("960x720")
        self.minsize(760, 560)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.cfg = load_config()
        self.proc: subprocess.Popen | None = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.client_script = locate_client_script()

        self._build_layout()
        self._bind_events()
        self._poll_log_queue()

        if self.client_script is None:
            self._append_log(
                "[GUI] client.py not found next to this app. "
                "Put client.py in the same folder before pressing Connect.\n"
            )

    # ---- layout ----

    def _build_layout(self):
        # Two-column layout: form on the left, log on the right.
        self.grid_columnconfigure(0, weight=0, minsize=380)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -------- left: form --------
        form_frame = ctk.CTkFrame(self, corner_radius=0)
        form_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        form_frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            form_frame, text="AI Party — Client Node",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        header.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 4))

        subheader = ctk.CTkLabel(
            form_frame,
            text="Join a cluster as a worker node",
            text_color=("#888", "#888"),
            font=ctk.CTkFont(size=12),
        )
        subheader.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 20))

        row = 2

        # API URL
        self._section_label(form_frame, row, "API host URL")
        row += 1
        self.entry_api = ctk.CTkEntry(form_frame, placeholder_text="http://192.168.1.6:8000")
        self.entry_api.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 14))
        self._set_entry(self.entry_api, self.cfg["api_url"])
        row += 1

        # Node IP
        self._section_label(form_frame, row, "This machine's IP (as seen by the API host)")
        row += 1
        ip_row = ctk.CTkFrame(form_frame, fg_color="transparent")
        ip_row.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 14))
        ip_row.grid_columnconfigure(0, weight=1)
        self.entry_node_ip = ctk.CTkEntry(ip_row, placeholder_text="192.168.1.14")
        self.entry_node_ip.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._set_entry(self.entry_node_ip, self.cfg["node_ip"])
        ctk.CTkButton(
            ip_row, text="Detect", width=72, command=self._detect_local_ip
        ).grid(row=0, column=1)
        row += 1

        # RPC port + bind host (side by side)
        self._section_label(form_frame, row, "RPC port and bind host")
        row += 1
        port_row = ctk.CTkFrame(form_frame, fg_color="transparent")
        port_row.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 14))
        port_row.grid_columnconfigure(0, weight=1)
        port_row.grid_columnconfigure(1, weight=1)
        self.entry_port = ctk.CTkEntry(port_row, placeholder_text="50052")
        self.entry_port.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._set_entry(self.entry_port, self.cfg["port"])
        self.entry_host = ctk.CTkEntry(port_row, placeholder_text="0.0.0.0")
        self.entry_host.grid(row=0, column=1, sticky="ew")
        self._set_entry(self.entry_host, self.cfg["host"])
        row += 1

        # Heartbeat interval
        self._section_label(form_frame, row, "Heartbeat interval (seconds)")
        row += 1
        self.entry_interval = ctk.CTkEntry(form_frame, placeholder_text="3")
        self.entry_interval.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 14))
        self._set_entry(self.entry_interval, self.cfg["interval"])
        row += 1

        # llama-dir picker
        self._section_label(form_frame, row, "llama.cpp binary folder (ggml-rpc-server)")
        row += 1
        dir_row = ctk.CTkFrame(form_frame, fg_color="transparent")
        dir_row.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 6))
        dir_row.grid_columnconfigure(0, weight=1)
        self.entry_llama_dir = ctk.CTkEntry(dir_row, placeholder_text="auto-detect")
        self.entry_llama_dir.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._set_entry(self.entry_llama_dir, self.cfg["llama_dir"])
        ctk.CTkButton(
            dir_row, text="Browse", width=72, command=self._pick_llama_dir
        ).grid(row=0, column=1)
        row += 1

        hint = ctk.CTkLabel(
            form_frame,
            text=("If left empty, client.py searches PATH, LLAMA_DIR, and\n"
                  "common project-relative locations."),
            text_color=("#888", "#888"),
            font=ctk.CTkFont(size=11),
            justify="left",
        )
        hint.grid(row=row, column=0, sticky="w", padx=20, pady=(0, 20))
        row += 1

        # spacer so buttons sit at the bottom
        form_frame.grid_rowconfigure(row, weight=1)
        row += 1

        # Buttons
        btn_row = ctk.CTkFrame(form_frame, fg_color="transparent")
        btn_row.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 20))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=0)
        self.btn_connect = ctk.CTkButton(
            btn_row, text="Connect", height=40,
            fg_color="#e8a33d", hover_color="#f0b452", text_color="#1a1305",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_connect_clicked,
        )
        self.btn_connect.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.btn_clear = ctk.CTkButton(
            btn_row, text="Clear log", height=40, width=110,
            fg_color="#2a3240", hover_color="#3a4356",
            command=self._clear_log,
        )
        self.btn_clear.grid(row=0, column=1)

        # -------- right: log --------
        log_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#10141b")
        log_frame.grid(row=0, column=1, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        log_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            log_header, text="Live log",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        self.status_label = ctk.CTkLabel(
            log_header, text="idle", text_color="#7c8798",
            font=ctk.CTkFont(size=12),
        )
        self.status_label.grid(row=0, column=1, sticky="e")

        self.log_box = ctk.CTkTextbox(
            log_frame, wrap="word", font=ctk.CTkFont(family="Consolas" if IS_WINDOWS else "Menlo", size=12),
            fg_color="#0c0f14", text_color="#d0d6df",
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.log_box.configure(state="disabled")

    def _section_label(self, parent, row, text):
        ctk.CTkLabel(
            parent, text=text, text_color="#9aa4b2",
            font=ctk.CTkFont(size=11, weight="bold"),
        ).grid(row=row, column=0, sticky="w", padx=20, pady=(0, 4))

    def _set_entry(self, entry: ctk.CTkEntry, value: str) -> None:
        if value:
            entry.delete(0, "end")
            entry.insert(0, value)

    # ---- events / helpers ----

    def _bind_events(self):
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        for entry in (
            self.entry_api, self.entry_node_ip, self.entry_port,
            self.entry_host, self.entry_interval, self.entry_llama_dir,
        ):
            entry.bind("<KeyRelease>", lambda _e: self._persist_form())

    def _persist_form(self):
        self.cfg = {
            "api_url": self.entry_api.get().strip(),
            "node_ip": self.entry_node_ip.get().strip(),
            "port": self.entry_port.get().strip(),
            "host": self.entry_host.get().strip(),
            "interval": self.entry_interval.get().strip(),
            "llama_dir": self.entry_llama_dir.get().strip(),
        }
        save_config(self.cfg)

    def _pick_llama_dir(self):
        initial = self.entry_llama_dir.get().strip() or str(Path.home())
        chosen = filedialog.askdirectory(
            title="Select folder containing ggml-rpc-server",
            initialdir=initial,
        )
        if chosen:
            self.entry_llama_dir.delete(0, "end")
            self.entry_llama_dir.insert(0, chosen)
            self._persist_form()

    def _detect_local_ip(self):
        """
        Best-effort primary LAN IP discovery. Uses the UDP connect trick:
        open a socket to a public address (no packets actually sent) and
        read the local end of the socket. Returns '' on failure.
        """
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            finally:
                s.close()
            self.entry_node_ip.delete(0, "end")
            self.entry_node_ip.insert(0, ip)
            self._persist_form()
        except Exception as e:
            messagebox.showwarning(
                "Couldn't detect IP",
                f"Automatic detection failed ({e}). Please enter your IP manually."
            )

    def _append_log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ---- connection lifecycle ----

    def _on_connect_clicked(self):
        if self.proc is not None:
            self._disconnect()
            return
        self._connect()

    def _connect(self):
        if self.client_script is None:
            messagebox.showerror(
                "Missing client.py",
                "client.py was not found next to this application.\n\n"
                "Place client.py in the same folder as the GUI and try again."
            )
            return

        api_url = self.entry_api.get().strip()
        if not api_url:
            messagebox.showerror("Missing API URL", "Enter the API host URL (e.g. http://192.168.1.6:8000).")
            return

        node_ip = self.entry_node_ip.get().strip()
        if not node_ip:
            messagebox.showerror(
                "Missing node IP",
                "Enter this machine's IP as seen by the API host.\n"
                "Use Detect to auto-fill if you're not sure."
            )
            return

        args = [sys.executable, "-u", str(self.client_script)]
        args += ["--api-url", api_url]
        args += ["--node-ip", node_ip]

        port = self.entry_port.get().strip()
        if port:
            args += ["--port", port]
        host = self.entry_host.get().strip()
        if host:
            args += ["--host", host]
        interval = self.entry_interval.get().strip()
        if interval:
            args += ["--interval", interval]
        llama_dir = self.entry_llama_dir.get().strip()
        if llama_dir:
            args += ["--llama-dir", llama_dir]

        self._append_log(f"[GUI] Launching: {' '.join(args)}\n")
        self._append_log("─" * 60 + "\n")

        # On Windows, avoid popping a console window.
        creationflags = 0
        if IS_WINDOWS:
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

        # On macOS, ensure we can terminate the whole process group.
        preexec_fn = None
        if not IS_WINDOWS:
            preexec_fn = os.setsid

        try:
            self.proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.client_script.parent),
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
        except Exception as e:
            self._append_log(f"[GUI] Failed to launch client: {e}\n")
            self.proc = None
            return

        self.btn_connect.configure(
            text="Disconnect",
            fg_color="#f2685c",
            hover_color="#ff7a6f",
        )
        self.status_label.configure(text="connected", text_color="#47d18c")

        t = threading.Thread(target=self._reader_thread, daemon=True)
        t.start()

    def _reader_thread(self):
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self.log_queue.put(line)
        except Exception:
            pass
        finally:
            rc = proc.wait()
            self.log_queue.put(f"\n[GUI] client.py exited (code {rc}).\n")
            self.log_queue.put("__EXIT__")

    def _disconnect(self):
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        self._append_log("[GUI] Sending terminate signal...\n")
        try:
            if IS_WINDOWS:
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "__EXIT__":
                    # process finished — reset button state
                    self.btn_connect.configure(
                        text="Connect",
                        fg_color="#e8a33d",
                        hover_color="#f0b452",
                    )
                    self.status_label.configure(text="idle", text_color="#7c8798")
                    continue
                self._append_log(line)
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askokcancel(
                "Disconnect",
                "A client is running. Disconnect and quit?"
            ):
                return
            self._disconnect()
        self.destroy()


def main():
    app = ClientGUI()
    app.mainloop()


if __name__ == "__main__":
    main()