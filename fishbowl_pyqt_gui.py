#!/usr/bin/env python3
"""
LocalSleep-MoE SOMA-Fishbowl Independent PyQt6 GUI + Network Control Server (:18940)
====================================================================================
Features requested by Architect Leon:
1. Independent PyQt6 Tactical Aquarium Interface (`fishbowl_pyqt_gui.py`):
   - Live Framebuffer Screen Monitor (`640x360 RGBA8888 /dev/fb0` auto-refreshed at 15 FPS via QImage/QPixmap).
   - Live Console Capture Dock (`stdout`, `stderr`, compiler output, Linux `/proc` vitals).
   - Source Code Editor / Inspector & Negative-Palindrome Immune Memory Viewer.
   - One-click Controls:
     * [🧠 AI 鍛造並投放器官 (AI Forge via :18935)]
     * [🔨 編譯並投放當前源碼 (Compile & Drop)]
     * [📸 抓圖 + 抓 Console 快照 (Capture Snapshot)]
     * [♻️ 魚缸換水回滾 (Reset to LKG)]
2. Built-in Network Control & Capture HTTP Server (Default Port `:18940`, Dual-Mode with PyQt6 or `--headless`):
   - `GET  /`                -> Web Mirror of the PyQt6 Fishbowl Observatory
   - `GET  /api/status`      -> JSON status + Linux `/proc` vitals + Negative-Palindrome DB
   - `GET  /api/console`     -> JSON captured console logs (`stdout`, `stderr`, history)
   - `GET  /api/screen.png`  -> Raw PNG image of the current Fishbowl `/dev/fb0` Framebuffer screen!
   - `POST /api/drop_source` -> Submit `{"name": "...", "lang": "rust|c", "code": "..."}` -> compiles, runs in Fishbowl, returns console + screen stats!
   - `POST /api/ai_forge`    -> Submit `{"name": "...", "lang": "rust|c", "task": "..."}` -> asks `:18935` to write code, compiles, drops into Fishbowl, returns console + screen!
   - `POST /api/reset`       -> Resets Fishbowl Framebuffer to LKG clean state.
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from localsleep_fishbowl import (
    DEFAULT_ENDPOINT,
    FB_BYTES,
    FB_FILE,
    FB_HEIGHT,
    FB_WIDTH,
    HUD_HTML_OUT,
    ORGAN_BAY_DIR,
    PALINDROME_DB,
    SNAPSHOT_DIR,
    SRC_BAY_DIR,
    TELEMETRY_LOG,
    ai_forge_and_drop,
    compile_source_into_organ,
    create_and_compile_compositor_v3,
    create_builtin_seed_organ,
    ensure_fishbowl_initialized,
    get_all_devmem_channels,
    get_ui_config,
    is_os_daemon_alive,
    list_os_apps_and_processes,
    poke_devmem_channel,
    render_aquarium_hud_html,
    rgba_to_png_b64,
    run_organ_in_fishbowl,
    run_os_bash_command,
    stop_active_os_daemon,
    update_ui_config,
)


class FishbowlController:
    """Thread-safe controller shared by the PyQt6 GUI and the :18940 Network Control Server."""

    def __init__(self, llm_endpoint: str = DEFAULT_ENDPOINT) -> None:
        self.lock = threading.Lock()
        self.llm_endpoint = llm_endpoint
        self.env_info = ensure_fishbowl_initialized()
        self.console_lines: list[str] = []
        self.latest_obs: Dict[str, Any] = {}
        self._bootstrap_seed_if_needed()

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.console_lines.append(line)
        if len(self.console_lines) > 500:
            del self.console_lines[:-500]
        print(line, flush=True)

    def _bootstrap_seed_if_needed(self) -> None:
        seed_bin = ORGAN_BAY_DIR / "soma_seed_pulse"
        if not seed_bin.exists():
            seed_rs = create_builtin_seed_organ()
            ok, bin_p, msg = compile_source_into_organ(seed_rs, "soma_seed_pulse")
            self._log(f"Seed Organ Init: {msg}")
            if ok and bin_p:
                self.latest_obs = run_organ_in_fishbowl(bin_p, "soma_seed_pulse")
                self._log(
                    f"Seed Organ ran in {self.latest_obs.get('execution_ms')}ms | "
                    f"FB coverage={self.latest_obs.get('fb_coverage_pct')}%"
                )
        else:
            if TELEMETRY_LOG.exists():
                lines = [ln for ln in TELEMETRY_LOG.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if lines:
                    try:
                        self.latest_obs = json.loads(lines[-1])
                    except Exception:
                        pass

    def get_fb_bytes(self) -> bytes:
        if FB_FILE.exists():
            data = FB_FILE.read_bytes()
            if len(data) == FB_BYTES:
                return data
        return bytes(FB_BYTES)

    def get_screen_png_bytes(self) -> bytes:
        b64 = rgba_to_png_b64(self.get_fb_bytes(), FB_WIDTH, FB_HEIGHT)
        return base64.b64decode(b64)

    def get_status_payload(self) -> Dict[str, Any]:
        pal_db = json.loads(PALINDROME_DB.read_text(encoding="utf-8")) if PALINDROME_DB.exists() else {}
        organs = sorted([p.name for p in ORGAN_BAY_DIR.iterdir() if p.is_file()]) if ORGAN_BAY_DIR.exists() else []
        sources = sorted([p.name for p in SRC_BAY_DIR.iterdir() if p.is_file()]) if SRC_BAY_DIR.exists() else []
        return {
            "ok": True,
            "fishbowl": self.env_info,
            "latest_observation": self.latest_obs,
            "organs_in_bay": organs,
            "sources_in_bay": sources,
            "negative_palindrome_db": pal_db,
            "console_tail": self.console_lines[-60:],
        }

    def drop_source_code(
        self,
        name: str,
        code: str,
        lang: str = "rust",
        timeout_sec: float = 5.0,
        daemonize: bool = False,
    ) -> Dict[str, Any]:
        with self.lock:
            clean_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name).strip("_") or "organ"
            ext = ".rs" if lang.lower() in ("rust", "rs") else ".c"
            src_path = SRC_BAY_DIR / f"{clean_name}{ext}"
            src_path.write_text(code, encoding="utf-8")
            self._log(f"🔨 Compiling `{src_path.name}` (daemon={daemonize})...")
            ok, bin_path, compile_msg = compile_source_into_organ(src_path, clean_name)
            self._log(compile_msg)
            if not ok or not bin_path:
                return {
                    "ok": False,
                    "stage": "compile",
                    "organ_name": clean_name,
                    "compiler_output": compile_msg,
                    "console_tail": self.console_lines[-40:],
                }

            self._log(
                f"🐟 Dropping `{clean_name}` into Linux Fishbowl "
                f"({'DAEMON MODE' if daemonize else f'timeout={timeout_sec}s'})..."
            )
            obs = run_organ_in_fishbowl(bin_path, clean_name, timeout_sec=timeout_sec, daemonize=daemonize)
            self.latest_obs = obs
            if obs.get("stdout"):
                for ln in obs["stdout"].splitlines()[-20:]:
                    self._log(f"[STDOUT] {ln}")
            if obs.get("stderr"):
                for ln in obs["stderr"].splitlines()[-20:]:
                    self._log(f"[STDERR] {ln}")
            self._log(
                f"✅ `{clean_name}` -> status={obs['status']} | PID={obs.get('daemon_pid')} | "
                f"anim_diff={obs.get('anim_diff_pixels',0)}px | FB={obs['fb_coverage_pct']}%"
            )
            return {
                "ok": obs["status"] in ("HEALTHY", "DAEMON_RUNNING"),
                "stage": "run",
                "observation": obs,
                "console_tail": self.console_lines[-40:],
            }

    def run_ai_forge(
        self,
        name: str,
        task: str,
        lang: str = "rust",
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            ep = endpoint or self.llm_endpoint
            self._log(f"🧠 AI-Forge requested: organ=`{name}` ({lang}) via {ep}")
            self._log(f"   Task: {task[:120]}...")
            try:
                obs = ai_forge_and_drop(
                    organ_name=name,
                    task_prompt=task,
                    lang=lang,
                    endpoint=ep,
                )
                self.latest_obs = obs
                if obs.get("stdout"):
                    for ln in obs["stdout"].splitlines():
                        self._log(f"[STDOUT] {ln}")
                self._log(
                    f"🎉 AI-Forge `{name}` complete: status={obs['status']} | "
                    f"FB={obs['fb_coverage_pct']}%"
                )
                return {"ok": True, "observation": obs, "console_tail": self.console_lines[-40:]}
            except Exception as e:
                self._log(f"❌ AI-Forge failed: {e}")
                return {"ok": False, "error": str(e), "console_tail": self.console_lines[-40:]}

    def reset_to_lkg(self) -> Dict[str, Any]:
        with self.lock:
            lkg = SNAPSHOT_DIR / "fb0_lkg.raw"
            if lkg.exists():
                FB_FILE.write_bytes(lkg.read_bytes())
            self._log("♻️ Fishbowl Framebuffer & state restored to LKG snapshot.")
            render_aquarium_hud_html(self.latest_obs, self.get_fb_bytes())
            return {"ok": True, "message": "Restored to LKG snapshot"}


def start_network_server(controller: FishbowlController, host: str, port: int) -> ThreadingHTTPServer:
    """Starts the background HTTP control server for network control + console/screen capture."""

    class FishbowlHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # Keep console clean

        def _send_json(self, obj: Dict[str, Any], code: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                render_aquarium_hud_html(controller.latest_obs, controller.get_fb_bytes())
                raw = HUD_HTML_OUT.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path.startswith("/api/status"):
                self._send_json(controller.get_status_payload())
            elif self.path.startswith("/api/console"):
                self._send_json(
                    {
                        "ok": True,
                        "console_lines": controller.console_lines[-150:],
                        "latest_stdout": controller.latest_obs.get("stdout", ""),
                        "latest_stderr": controller.latest_obs.get("stderr", ""),
                    }
                )
            elif self.path.startswith("/api/screen.png"):
                png_bytes = controller.get_screen_png_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png_bytes)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(png_bytes)
            elif self.path.startswith("/api/screen_b64"):
                b64 = rgba_to_png_b64(controller.get_fb_bytes(), FB_WIDTH, FB_HEIGHT)
                self._send_json(
                    {
                        "ok": True,
                        "width": FB_WIDTH,
                        "height": FB_HEIGHT,
                        "png_base64": b64,
                        "latest_observation": controller.latest_obs,
                    }
                )
            elif self.path.startswith("/api/ui_config"):
                self._send_json({"ok": True, "ui_config": get_ui_config()})
            elif self.path.startswith("/api/devmem"):
                self._send_json({"ok": True, "channels": get_all_devmem_channels()})
            elif self.path.startswith("/api/os/apps"):
                self._send_json(list_os_apps_and_processes())
            else:
                self._send_json({"ok": False, "error": f"Unknown endpoint: {self.path}"}, 404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                body = json.loads(raw_body) if raw_body.strip() else {}
            except Exception as e:
                self._send_json({"ok": False, "error": f"Invalid JSON: {e}"}, 400)
                return

            if self.path.startswith("/api/drop_source"):
                name = str(body.get("name") or "net_organ")
                code = str(body.get("code") or "")
                lang = str(body.get("lang") or "rust")
                timeout_sec = float(body.get("timeout") or 5.0)
                daemonize = bool(body.get("daemonize", False))
                if not code.strip():
                    self._send_json({"ok": False, "error": "Missing `code` field"}, 400)
                    return
                res = controller.drop_source_code(
                    name=name, code=code, lang=lang, timeout_sec=timeout_sec, daemonize=daemonize
                )
                self._send_json(res, 200 if res.get("ok") else 422)

            elif self.path.startswith("/api/stop_daemon"):
                killed_pid = stop_active_os_daemon()
                controller._log(f"🛑 Stopped active OS daemon (PID={killed_pid})")
                self._send_json({"ok": True, "stopped_pid": killed_pid})

            elif self.path.startswith("/api/start_compositor_v3"):
                ok_c, bin_p, c_msg = create_and_compile_compositor_v3()
                controller._log(f"🔨 Compositor v3: {c_msg}")
                if not ok_c or not bin_p:
                    self._send_json({"ok": False, "error": c_msg}, 500)
                    return
                obs = run_organ_in_fishbowl(bin_p, "soma_compositor_v3", daemonize=True)
                controller.latest_obs = obs
                controller._log(f"🟢 SOMA-OS Compositor v3 running (PID={obs.get('daemon_pid')})")
                self._send_json({"ok": True, "observation": obs})

            elif self.path.startswith("/api/ui_config"):
                updated = update_ui_config(body)
                controller._log(f"⚡ Hot-reloaded UI Config: {json.dumps(body, ensure_ascii=False)}")
                self._send_json({"ok": True, "ui_config": updated})

            elif self.path.startswith("/api/devmem"):
                ch = int(body.get("channel", 4))
                reg_val = body.get("reg_u64")
                if isinstance(reg_val, str):
                    reg_val = int(reg_val, 16) if reg_val.startswith(("0x", "0X")) else int(reg_val)
                tag = body.get("tag")
                payload = body.get("payload")
                ch_info = poke_devmem_channel(ch, reg_u64=reg_val, tag=tag, payload=payload)
                controller._log(f"⚡ DevMem Poke CH{ch} -> SEQ={ch_info['seq']} REG={ch_info['reg_hex']} TAG={ch_info['tag']}")
                self._send_json({"ok": True, "channel": ch_info})

            elif self.path.startswith("/api/os/bash"):
                cmd_str = str(body.get("cmd") or body.get("command") or "soma_devmem status")
                timeout_s = float(body.get("timeout", 20.0))
                res = run_os_bash_command(cmd_str, timeout_sec=timeout_s)
                controller._log(f"💻 [OS-BASH] $ {cmd_str} -> rc={res['return_code']} ({res['execution_ms']}ms)")
                if res.get("stdout"):
                    for ln in res["stdout"].splitlines()[-10:]:
                        controller._log(f"   {ln}")
                self._send_json(res)

            elif self.path.startswith("/api/ai_forge"):
                name = str(body.get("name") or f"ai_organ_{int(time.time()) % 10000}")
                task = str(
                    body.get("task")
                    or "Read /proc/loadavg and /proc/meminfo, draw a cybernetic 16-expert neural matrix and waveform on the 640x360 RGBA framebuffer, and print system stats."
                )
                lang = str(body.get("lang") or "rust")
                endpoint = body.get("endpoint")
                res = controller.run_ai_forge(name=name, task=task, lang=lang, endpoint=endpoint)
                self._send_json(res, 200 if res.get("ok") else 500)

            elif self.path.startswith("/api/palindrome_step"):
                from soma_os_palindrome_loop import LOOP_HISTORY_FILE, run_palindrome_iteration
                iters = int(body.get("iterations") or 1)
                history = []
                if LOOP_HISTORY_FILE.exists():
                    try:
                        history = json.loads(LOOP_HISTORY_FILE.read_text(encoding="utf-8"))
                    except Exception:
                        history = []
                results = []
                start_i = len(history) + 1
                for idx in range(start_i, start_i + iters):
                    rec = run_palindrome_iteration(idx, history, endpoint=controller.llm_endpoint)
                    results.append(rec)
                self._send_json({"ok": True, "completed_steps": results})

            elif self.path.startswith("/api/reset"):
                self._send_json(controller.reset_to_lkg())

            else:
                self._send_json({"ok": False, "error": f"Unknown POST endpoint: {self.path}"}, 404)

    httpd = ThreadingHTTPServer((host, port), FishbowlHTTPHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    controller._log(f"🌐 Fishbowl Network Control & Capture API listening on http://{host}:{port}")
    return httpd


def run_pyqt_gui(controller: FishbowlController, port: int) -> int:
    """Launches the independent PyQt6 Tactical Fishbowl Window."""
    from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt6.QtGui import QFont, QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    class AsyncWorker(QThread):
        finished_signal = pyqtSignal(dict)

        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self.fn = fn
            self.args = args
            self.kwargs = kwargs

        def run(self):
            res = self.fn(*self.args, **self.kwargs)
            self.finished_signal.emit(res)

    class FishbowlMainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(
                f"🐟 SOMA-Fishbowl 數位魚缸觀測艙 (PyQt6 + Network API :{port}) — LocalSleep-MoE Linux Organ Sandbox"
            )
            self.resize(1380, 860)
            self.setStyleSheet(
                """
                QMainWindow, QWidget { background-color: #060911; color: #e2e8f0; font-family: 'JetBrains Mono', 'Inter', monospace; }
                QLabel { color: #e2e8f0; }
                QPlainTextEdit, QLineEdit, QComboBox {
                    background-color: #0b1120; color: #38bdf8; border: 1px solid #1e293b;
                    border-radius: 6px; padding: 6px; font-size: 12px;
                }
                QPushButton {
                    background-color: #0f172a; color: #38bdf8; border: 1px solid #38bdf8;
                    border-radius: 6px; padding: 7px 12px; font-weight: bold; font-size: 12px;
                }
                QPushButton:hover { background-color: #1e293b; }
                """
            )
            self.worker: Optional[AsyncWorker] = None

            central = QWidget()
            self.setCentralWidget(central)
            root_layout = QVBoxLayout(central)

            # Top Header Bar
            hdr = QHBoxLayout()
            title_lbl = QLabel(
                f"🐟 SOMA-Fishbowl 數位魚缸 ｜ 網路控制端點: http://127.0.0.1:{port} "
                f"(GET /api/screen.png | GET /api/console | POST /api/drop_source | POST /api/ai_forge)"
            )
            title_lbl.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
            self.status_lbl = QLabel("🟢 HEALTHY")
            self.status_lbl.setStyleSheet("color: #34d399; font-weight: bold; font-size: 13px;")
            hdr.addWidget(title_lbl)
            hdr.addStretch()
            hdr.addWidget(self.status_lbl)
            root_layout.addLayout(hdr)

            # Main Splitter (Left: Screen + Console | Right: AI Forge + Source Editor + Palindrome DB)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            root_layout.addWidget(splitter, stretch=1)

            # Left Pane: Framebuffer Screen + Console Capture
            left_w = QWidget()
            left_l = QVBoxLayout(left_w)

            scr_hdr = QLabel(f"👁️ 魚缸玻璃：Linux /dev/fb0 AI-Native Framebuffer ({FB_WIDTH}×{FB_HEIGHT} RGBA8888)")
            scr_hdr.setStyleSheet("color: #38bdf8; font-weight: bold;")
            left_l.addWidget(scr_hdr)

            self.screen_lbl = QLabel()
            self.screen_lbl.setMinimumSize(640, 480)
            self.screen_lbl.setMaximumSize(1024, 768)
            self.screen_lbl.setStyleSheet("border: 2px solid #38bdf8; background-color: #020408; border-radius: 6px;")
            self.screen_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            left_l.addWidget(self.screen_lbl, alignment=Qt.AlignmentFlag.AlignHCenter)

            self.metrics_lbl = QLabel("Organ: - | Latency: - ms | FB Pixels: - | LoadAvg: -")
            self.metrics_lbl.setStyleSheet("color: #fbbf24; font-size: 12px; padding: 4px;")
            left_l.addWidget(self.metrics_lbl)

            con_hdr = QLabel("📟 魚缸 Console 捕獲艙 (Stdout / Stderr / Compiler / Watchdog Log)")
            con_hdr.setStyleSheet("color: #34d399; font-weight: bold;")
            left_l.addWidget(con_hdr)

            self.console_box = QPlainTextEdit()
            self.console_box.setReadOnly(True)
            self.console_box.setFont(QFont("Monospace", 10))
            left_l.addWidget(self.console_box, stretch=1)

            splitter.addWidget(left_w)

            # Right Pane: AI Task Bar + Source Editor + Negative-Palindrome Immune Rules
            right_w = QWidget()
            right_l = QVBoxLayout(right_w)

            ai_hdr = QLabel("🧠 投餵口：呼叫地端 :18935 AI 編譯投放 或 手動修改 Rust/C 器官源碼")
            ai_hdr.setStyleSheet("color: #38bdf8; font-weight: bold;")
            right_l.addWidget(ai_hdr)

            ctrl_row1 = QHBoxLayout()
            self.name_input = QLineEdit("cyber_pulse_organ")
            self.name_input.setPlaceholderText("器官名稱 (Organ Name)")
            self.lang_combo = QComboBox()
            self.lang_combo.addItems(["rust", "c"])
            ctrl_row1.addWidget(QLabel("名稱:"))
            ctrl_row1.addWidget(self.name_input, stretch=1)
            ctrl_row1.addWidget(QLabel("語言:"))
            ctrl_row1.addWidget(self.lang_combo)
            right_l.addLayout(ctrl_row1)

            self.task_input = QLineEdit(
                "Read /proc/loadavg and /proc/meminfo, draw a 16-expert neural hypercube radar + live CPU/RAM bars into the 640x360 RGBA framebuffer, and print telemetry."
            )
            right_l.addWidget(self.task_input)

            btn_row = QHBoxLayout()
            self.btn_ai = QPushButton("🧠 呼叫 :18935 AI 單次鍛造")
            self.btn_loop = QPushButton("🔄 31B 自我推論迴文建構 OS (步進)")
            self.btn_loop.setStyleSheet("border-color: #34d399; color: #34d399;")
            self.btn_compile = QPushButton("🔨 編譯並投放下方源碼")
            self.btn_reset = QPushButton("♻️ 魚缸換水 (回滾 LKG)")
            self.btn_ResetStyle = "border-color: #f87171; color: #f87171;"
            self.btn_reset.setStyleSheet(self.btn_ResetStyle)

            self.btn_ai.clicked.connect(self.on_ai_forge)
            self.btn_loop.clicked.connect(self.on_palindrome_loop)
            self.btn_compile.clicked.connect(self.on_compile_drop)
            self.btn_reset.clicked.connect(self.on_reset)

            btn_row.addWidget(self.btn_ai)
            btn_row.addWidget(self.btn_loop)
            btn_row.addWidget(self.btn_compile)
            btn_row.addWidget(self.btn_reset)
            right_l.addLayout(btn_row)

            src_hdr = QLabel("🦀 魚缸器官源碼編輯器 (`fishbowl/src_bay/`)")
            src_hdr.setStyleSheet("color: #38bdf8; font-weight: bold;")
            right_l.addWidget(src_hdr)

            self.code_editor = QPlainTextEdit()
            self.code_editor.setFont(QFont("Monospace", 10))
            seed_src = SRC_BAY_DIR / "soma_seed_pulse.rs"
            if seed_src.exists():
                self.code_editor.setPlainText(seed_src.read_text(encoding="utf-8"))
            right_l.addWidget(self.code_editor, stretch=2)

            pal_hdr = QLabel("🪞 負面迴文免疫庫 (`negative_palindrome_db.json` — 自動抑制重複編譯/執行期錯誤)")
            pal_hdr.setStyleSheet("color: #fbbf24; font-weight: bold;")
            right_l.addWidget(pal_hdr)

            self.pal_box = QPlainTextEdit()
            self.pal_box.setReadOnly(True)
            self.pal_box.setMaximumHeight(150)
            right_l.addWidget(self.pal_box)

            splitter.addWidget(right_w)

            # Live Refresh Timer (10 FPS screen + console sync)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh_aquarium_view)
            self.timer.start(120)
            self.last_console_len = -1
            self.refresh_aquarium_view()

        def refresh_aquarium_view(self) -> None:
            # 1. Update Framebuffer QPixmap (scaled smoothly to fit viewport while keeping 1024x768 native bytes)
            fb_data = controller.get_fb_bytes()
            qimg = QImage(fb_data, FB_WIDTH, FB_HEIGHT, FB_WIDTH * 4, QImage.Format.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.screen_lbl.width(),
                self.screen_lbl.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self.screen_lbl.setPixmap(pix)

            # 2. Update Metrics & Console if changed
            obs = controller.latest_obs
            if obs:
                st = obs.get("status", "READY")
                self.status_lbl.setText(f"🟢 {st}" if st == "HEALTHY" else f"🔴 {st}")
                self.metrics_lbl.setText(
                    f"Organ: {obs.get('organ_name','-')} | Time: {obs.get('execution_ms',0)} ms | "
                    f"FB Rendered: {obs.get('fb_coverage_pct',0)}% ({obs.get('fb_changed_pixels',0):,} px) | "
                    f"LoadAvg: {obs.get('linux_loadavg','-')}"
                )

            if len(controller.console_lines) != self.last_console_len:
                self.last_console_len = len(controller.console_lines)
                self.console_box.setPlainText("\n".join(controller.console_lines[-120:]))
                sb = self.console_box.verticalScrollBar()
                if sb:
                    sb.setValue(sb.maximum())

                if PALINDROME_DB.exists():
                    pal = json.loads(PALINDROME_DB.read_text(encoding="utf-8"))
                    rules = pal.get("ActiveNegativeConstraints", [])
                    self.pal_box.setPlainText("\n".join(f"⛔ {r}" for r in rules))

        def on_compile_drop(self) -> None:
            name = self.name_input.text().strip() or "custom_organ"
            lang = self.lang_combo.currentText()
            code = self.code_editor.toPlainText()
            self.btn_compile.setEnabled(False)
            self.worker = AsyncWorker(controller.drop_source_code, name, code, lang)
            self.worker.finished_signal.connect(lambda _: self.btn_compile.setEnabled(True))
            self.worker.start()

        def on_ai_forge(self) -> None:
            name = self.name_input.text().strip() or "ai_organ"
            lang = self.lang_combo.currentText()
            task = self.task_input.text().strip()
            self.btn_ai.setEnabled(False)

            def _done(res: dict) -> None:
                self.btn_ai.setEnabled(True)
                ext = ".rs" if lang == "rust" else ".c"
                src_p = SRC_BAY_DIR / f"{name}{ext}"
                if src_p.exists():
                    self.code_editor.setPlainText(src_p.read_text(encoding="utf-8"))

            self.worker = AsyncWorker(controller.run_ai_forge, name, task, lang)
            self.worker.finished_signal.connect(_done)
            self.worker.start()

        def on_palindrome_loop(self) -> None:
            from soma_os_palindrome_loop import LOOP_HISTORY_FILE, run_palindrome_iteration
            self.btn_loop.setEnabled(False)

            def _run_step() -> dict:
                history = []
                if LOOP_HISTORY_FILE.exists():
                    try:
                        history = json.loads(LOOP_HISTORY_FILE.read_text(encoding="utf-8"))
                    except Exception:
                        history = []
                nxt = len(history) + 1
                rec = run_palindrome_iteration(nxt, history, endpoint=controller.llm_endpoint)
                return {"ok": True, "record": rec}

            def _done(res: dict) -> None:
                self.btn_loop.setEnabled(True)
                rec = res.get("record", {})
                organ = rec.get("subsystem_name", "")
                src_p = SRC_BAY_DIR / f"{organ}.rs"
                if src_p.exists():
                    self.code_editor.setPlainText(src_p.read_text(encoding="utf-8"))

            self.worker = AsyncWorker(_run_step)
            self.worker.finished_signal.connect(_done)
            self.worker.start()

        def on_reset(self) -> None:
            controller.reset_to_lkg()

    app = QApplication(sys.argv)
    win = FishbowlMainWindow()
    win.show()
    return app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="SOMA-Fishbowl PyQt6 GUI + Network Control Server (:18940)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18940)
    parser.add_argument("--llm-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--headless", action="store_true", help="Run network control server without PyQt6 window")
    args = parser.parse_args()

    controller = FishbowlController(llm_endpoint=args.llm_endpoint)
    httpd = start_network_server(controller, args.host, args.port)

    if args.headless or not os.environ.get("DISPLAY"):
        controller._log("Running in headless network-server mode (Ctrl+C to stop).")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            httpd.shutdown()
    else:
        sys.exit(run_pyqt_gui(controller, args.port))


if __name__ == "__main__":
    main()
