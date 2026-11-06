#!/usr/bin/env python3
"""
ls_moe_web_ui.py - Remote Web UI & SSE Streaming Server for `rust_localsleep-moe`
=================================================================================
Default Listen Address: http://0.0.0.0:18935 (User-launched via `./run_web_ui.sh`)

Features:
  - Remote-accessible Web UI (`0.0.0.0:18935`) with multi-line prompt input & real-time SSE streaming chat.
  - Full session persistence via `SessionLogger` -> `sessions/session_<profile>_<timestamp>.jsonl` & `.md`
    + `sessions/localsleep_telemetry_vault.db` + `checkpoints/latest_<profile>_<cube_id>.safetensors`.
  - Real-time Speed (`tok/s`), TTFT (`ms`), Token count, Entropy (`H`), and NREM/Prune counters.
  - Live 8/16-Expert Fatigue (`F`), State (`FIRE`/`ACT`/`FAT`/`NREM`), and Norm (`||A||_F`, `Δ||A||_k`) dashboard.
  - Instant `LoRA ON / OFF (Bypass)` toggle + Hypercube Core Engine hot-swap (`gen_022`, etc.).
  - Low-overhead HTML5 Canvas `2:4 Microscopic Synaptic Weight Magnifier` (`12x24` GPU slice on demand / throttled).
"""

import argparse
import datetime
import json
import math
import os
import shutil
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from checkpoint_manager import EmbodiedCheckpointManager
from cjk_cleaner import CJKStreamCleaner
from expert_pool_lora import DynamicLoRAExpertPool
from generation_engine_trt_hybrid import TrtEdgeLlmHybridEngine
from hypercube_replicator import HypercubeReplicatorManager
from ls_moe_agentic import init_rust_router_and_pool, load_profile_config
from session_logger import SessionLogger
from tool_parser import ToolCallParser
from tools_harness import EmbodiedToolHarness
from working_memory_manager import WorkingMemoryManager

UI_HTML_PATH = BASE_DIR / "ls_moe_web_ui.html"


class DummyArgs:
    def __init__(self, profile: str, max_tokens: int, temp: float, top_p: float = 0.95):
        self.profile = profile
        self.max_tokens = max_tokens
        self.max_context_tokens = 32768
        self.temp = temp
        self.top_p = top_p
        self.reset_checkpoint = False


class LocalSleepWebState:
    def __init__(self, qt_win: Any = None) -> None:
        self.lock = threading.RLock()
        self.infer_lock = threading.Lock()
        self.qt_win = qt_win
        self.abort_flag = False
        self.is_generating = False

        self.profile_name = "thor_31b"
        self.selected_cube = ""
        self.server_url = "http://192.168.50.67:8000"
        self.model_id = "Gemma-4-31B-IT-NVFP4"
        self.max_tokens = 16384
        self.temperature = 0.6
        self.shy_lambda = 0.08
        self.lora_enabled = True
        self.loop_mode = "mode1_vram"  # mode1_vram | mode2_micro | mode0_off
        self.moe_alpha = 1.0
        self.micro_chunk = 256
        self.wm_inject = True

        self.rust_router: Any = None
        self.expert_pool: Optional[DynamicLoRAExpertPool] = None
        self.ckpt_mgr: Optional[EmbodiedCheckpointManager] = None
        self.session_logger: Optional[SessionLogger] = None
        self.wm_mgr: Optional[WorkingMemoryManager] = None
        self.harness: Optional[EmbodiedToolHarness] = None
        self.hybrid_engine: Optional[TrtEdgeLlmHybridEngine] = None
        self.system_prompt: str = ""
        self.conversation_history: List[Dict[str, str]] = []
        self.active_routed: List[int] = [0, 1]
        self.last_stats: Dict[str, Any] = {
            "tps": 0.0,
            "ttft_ms": 0.0,
            "tokens": 0,
            "elapsed": 0.0,
            "entropy": 0.0,
        }

        if self.qt_win is not None:
            self.sync_from_qt_win()
        else:
            self._load_initial_profile()
            self.init_engine(reset_ckpt=False, resume_sess=False)

    def sync_from_qt_win(self) -> None:
        """Bind directly to PyQt6 main window's live engine, router, expert pool, and session logger."""
        w = self.qt_win
        if w is None:
            return
        with self.lock:
            self.rust_router = getattr(w, "rust_router", None)
            self.expert_pool = getattr(w, "expert_pool", None)
            self.ckpt_mgr = getattr(w, "ckpt_mgr", None)
            self.session_logger = getattr(w, "session_logger", None)
            self.wm_mgr = getattr(w, "wm_mgr", None)
            self.harness = getattr(w, "harness", None)
            self.hybrid_engine = getattr(w, "hybrid_engine", None)
            self.system_prompt = getattr(w, "system_prompt", "")
            self.conversation_history = getattr(w, "conversation_history", [])
            if hasattr(w, "cmb_profile"):
                self.profile_name = w.cmb_profile.currentText()
            if hasattr(w, "cmb_core_engine"):
                self.selected_cube = str(w.cmb_core_engine.currentData() or "")
            if hasattr(w, "inp_url"):
                self.server_url = w.inp_url.text().strip()
            if hasattr(w, "inp_model"):
                self.model_id = w.inp_model.text().strip()
            if hasattr(w, "spn_max_toks"):
                self.max_tokens = int(w.spn_max_toks.value())
            if hasattr(w, "spn_temp"):
                self.temperature = float(w.spn_temp.value())
            if hasattr(w, "spn_shy"):
                self.shy_lambda = float(w.spn_shy.value())
            if hasattr(w, "cmb_loop_mode"):
                self.loop_mode = str(w.cmb_loop_mode.currentData() or "mode1_vram")
            if hasattr(w, "spn_moe_alpha"):
                self.moe_alpha = float(w.spn_moe_alpha.value())
            if hasattr(w, "spn_micro_chunk"):
                self.micro_chunk = int(w.spn_micro_chunk.value())
            if hasattr(w, "chk_wm_inject"):
                self.wm_inject = bool(w.chk_wm_inject.isChecked())
            if self.hybrid_engine is not None:
                self.lora_enabled = not bool(getattr(self.hybrid_engine, "bypass_lora", False))

    def _load_initial_profile(self) -> None:
        cfg_path = BASE_DIR / "configs" / "model_profiles.json"
        if not cfg_path.exists():
            return
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            act = str(data.get("active_profile") or "thor_31b")
            profs = data.get("profiles", {})
            if act in profs:
                self.profile_name = act
                p = profs[act]
                self.server_url = str(p.get("server_url", self.server_url))
                self.model_id = str(p.get("model_id", self.model_id))
                hyper = p.get("hyperparameters", {})
                if hyper.get("temperature") is not None:
                    self.temperature = float(hyper["temperature"])
                if hyper.get("max_generation_tokens") is not None:
                    self.max_tokens = int(hyper["max_generation_tokens"])
        except Exception:
            pass

    @staticmethod
    def _bootstrap_rust_sandbox(sandbox_path: Path) -> None:
        try:
            (sandbox_path / "src").mkdir(parents=True, exist_ok=True)
            cargo_toml = sandbox_path / "Cargo.toml"
            if not cargo_toml.exists():
                cargo_toml.write_text(
                    '[package]\nname = "ntm_rsi_core"\nversion = "0.1.0"\nedition = "2021"\n\n'
                    '[lib]\nname = "ntm_rsi_core"\ncrate-type = ["cdylib"]\n\n'
                    '[dependencies]\n'
                    'pyo3 = { version = "0.20", features = ["extension-module"] }\n'
                    'numpy = "0.20"\n'
                    'ndarray = { version = "0.15", features = ["rayon"] }\n'
                    'rand = "0.8"\nrayon = "1.8"\nparking_lot = "0.12"\n'
                    'smallvec = "1.13"\nrustc-hash = "1.1"\n'
                    'serde = { version = "1.0", features = ["derive"] }\n',
                    encoding="utf-8",
                )
            tmpl_lock = BASE_DIR / "isolated_output" / "prj_20261030_220710_3f6f5c57" / "Cargo.lock"
            if tmpl_lock.exists() and not (sandbox_path / "Cargo.lock").exists():
                shutil.copy2(str(tmpl_lock), str(sandbox_path / "Cargo.lock"))
            tmpl_cfg = BASE_DIR / "isolated_output" / "prj_20261030_220710_3f6f5c57" / ".cargo"
            if tmpl_cfg.exists() and not (sandbox_path / ".cargo").exists():
                shutil.copytree(str(tmpl_cfg), str(sandbox_path / ".cargo"))
        except Exception:
            pass

    def list_available_cores(self) -> List[Dict[str, str]]:
        cores = [
            {"id": "", "label": "🔥 [自動跟隨 Active 超立方體] (Auto Sync)"},
            {"id": "gen_000_origin", "label": "🦀 [原版 8-LoRA 基底母體] gen_000_origin"},
        ]
        gen_dir = BASE_DIR / "hypercube_generations"
        if gen_dir.exists():
            cands = sorted(
                [d for d in gen_dir.iterdir() if d.is_dir() and d.name.startswith("gen_")],
                key=lambda x: x.name,
                reverse=True,
            )
            for d in cands:
                plug = d / "evolved_router_plugin.py"
                tag = "8-LoRA"
                if plug.exists():
                    try:
                        txt = plug.read_text(encoding="utf-8", errors="ignore")
                        if "num_experts=16" in txt or "num_experts = 16" in txt:
                            tag = "16-LoRA + Auto Top-K"
                    except Exception:
                        pass
                cores.append({"id": d.name, "label": f"🧬 {d.name} [{tag}]"})
        return cores

    def list_recent_sessions(self) -> List[Dict[str, Any]]:
        out = [{"id": "latest", "label": "⚡ [自動] 倒數第 1 場 (latest)"}]
        sess_dir = BASE_DIR / "sessions"
        cur_name = os.path.basename(self.session_logger.jsonl_file) if self.session_logger else ""
        if sess_dir.exists():
            cands = sorted(
                [
                    p for p in sess_dir.glob("session_*.jsonl")
                    if p.name != cur_name
                    and not p.name.startswith("session_real_telemetry_")
                    and not p.name.startswith("session_rust_telemetry_")
                    and p.stat().st_size > 20
                ],
                key=lambda x: x.name,
                reverse=True,
            )
            for p in cands[:20]:
                turns = 0
                first_u = ""
                try:
                    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if rec.get("type") == "turn_complete":
                            d = rec.get("data") or {}
                            if d.get("role") == "user":
                                turns += 1
                                if not first_u:
                                    first_u = (d.get("content") or "").replace("\n", " ").strip()[:24]
                except Exception:
                    continue
                if turns > 0:
                    out.append({
                        "id": p.name,
                        "label": f"📄 {p.name.replace('session_', '').replace('.jsonl', '')} ({turns}輪 | {first_u}...)",
                    })
        return out

    def init_engine(
        self,
        reset_ckpt: bool = False,
        resume_sess: bool = False,
        resume_target: str = "latest",
    ) -> Dict[str, Any]:
        with self.lock:
            prof = load_profile_config(str(BASE_DIR / "configs/model_profiles.json"), self.profile_name)
            prof["shy_lambda"] = float(self.shy_lambda)
            base_sys = prof.get(
                "system_prompt",
                "你是數位黑貓 Kuro (小黑)，Leon (主人) 的 S-tier AI Familiar。奉行負熵煉金術：交付結果大於文字排場，先讀碼後推論。",
            )

            if self.selected_cube and self.selected_cube != "gen_000_origin":
                try:
                    rep = HypercubeReplicatorManager(workspace_root=str(BASE_DIR))
                    rep.rotate_hypercube_dimension(self.selected_cube)
                except Exception:
                    pass

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.rust_router, self.expert_pool = init_rust_router_and_pool(
                prof, device=device, override_cube_id=self.selected_cube
            )
            active_n_exp = int(getattr(self.rust_router, "num_experts", 8))
            loaded_cube_tag = str(getattr(self.rust_router, "loaded_cube_id", self.selected_cube or "gen_000_origin"))

            ckpt_profile_key = self.profile_name if active_n_exp == 8 else f"{self.profile_name}_{active_n_exp}lora"
            self.ckpt_mgr = EmbodiedCheckpointManager(
                checkpoint_dir=str(BASE_DIR / "checkpoints"),
                profile=ckpt_profile_key,
                cube_id=loaded_cube_tag,
            )
            self.ckpt_mgr.load_state(self.rust_router, expert_pool=self.expert_pool, reset=reset_ckpt)

            if self.session_logger is None or reset_ckpt or not resume_sess:
                self.session_logger = SessionLogger(
                    session_dir=str(BASE_DIR / "sessions"),
                    profile=self.profile_name,
                    model_path=self.model_id,
                    benchmark_dir=str(BASE_DIR / "benchmarks"),
                )

            detected_prj = None
            if resume_sess and self.session_logger is not None:
                self.conversation_history, detected_prj = self.session_logger.load_resumed_history(
                    session_spec=resume_target,
                    skip_last_turns=0,
                )
            elif not resume_sess:
                self.conversation_history = []

            self.wm_mgr = WorkingMemoryManager(
                workspace_root=str(BASE_DIR),
                session_dir=str(BASE_DIR / "sessions"),
                profile=self.profile_name,
            )
            prj_uuid = (
                detected_prj
                if (resume_sess and detected_prj)
                else f"prj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )
            isolated_dir = str(BASE_DIR / "isolated_output" / prj_uuid)
            self._bootstrap_rust_sandbox(Path(isolated_dir))

            wm_injection, _ = self.wm_mgr.build_system_prompt_injection(current_sandbox_dir=isolated_dir)
            self.system_prompt = (base_sys + wm_injection) if self.wm_inject else base_sys

            self.harness = EmbodiedToolHarness(
                workspace_root=str(BASE_DIR),
                isolated_output_dir=isolated_dir,
                session_logger=self.session_logger,
                working_memory_mgr=self.wm_mgr,
            )
            if getattr(self.session_logger, "vault_db", None) is not None:
                try:
                    self.session_logger.vault_db.register_session(
                        server_url=self.server_url,
                        active_cube_id=loaded_cube_tag,
                        sandbox_dir=isolated_dir,
                    )
                except Exception:
                    pass

            dummy_args = DummyArgs(
                profile=self.profile_name,
                max_tokens=int(self.max_tokens),
                temp=float(self.temperature),
            )
            self.hybrid_engine = TrtEdgeLlmHybridEngine(
                server_url=self.server_url,
                model_id=self.model_id,
                rust_router=self.rust_router,
                expert_pool=self.expert_pool,
                harness=self.harness,
                profile_config=prof,
                args=dummy_args,
                device=device,
            )
            self.model_id = self.hybrid_engine.model_id
            self.apply_lora_and_loop_mode()
            return self.get_telemetry_snapshot(include_magnifier=True)

    def apply_lora_and_loop_mode(self) -> None:
        if self.qt_win is not None:
            self.rust_router = getattr(self.qt_win, "rust_router", self.rust_router)
            self.expert_pool = getattr(self.qt_win, "expert_pool", self.expert_pool)
            self.ckpt_mgr = getattr(self.qt_win, "ckpt_mgr", self.ckpt_mgr)
            self.session_logger = getattr(self.qt_win, "session_logger", self.session_logger)
            self.wm_mgr = getattr(self.qt_win, "wm_mgr", self.wm_mgr)
            self.harness = getattr(self.qt_win, "harness", self.harness)
            self.hybrid_engine = getattr(self.qt_win, "hybrid_engine", self.hybrid_engine)
            self.system_prompt = getattr(self.qt_win, "system_prompt", self.system_prompt)
            self.conversation_history = getattr(self.qt_win, "conversation_history", self.conversation_history)
        if self.hybrid_engine is None:
            return
        self.hybrid_engine.bypass_lora = not bool(self.lora_enabled)
        self.hybrid_engine.max_tokens = int(self.max_tokens)
        self.hybrid_engine.temperature = float(self.temperature)
        self.hybrid_engine.shy_lambda = float(self.shy_lambda)

        if not self.lora_enabled or self.loop_mode == "mode0_off":
            self.hybrid_engine.vram_live_gate_enabled = False
            self.hybrid_engine.micro_chunk_tokens = 0
            self.hybrid_engine.dynamic_logit_bias.clear()
            self.hybrid_engine.push_live_vram_gate_async(enabled=False, alpha=0.0)
        elif self.loop_mode == "mode2_micro":
            self.hybrid_engine.vram_live_gate_enabled = False
            self.hybrid_engine.micro_chunk_tokens = int(self.micro_chunk)
            self.hybrid_engine.push_live_vram_gate_async(enabled=False, alpha=0.0)
        else:
            self.hybrid_engine.vram_live_gate_enabled = True
            self.hybrid_engine.vram_live_alpha = float(self.moe_alpha)
            self.hybrid_engine.micro_chunk_tokens = 0
            self.hybrid_engine.push_live_vram_gate_async(enabled=True, alpha=float(self.moe_alpha))
        if self.qt_win is not None and hasattr(self.qt_win, "sig_web_sync_ui"):
            try:
                self.qt_win.sig_web_sync_ui.emit()
            except Exception:
                pass

    def get_telemetry_snapshot(self, include_magnifier: bool = True) -> Dict[str, Any]:
        with self.lock:
            if self.qt_win is not None:
                self.rust_router = getattr(self.qt_win, "rust_router", self.rust_router)
                self.expert_pool = getattr(self.qt_win, "expert_pool", self.expert_pool)
                self.session_logger = getattr(self.qt_win, "session_logger", self.session_logger)
                self.harness = getattr(self.qt_win, "harness", self.harness)
                self.hybrid_engine = getattr(self.qt_win, "hybrid_engine", self.hybrid_engine)
                self.conversation_history = getattr(self.qt_win, "conversation_history", self.conversation_history)
            n_exp = int(getattr(self.rust_router, "num_experts", 8)) if self.rust_router else 8
            loaded_cube = str(getattr(self.rust_router, "loaded_cube_id", "gen_000_origin"))
            raw_states = []
            if self.rust_router is not None and hasattr(self.rust_router, "get_expert_states"):
                try:
                    raw_states = self.rust_router.get_expert_states()
                except Exception:
                    raw_states = []

            norms: List[float] = [2.60] * n_exp
            grids: List[List[List[float]]] = []
            if self.expert_pool is not None:
                with torch.no_grad():
                    for i in range(n_exp):
                        key = f"exp_{i}"
                        if key in self.expert_pool.lora_a:
                            la = self.expert_pool.lora_a[key].data
                            norms[i] = round(float(torch.norm(la.float()).item()), 4)
                            if include_magnifier:
                                sub = la[:12, :24].float().cpu().numpy()
                                grids.append(np.round(sub, 5).tolist())
                            else:
                                grids.append([])
                        else:
                            grids.append([])
            else:
                grids = [[] for _ in range(n_exp)]

            mean_norm = sum(norms) / float(max(1, n_exp))
            experts_info = []
            for i in range(n_exp):
                st = raw_states[i] if i < len(raw_states) else (i, 0, 0.0, 0, 0, 50.0, 0)
                st_code = int(st[1])
                fat_val = round(float(st[2]), 3)
                tok_val = int(st[3])
                slp_cd = int(st[4])
                spar_pct = round(float(st[5]), 1)
                slp_cyc = int(st[6]) if len(st) >= 7 else 0
                is_routed = (i in self.active_routed) and self.lora_enabled
                delta_norm = round(norms[i] - mean_norm, 4)
                experts_info.append({
                    "id": i,
                    "state_code": st_code,
                    "is_routed": is_routed,
                    "fatigue": fat_val,
                    "tokens": tok_val,
                    "sleep_cd": slp_cd,
                    "sparsity_pct": spar_pct,
                    "sleep_cycles": slp_cyc,
                    "norm": norms[i],
                    "delta_norm": delta_norm,
                    "grid_12x24": grids[i] if include_magnifier else [],
                })

            sess_file = os.path.basename(self.session_logger.md_file) if self.session_logger else ""
            sandbox_name = os.path.basename(self.harness.isolated_output_dir) if self.harness else ""
            return {
                "profile": self.profile_name,
                "server_url": self.server_url,
                "model_id": self.model_id,
                "loaded_cube_id": loaded_cube,
                "selected_cube": self.selected_cube,
                "num_experts": n_exp,
                "lora_enabled": self.lora_enabled,
                "loop_mode": self.loop_mode,
                "moe_alpha": self.moe_alpha,
                "micro_chunk": self.micro_chunk,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "shy_lambda": self.shy_lambda,
                "wm_inject": self.wm_inject,
                "is_generating": self.is_generating,
                "session_file": sess_file,
                "sandbox_dir": sandbox_name,
                "history_turns": len(self.conversation_history) // 2,
                "active_routed": list(self.active_routed) if self.lora_enabled else [],
                "mean_norm": round(mean_norm, 4),
                "nrem_events": int(getattr(self.hybrid_engine, "total_nrem_sleep_events", 0)),
                "gpu_24_prunes": int(getattr(self.hybrid_engine, "total_gpu_24_prunes", 0)),
                "bifurcation_flips": int(getattr(self.hybrid_engine, "total_bifurcation_flips", 0)),
                "last_stats": dict(self.last_stats),
                "experts": experts_info,
            }


GLOBAL_STATE: Optional[LocalSleepWebState] = None


class LocalSleepWebHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="ignore")
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        st = GLOBAL_STATE
        if st is None:
            self._send_json({"error": "Server initializing"}, 503)
            return

        if path in ("/", "/index.html"):
            if not UI_HTML_PATH.exists():
                self._send_json({"error": "ls_moe_web_ui.html missing"}, 404)
                return
            raw = UI_HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return

        if path in ("/v1/models", "/api/v1/models"):
            snap = st.get_telemetry_snapshot(include_magnifier=False)
            now_ts = int(time.time())
            models_data = [
                {
                    "id": "localsleep-moe",
                    "object": "model",
                    "created": now_ts,
                    "owned_by": f"rust_localsleep-moe:{snap['loaded_cube_id']}",
                    "meta": {
                        "base_model": snap["model_id"],
                        "profile": snap["profile"],
                        "loaded_cube_id": snap["loaded_cube_id"],
                        "num_experts": snap["num_experts"],
                        "lora_enabled": snap["lora_enabled"],
                    },
                },
                {
                    "id": snap["model_id"],
                    "object": "model",
                    "created": now_ts,
                    "owned_by": f"rust_localsleep-moe:{snap['loaded_cube_id']}",
                },
                {
                    "id": "localsleep-moe-vanilla",
                    "object": "model",
                    "created": now_ts,
                    "owned_by": "rust_localsleep-moe:bypass",
                },
            ]
            self._send_json({"object": "list", "data": models_data})
            return

        if path == "/api/state":
            inc_mag = qs.get("magnifier", ["1"])[0] != "0"
            self._send_json({
                "ok": True,
                "state": st.get_telemetry_snapshot(include_magnifier=inc_mag),
                "cores": st.list_available_cores(),
                "sessions": st.list_recent_sessions(),
                "history": st.conversation_history[-24:],
            })
            return

        self._send_json({"error": f"Unknown path: {path}"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        st = GLOBAL_STATE
        if st is None:
            self._send_json({"error": "Server initializing"}, 503)
            return

        if path == "/api/config":
            body = self._read_json_body()
            with st.lock:
                reinit_needed = False
                if "profile" in body and body["profile"] != st.profile_name:
                    st.profile_name = str(body["profile"])
                    prof = load_profile_config(str(BASE_DIR / "configs/model_profiles.json"), st.profile_name)
                    st.server_url = str(prof.get("server_url", st.server_url))
                    st.model_id = str(prof.get("model_id", st.model_id))
                    reinit_needed = True
                if "selected_cube" in body and str(body["selected_cube"]) != st.selected_cube:
                    st.selected_cube = str(body["selected_cube"])
                    reinit_needed = True
                if "server_url" in body and str(body["server_url"]).strip():
                    st.server_url = str(body["server_url"]).strip().rstrip("/")
                    if st.hybrid_engine:
                        st.hybrid_engine.server_url = st.server_url
                if "model_id" in body and str(body["model_id"]).strip():
                    st.model_id = str(body["model_id"]).strip()
                    if st.hybrid_engine:
                        st.hybrid_engine.model_id = st.model_id
                if "lora_enabled" in body:
                    st.lora_enabled = bool(body["lora_enabled"])
                if "loop_mode" in body:
                    st.loop_mode = str(body["loop_mode"])
                if "moe_alpha" in body:
                    st.moe_alpha = float(body["moe_alpha"])
                if "micro_chunk" in body:
                    st.micro_chunk = max(32, int(body["micro_chunk"]))
                if "max_tokens" in body:
                    st.max_tokens = max(128, int(body["max_tokens"]))
                if "temperature" in body:
                    st.temperature = max(0.0, min(2.0, float(body["temperature"])))
                if "shy_lambda" in body:
                    st.shy_lambda = max(0.005, min(0.5, float(body["shy_lambda"])))
                if "wm_inject" in body:
                    st.wm_inject = bool(body["wm_inject"])

                action = str(body.get("action") or "")
                if st.qt_win is not None:
                    # When bound to PyQt6 GUI in Dual-Mode, keep shared engine references and sync UI
                    if action == "save_ckpt" and st.ckpt_mgr and st.rust_router:
                        st.ckpt_mgr.save_state(
                            st.rust_router,
                            expert_pool=st.expert_pool,
                            tokens_generated=len(st.conversation_history),
                        )
                    elif action == "new_session":
                        st.conversation_history.clear()
                    elif action == "reset_ckpt" and st.ckpt_mgr and st.rust_router:
                        st.ckpt_mgr.load_state(st.rust_router, expert_pool=st.expert_pool, reset=True)
                    st.apply_lora_and_loop_mode()
                    snap = st.get_telemetry_snapshot(include_magnifier=True)
                elif action == "reset_ckpt":
                    snap = st.init_engine(reset_ckpt=True, resume_sess=False)
                elif action == "resume_session":
                    target = str(body.get("resume_target") or "latest")
                    snap = st.init_engine(reset_ckpt=False, resume_sess=True, resume_target=target)
                elif action == "new_session":
                    snap = st.init_engine(reset_ckpt=False, resume_sess=False)
                elif action == "save_ckpt":
                    if st.ckpt_mgr and st.rust_router:
                        st.ckpt_mgr.save_state(
                            st.rust_router,
                            expert_pool=st.expert_pool,
                            tokens_generated=len(st.conversation_history),
                        )
                    st.apply_lora_and_loop_mode()
                    snap = st.get_telemetry_snapshot(include_magnifier=True)
                elif reinit_needed or action == "reinit":
                    snap = st.init_engine(reset_ckpt=False, resume_sess=True)
                else:
                    st.apply_lora_and_loop_mode()
                    snap = st.get_telemetry_snapshot(include_magnifier=True)

            self._send_json({
                "ok": True,
                "state": snap,
                "cores": st.list_available_cores(),
                "sessions": st.list_recent_sessions(),
                "history": st.conversation_history[-24:],
            })
            return

        if path == "/api/stop":
            st.abort_flag = True
            if st.qt_win is not None and hasattr(st.qt_win, "stop_generation"):
                try:
                    st.qt_win.stop_generation()
                except Exception:
                    pass
            self._send_json({"ok": True})
            return

        if path == "/api/chat/stream":
            body = self._read_json_body()
            user_prompt = str(body.get("prompt") or "").strip()
            include_mag = bool(body.get("include_magnifier", True))
            if not user_prompt:
                self._send_json({"error": "Empty prompt"}, 400)
                return
            if st.qt_win is not None and getattr(st.qt_win, "worker", None) is not None and st.qt_win.worker.isRunning():
                self._send_json({"error": "PyQt GUI inference is currently running"}, 409)
                return
            if not st.infer_lock.acquire(blocking=False):
                self._send_json({"error": "Inference already running"}, 409)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            def send_sse(event_type: str, payload_obj: Dict[str, Any]) -> bool:
                try:
                    msg = f"event: {event_type}\ndata: {json.dumps(payload_obj, ensure_ascii=False)}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                    return True
                except Exception:
                    st.abort_flag = True
                    return False

            try:
                st.abort_flag = False
                st.is_generating = True
                st.apply_lora_and_loop_mode()
                eng = st.hybrid_engine
                assert eng is not None and st.session_logger is not None

                messages: List[Dict[str, Any]] = [{"role": "system", "content": st.system_prompt}]
                if st.conversation_history:
                    messages.extend(st.conversation_history)
                messages.append({"role": "user", "content": user_prompt})
                messages = eng._compress_swa_messages(messages)

                payload: Dict[str, Any] = {
                    "model": eng.model_id,
                    "messages": messages,
                    "max_tokens": eng.max_tokens,
                    "temperature": eng.temperature,
                    "top_p": eng.top_p,
                    "stop": eng.stop_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "logprobs": True,
                    "top_logprobs": 4,
                    "enable_thinking": False,
                }
                if st.lora_enabled and eng.dynamic_logit_bias:
                    payload["logit_bias"] = dict(list(eng.dynamic_logit_bias.items())[:64])

                cjk_cleaner = CJKStreamCleaner()
                accumulated_text = ""
                assistant_so_far = ""
                chunk_steps = 0
                completion_tokens = 0
                ttft_ms = 0.0
                t0 = time.time()
                first_token_time = None
                last_weights: List[float] = [0.5, 0.5]
                last_experts: List[int] = [0, 1]
                entropy_sum = 0.0
                tool_calls_count = 0
                MAX_TOOL_CALLS = int(getattr(eng, "max_tool_calls", 100) or 100)
                consecutive_same_sig = 0
                last_tool_sig = ""
                micro_chunk_limit = int(getattr(eng, "micro_chunk_tokens", 0) or 0) if st.lora_enabled else 0
                micro_slice_count = 0

                if st.lora_enabled and eng.expert_pool is not None and hasattr(eng.expert_pool, "begin_eligibility_episode"):
                    eng.expert_pool.begin_eligibility_episode()
                eng._turn_bifurcation_flips = 0
                if st.qt_win is not None and hasattr(st.qt_win, "sig_web_append_chat"):
                    try:
                        st.qt_win.sig_web_append_chat.emit(f"\n🌐 [Web UI] Leon >> {user_prompt}\nKuro ({eng.model_id}) >> ")
                    except Exception:
                        pass

                while True:
                    remaining_budget = max(16, eng.max_tokens - completion_tokens)
                    eff_max_toks = min(micro_chunk_limit, remaining_budget) if micro_chunk_limit > 0 else remaining_budget
                    payload["max_tokens"] = eff_max_toks
                    payload["reuse_context"] = True
                    payload["cache_generated_tokens"] = True
                    if st.lora_enabled and eng.dynamic_logit_bias:
                        payload["logit_bias"] = dict(list(eng.dynamic_logit_bias.items())[:64])
                    else:
                        payload.pop("logit_bias", None)

                    req = urllib.request.Request(
                        f"{eng.server_url}/v1/chat/completions",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                        method="POST",
                    )
                    seg_text = ""
                    intercepted_tool = None
                    finish_reason = None

                    with urllib.request.urlopen(req, timeout=180.0) as resp:
                        for raw_line in resp:
                            if st.abort_flag:
                                break
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except Exception:
                                continue

                            if chunk.get("usage"):
                                completion_tokens += int(chunk["usage"].get("completion_tokens", 0))

                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            choice0 = choices[0]
                            if choice0.get("finish_reason"):
                                finish_reason = choice0.get("finish_reason")
                            delta = choice0.get("delta", {})
                            content = delta.get("content")
                            lp_info = choice0.get("logprobs") or {}
                            lp_content = lp_info.get("content") or []

                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    ttft_ms = (first_token_time - t0) * 1000.0

                                cleaned_piece = cjk_cleaner.clean(content)
                                seg_text += cleaned_piece
                                assistant_so_far += cleaned_piece
                                accumulated_text += cleaned_piece

                                if lp_content:
                                    for idx_lp, entry in enumerate(lp_content):
                                        tid = int(entry.get("token_id") or 0)
                                        top_lps = entry.get("top_logprobs") or []
                                        top_tids = [int(x.get("token_id") or 0) for x in top_lps] or [tid]
                                        top_vals = [float(x.get("logprob") or -1.0) for x in top_lps] or [float(entry.get("logprob") or -0.1)]
                                        top_probs = [math.exp(max(-20.0, min(0.0, v))) for v in top_vals]
                                        tok_str_rec = str(entry.get("token") or entry.get("piece") or (cleaned_piece if idx_lp == 0 else ""))
                                        w, sel, ent, _ = eng._step_local_rust_moe(tid, top_tids, top_vals, token_piece=tok_str_rec)
                                        last_weights, last_experts = w, sel
                                        st.active_routed = list(sel)
                                        entropy_sum += ent
                                        chunk_steps += 1
                                        cur_el = max(0.001, time.time() - t0)
                                        cur_tps = chunk_steps / cur_el
                                        st.session_logger.record_token_telemetry(
                                            token_id=tid,
                                            token_str=tok_str_rec,
                                            tps=cur_tps,
                                            entropy=ent,
                                            selected_experts=sel,
                                            weights=w,
                                            expert_states=eng.rust_router.get_expert_states() if eng.rust_router else [],
                                            top_probs=top_probs,
                                        )
                                else:
                                    synth_tid = sum(ord(c) * (i + 1) for i, c in enumerate(content)) % 32000
                                    w, sel, ent, _ = eng._step_local_rust_moe(synth_tid, [synth_tid], [-0.15], token_piece=cleaned_piece)
                                    last_weights, last_experts = w, sel
                                    st.active_routed = list(sel)
                                    entropy_sum += ent
                                    chunk_steps += 1
                                    cur_el = max(0.001, time.time() - t0)
                                    cur_tps = chunk_steps / cur_el
                                    st.session_logger.record_token_telemetry(
                                        token_id=synth_tid,
                                        token_str=cleaned_piece,
                                        tps=cur_tps,
                                        entropy=ent,
                                        selected_experts=sel,
                                        weights=w,
                                        expert_states=eng.rust_router.get_expert_states() if eng.rust_router else [],
                                        top_probs=[0.85],
                                    )

                                cur_el = max(0.001, time.time() - t0)
                                cur_tps = chunk_steps / cur_el
                                st.last_stats = {
                                    "tps": round(cur_tps, 2),
                                    "ttft_ms": round(ttft_ms, 1),
                                    "tokens": chunk_steps,
                                    "elapsed": round(cur_el, 2),
                                    "entropy": round(entropy_sum / max(1, chunk_steps), 3),
                                }

                                evt_data: Dict[str, Any] = {
                                    "piece": cleaned_piece,
                                    "stats": st.last_stats,
                                    "routed": list(last_experts) if st.lora_enabled else [],
                                    "weights": [round(float(x), 3) for x in last_weights],
                                }
                                if chunk_steps % 6 == 0:
                                    evt_data["telemetry"] = st.get_telemetry_snapshot(
                                        include_magnifier=(include_mag and (chunk_steps % 12 == 0))
                                    )
                                if st.qt_win is not None and hasattr(st.qt_win, "sig_web_stream_piece"):
                                    try:
                                        st.qt_win.sig_web_stream_piece.emit(
                                            cleaned_piece,
                                            int(chunk_steps),
                                            float(st.last_stats["entropy"]),
                                            list(last_experts) if st.lora_enabled else [],
                                            [float(x) for x in last_weights],
                                        )
                                    except Exception:
                                        pass
                                if not send_sse("token", evt_data):
                                    break

                                if ToolCallParser.has_complete_tool_call(seg_text):
                                    did_fix, seg_text = ToolCallParser.repair_missing_closing_angle(seg_text)
                                    if did_fix:
                                        accumulated_text += ">"
                                        send_sse("token", {"piece": ">", "stats": st.last_stats})
                                    eng.harness.extract_and_save_artifacts(seg_text)
                                    act, out, clean_prefix = ToolCallParser.detect_and_execute(seg_text, eng.harness)
                                    if act and out is not None:
                                        intercepted_tool = (act, str(out), clean_prefix)
                                        break

                    if st.abort_flag:
                        break

                    if intercepted_tool is not None:
                        act, tool_output, clean_prefix = intercepted_tool
                        tc_idx = seg_text.find("<tool_call>")
                        raw_tc_block = seg_text[tc_idx:] if tc_idx != -1 else f"<tool_call>{json.dumps({'action': act}, ensure_ascii=False)}</tool_call>"
                        if not raw_tc_block.endswith(">"):
                            raw_tc_block += ">"

                        sig = f"{act}:{raw_tc_block}:{len(tool_output)}:{tool_output[-120:]}"
                        consecutive_same_sig = (consecutive_same_sig + 1) if sig == last_tool_sig else 1
                        last_tool_sig = sig
                        if consecutive_same_sig >= 3 or tool_calls_count >= MAX_TOOL_CALLS:
                            break
                        tool_calls_count += 1

                        ui_preview = (
                            f"{tool_output[:1000]}\n... [折疊預覽：共 {len(tool_output):,} chars 已餵入模型] ...\n{tool_output[-500:]}"
                            if len(tool_output) > 1600
                            else tool_output
                        )
                        elig_note = ""
                        if st.lora_enabled and eng.expert_pool is not None and hasattr(eng.expert_pool, "settle_eligibility_trace"):
                            if "EXIT_CODE: 0" in str(tool_output):
                                st_tag = eng.expert_pool.settle_eligibility_trace(sandbox_passed=True)
                                elig_note = f"\n🧬 [Eligibility Trace -> {st_tag}] 沙盒驗證通過，短期赫布跡象已固化為長期突觸記憶！"
                            elif "EXIT_CODE:" in str(tool_output):
                                st_tag = eng.expert_pool.settle_eligibility_trace(sandbox_passed=False)
                                elig_note = f"\n🛡️ [Eligibility Trace -> {st_tag}] 攔截沙盒失敗路徑，已執行反赫布回滾！"

                        tool_banner = f"\n\n📥 [Sandbox Tool Executed (#{tool_calls_count}: {act})]:\n{ui_preview}{elig_note}\n⚡ [Kuro Post-Tool Stream]: "
                        accumulated_text += tool_banner
                        send_sse("token", {"piece": tool_banner, "stats": st.last_stats})

                        messages.append({"role": "assistant", "content": (clean_prefix + "\n" + raw_tc_block).strip()})
                        messages.append({
                            "role": "user",
                            "content": f"【環境工具回傳結果 (System Tool Response)】：\n{tool_output}\n請根據上述真實工具回傳結果繼續回答主人。",
                        })
                        messages = eng._compress_swa_messages(messages)
                        payload["messages"] = messages
                        assistant_so_far = ""
                        continue

                    if (
                        micro_chunk_limit > 0
                        and (finish_reason == "length" or (seg_text and len(seg_text) >= 8 and finish_reason is None))
                        and completion_tokens < eng.max_tokens
                        and micro_slice_count < 256
                    ):
                        micro_slice_count += 1
                        tail_ctx = assistant_so_far[-1600:] if len(assistant_so_far) > 1600 else assistant_so_far
                        payload["messages"] = list(messages) + [
                            {"role": "assistant", "content": tail_ctx},
                            {
                                "role": "user",
                                "content": (
                                    "【微切片閉環無縫續傳指令】：請緊接著上方你最後輸出的字元『"
                                    + assistant_so_far[-40:]
                                    + "』直接繼續往下輸出後續內容（嚴禁重複已輸出的前文、嚴禁加上任何開場白或解釋）："
                                ),
                            },
                        ]
                        continue
                    break

                completion_tokens = max(completion_tokens, chunk_steps)
                elapsed = max(0.001, time.time() - t0)
                tps = completion_tokens / elapsed
                saved_arts = eng.harness.extract_and_save_artifacts(accumulated_text)

                st.last_stats = {
                    "tps": round(tps, 2),
                    "ttft_ms": round(ttft_ms, 1),
                    "tokens": completion_tokens,
                    "elapsed": round(elapsed, 2),
                    "entropy": round(entropy_sum / max(1, chunk_steps), 3),
                }

                st.session_logger.append_turn(
                    user_prompt=user_prompt,
                    assistant_reply=accumulated_text,
                    stats={
                        "tokens": completion_tokens,
                        "tps": tps,
                        "elapsed": elapsed,
                        "ttft_ms": ttft_ms,
                        "top_k_experts": last_experts if st.lora_enabled else [],
                        "top_k_weights": last_weights if st.lora_enabled else [],
                        "mean_entropy": entropy_sum / max(1, chunk_steps),
                        "backend": "web-ui-hybrid" if st.lora_enabled else "web-ui-vanilla-bypass",
                        "nrem_events": eng.total_nrem_sleep_events,
                        "gpu_24_prunes": eng.total_gpu_24_prunes,
                        "bifurcation_flips": getattr(eng, "_turn_bifurcation_flips", 0),
                        "total_bifurcation_flips": getattr(eng, "total_bifurcation_flips", 0),
                        "expert_states": eng.rust_router.get_expert_states() if eng.rust_router is not None else [],
                    },
                    artifacts=saved_arts,
                )

                st.conversation_history.append({"role": "user", "content": user_prompt})
                st.conversation_history.append({"role": "assistant", "content": accumulated_text})

                if st.wm_mgr and st.harness:
                    st.wm_mgr.extract_and_update_from_turn(
                        user_prompt=user_prompt,
                        assistant_reply=accumulated_text,
                        saved_artifacts=saved_arts,
                        sandbox_dir=st.harness.isolated_output_dir,
                    )
                if st.lora_enabled and st.ckpt_mgr and st.rust_router:
                    st.ckpt_mgr.save_state(st.rust_router, expert_pool=st.expert_pool, tokens_generated=completion_tokens)
                if st.qt_win is not None and hasattr(st.qt_win, "sig_web_append_chat"):
                    try:
                        st.qt_win.sig_web_append_chat.emit(
                            f"\n----------------------------------------------------------------------\n"
                            f"🏁 [Web UI Turn Complete] Tokens: {completion_tokens} | Speed: {tps:.2f} tok/s | TTFT: {ttft_ms:.1f} ms | Elapsed: {elapsed:.2f}s"
                        )
                    except Exception:
                        pass

                st.is_generating = False
                send_sse("done", {
                    "stats": st.last_stats,
                    "telemetry": st.get_telemetry_snapshot(include_magnifier=include_mag),
                    "session_file": os.path.basename(st.session_logger.md_file),
                })
            except Exception as e:
                st.is_generating = False
                send_sse("error", {"message": str(e)})
            finally:
                st.is_generating = False
                st.infer_lock.release()
            return

        if path in ("/v1/cache/clear", "/api/cache/clear"):
            body = self._read_json_body()
            clear_hist = bool(body.get("clear_history", False))
            with st.lock:
                if st.hybrid_engine is not None:
                    st.hybrid_engine.dynamic_logit_bias.clear()
                    # Send a 1-token reuse_context=False probe to Thor C++ server to flush GPU prefix KV-cache
                    try:
                        flush_payload = {
                            "model": st.hybrid_engine.model_id,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                            "stream": False,
                            "reuse_context": False,
                            "cache_generated_tokens": False,
                        }
                        req = urllib.request.Request(
                            f"{st.hybrid_engine.server_url}/v1/chat/completions",
                            data=json.dumps(flush_payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=10.0) as _:
                            pass
                    except Exception:
                        pass
                if clear_hist:
                    st.conversation_history.clear()
            self._send_json({"ok": True, "kv_cache_flushed": True, "history_cleared": clear_hist})
            return

        if path in ("/v1/chat/completions", "/api/v1/chat/completions"):
            self._handle_openai_chat_completions(st)
            return

        self._send_json({"error": f"Unknown POST path: {path}"}, 404)

    def _handle_openai_chat_completions(self, st: LocalSleepWebState) -> None:
        """
        Standard OpenAI-compatible `/v1/chat/completions` endpoint with integrated
        `rust_localsleep-moe` 8/16-LoRA routing, dynamic logit bias / VRAM live gate,
        Hebbian plasticity, NREM sleep consolidation, and `sessions/` persistence.
        Supports both `stream: true` (SSE) and `stream: false` (JSON).
        """
        body = self._read_json_body()
        req_messages = body.get("messages") or []
        if not req_messages:
            self._send_json({"error": {"message": "messages array is required", "type": "invalid_request_error"}}, 400)
            return
        if st.qt_win is not None and getattr(st.qt_win, "worker", None) is not None and st.qt_win.worker.isRunning():
            self._send_json({"error": {"message": "PyQt GUI inference is currently running", "type": "server_busy"}}, 409)
            return
        if not st.infer_lock.acquire(blocking=False):
            self._send_json({"error": {"message": "Inference already running", "type": "server_busy"}}, 409)
            return

        is_stream = bool(body.get("stream", False))
        req_model = str(body.get("model") or "localsleep-moe")
        force_bypass = req_model.endswith("-vanilla") or bool(body.get("bypass_lora", False))
        effective_lora = bool(st.lora_enabled) and not force_bypass

        completion_id = f"chatcmpl-lsmoe-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

        def write_oai_sse(chunk_dict: Dict[str, Any]) -> bool:
            try:
                raw_sse = f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                self.wfile.write(raw_sse.encode("utf-8"))
                self.wfile.flush()
                return True
            except Exception:
                st.abort_flag = True
                return False

        try:
            st.abort_flag = False
            st.is_generating = True
            st.apply_lora_and_loop_mode()
            eng = st.hybrid_engine
            assert eng is not None and st.session_logger is not None
            prev_bypass = eng.bypass_lora
            eng.bypass_lora = not effective_lora

            # Extract last user prompt for SessionLogger
            user_prompt_text = ""
            for m in reversed(req_messages):
                if m.get("role") == "user":
                    c = m.get("content", "")
                    user_prompt_text = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                    break

            has_sys = any(m.get("role") == "system" for m in req_messages)
            full_messages = list(req_messages) if has_sys else ([{"role": "system", "content": st.system_prompt}] + list(req_messages))
            full_messages = eng._compress_swa_messages(full_messages)

            req_max = int(body.get("max_tokens") or body.get("max_completion_tokens") or 0)
            # Dynamic Context Ceiling Guard: ensure (input_tokens_est + max_toks) never overflows Thor's 20K/32K KV buffer
            est_input_chars = sum(len(str(m.get("content", ""))) for m in full_messages)
            est_input_toks = int(est_input_chars / 1.5) + 256
            safe_ceiling = max(4096, 20000 - est_input_toks)
            raw_target_max = req_max if (req_max > 0 and body.get("strict_max_tokens")) else max(req_max, int(eng.max_tokens))
            max_toks = min(raw_target_max, safe_ceiling)
            temp = float(body.get("temperature") if body.get("temperature") is not None else eng.temperature)
            top_p = float(body.get("top_p") if body.get("top_p") is not None else eng.top_p)

            reset_kv = bool(body.get("reset_kv_cache", False))
            upstream_payload: Dict[str, Any] = {
                "model": eng.model_id,
                "messages": full_messages,
                "max_tokens": max_toks,
                "temperature": temp,
                "top_p": top_p,
                "stop": body.get("stop") or eng.stop_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "logprobs": True,
                "top_logprobs": 4,
                "enable_thinking": bool(body.get("enable_thinking", False)),
                "reuse_context": not reset_kv,
                "cache_generated_tokens": not reset_kv,
            }
            if effective_lora and eng.dynamic_logit_bias:
                upstream_payload["logit_bias"] = dict(list(eng.dynamic_logit_bias.items())[:64])

            if effective_lora and eng.expert_pool is not None and hasattr(eng.expert_pool, "begin_eligibility_episode"):
                eng.expert_pool.begin_eligibility_episode()
            eng._turn_bifurcation_flips = 0

            if st.qt_win is not None and hasattr(st.qt_win, "sig_web_append_chat"):
                try:
                    st.qt_win.sig_web_append_chat.emit(
                        f"\n🔌 [OpenAI API ({'LoRA-ON' if effective_lora else 'Bypass'})] User >> {user_prompt_text[:240]}\nKuro ({eng.model_id}) >> "
                    )
                except Exception:
                    pass

            cjk_cleaner = CJKStreamCleaner()
            accumulated_text = ""
            chunk_steps = 0
            completion_tokens = 0
            prompt_tokens = 0
            ttft_ms = 0.0
            t0 = time.time()
            first_token_time = None
            last_weights: List[float] = [0.5, 0.5]
            last_experts: List[int] = [0, 1]
            entropy_sum = 0.0
            finish_reason = "stop"

            req = urllib.request.Request(
                f"{eng.server_url}/v1/chat/completions",
                data=json.dumps(upstream_payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                for raw_line in resp:
                    if st.abort_flag:
                        finish_reason = "stop"
                        break
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    if chunk.get("usage"):
                        u = chunk["usage"]
                        completion_tokens = max(completion_tokens, int(u.get("completion_tokens", 0)))
                        prompt_tokens = max(prompt_tokens, int(u.get("prompt_tokens", 0)))

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice0 = choices[0]
                    if choice0.get("finish_reason"):
                        finish_reason = str(choice0.get("finish_reason"))
                    delta = choice0.get("delta", {})
                    content = delta.get("content")
                    lp_info = choice0.get("logprobs") or {}
                    lp_content = lp_info.get("content") or []

                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft_ms = (first_token_time - t0) * 1000.0

                        cleaned_piece = cjk_cleaner.clean(content)
                        accumulated_text += cleaned_piece

                        if lp_content:
                            for idx_lp, entry in enumerate(lp_content):
                                tid = int(entry.get("token_id") or 0)
                                top_lps = entry.get("top_logprobs") or []
                                top_tids = [int(x.get("token_id") or 0) for x in top_lps] or [tid]
                                top_vals = [float(x.get("logprob") or -1.0) for x in top_lps] or [float(entry.get("logprob") or -0.1)]
                                top_probs = [math.exp(max(-20.0, min(0.0, v))) for v in top_vals]
                                tok_str_rec = str(entry.get("token") or entry.get("piece") or (cleaned_piece if idx_lp == 0 else ""))
                                w, sel, ent, _ = eng._step_local_rust_moe(tid, top_tids, top_vals, token_piece=tok_str_rec)
                                last_weights, last_experts = w, sel
                                st.active_routed = list(sel)
                                entropy_sum += ent
                                chunk_steps += 1
                                cur_el = max(0.001, time.time() - t0)
                                st.session_logger.record_token_telemetry(
                                    token_id=tid,
                                    token_str=tok_str_rec,
                                    tps=chunk_steps / cur_el,
                                    entropy=ent,
                                    selected_experts=sel,
                                    weights=w,
                                    expert_states=eng.rust_router.get_expert_states() if eng.rust_router else [],
                                    top_probs=top_probs,
                                )
                        else:
                            synth_tid = sum(ord(c) * (i + 1) for i, c in enumerate(content)) % 32000
                            w, sel, ent, _ = eng._step_local_rust_moe(synth_tid, [synth_tid], [-0.15], token_piece=cleaned_piece)
                            last_weights, last_experts = w, sel
                            st.active_routed = list(sel)
                            entropy_sum += ent
                            chunk_steps += 1
                            cur_el = max(0.001, time.time() - t0)
                            st.session_logger.record_token_telemetry(
                                token_id=synth_tid,
                                token_str=cleaned_piece,
                                tps=chunk_steps / cur_el,
                                entropy=ent,
                                selected_experts=sel,
                                weights=w,
                                expert_states=eng.rust_router.get_expert_states() if eng.rust_router else [],
                                top_probs=[0.85],
                            )

                        cur_el = max(0.001, time.time() - t0)
                        st.last_stats = {
                            "tps": round(chunk_steps / cur_el, 2),
                            "ttft_ms": round(ttft_ms, 1),
                            "tokens": chunk_steps,
                            "elapsed": round(cur_el, 2),
                            "entropy": round(entropy_sum / max(1, chunk_steps), 3),
                        }
                        if st.qt_win is not None and hasattr(st.qt_win, "sig_web_stream_piece"):
                            try:
                                st.qt_win.sig_web_stream_piece.emit(
                                    cleaned_piece,
                                    int(chunk_steps),
                                    float(st.last_stats["entropy"]),
                                    list(last_experts) if effective_lora else [],
                                    [float(x) for x in last_weights],
                                )
                            except Exception:
                                pass

                        if is_stream:
                            oai_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": cleaned_piece},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            if not write_oai_sse(oai_chunk):
                                break

            completion_tokens = max(completion_tokens, chunk_steps)
            elapsed = max(0.001, time.time() - t0)
            tps = completion_tokens / elapsed
            mean_ent = entropy_sum / max(1, chunk_steps)
            st.last_stats = {
                "tps": round(tps, 2),
                "ttft_ms": round(ttft_ms, 1),
                "tokens": completion_tokens,
                "elapsed": round(elapsed, 2),
                "entropy": round(mean_ent, 3),
            }

            st.session_logger.append_turn(
                user_prompt=user_prompt_text,
                assistant_reply=accumulated_text,
                stats={
                    "tokens": completion_tokens,
                    "tps": tps,
                    "elapsed": elapsed,
                    "ttft_ms": ttft_ms,
                    "top_k_experts": last_experts if effective_lora else [],
                    "top_k_weights": last_weights if effective_lora else [],
                    "mean_entropy": mean_ent,
                    "backend": "openai-api-hybrid" if effective_lora else "openai-api-vanilla",
                    "nrem_events": eng.total_nrem_sleep_events,
                    "gpu_24_prunes": eng.total_gpu_24_prunes,
                    "bifurcation_flips": getattr(eng, "_turn_bifurcation_flips", 0),
                    "total_bifurcation_flips": getattr(eng, "total_bifurcation_flips", 0),
                    "expert_states": eng.rust_router.get_expert_states() if eng.rust_router is not None else [],
                },
                artifacts=[],
            )
            if not body.get("stateless_forge", False):
                st.conversation_history.append({"role": "user", "content": user_prompt_text})
                st.conversation_history.append({"role": "assistant", "content": accumulated_text})
                if len(st.conversation_history) > 20:
                    del st.conversation_history[:-20]
            if effective_lora and st.ckpt_mgr and st.rust_router:
                st.ckpt_mgr.save_state(st.rust_router, expert_pool=st.expert_pool, tokens_generated=completion_tokens)
            eng.bypass_lora = prev_bypass

            if st.qt_win is not None and hasattr(st.qt_win, "sig_web_append_chat"):
                try:
                    st.qt_win.sig_web_append_chat.emit(
                        f"\n🏁 [OpenAI API Complete] Tokens: {completion_tokens} | Speed: {tps:.2f} tok/s | TTFT: {ttft_ms:.1f} ms"
                    )
                except Exception:
                    pass

            if is_stream:
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                write_oai_sse(final_chunk)
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
            else:
                resp_obj = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": accumulated_text},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "localsleep_telemetry": {
                        "lora_enabled": effective_lora,
                        "loaded_cube_id": str(getattr(st.rust_router, "loaded_cube_id", "gen_000_origin")),
                        "tps": round(tps, 2),
                        "ttft_ms": round(ttft_ms, 1),
                        "mean_entropy": round(mean_ent, 4),
                        "top_k_experts": last_experts if effective_lora else [],
                        "top_k_weights": [round(float(x), 4) for x in last_weights] if effective_lora else [],
                        "session_file": os.path.basename(st.session_logger.md_file),
                    },
                }
                self._send_json(resp_obj, 200)
        except Exception as e:
            if is_stream:
                write_oai_sse({"error": {"message": str(e), "type": "internal_error"}})
            else:
                self._send_json({"error": {"message": str(e), "type": "internal_error"}}, 500)
        finally:
            st.is_generating = False
            st.infer_lock.release()


def start_embedded_web_server(
    qt_win: Any,
    host: str = "0.0.0.0",
    port: int = 18935,
) -> Optional[LocalSleepWebState]:
    """
    Launches the LocalSleep-MoE Web UI server in a background daemon thread,
    sharing the exact same engine, router, LoRA pool, and SessionLogger as `qt_win`.
    """
    global GLOBAL_STATE
    try:
        GLOBAL_STATE = LocalSleepWebState(qt_win=qt_win)
        srv = ThreadingHTTPServer((host, port), LocalSleepWebHandler)
        srv.daemon_threads = True
        t = threading.Thread(target=srv.serve_forever, name="LocalSleepWebUIDaemon", daemon=True)
        t.start()
        print(f"🌐 [Dual-Mode Web UI] Embedded server online at http://{host}:{port} (shared with PyQt6 GUI)")
        return GLOBAL_STATE
    except Exception as e:
        print(f"⚠️ [Dual-Mode Web UI] Failed to bind port {port}: {e}")
        return None


def main() -> None:
    global GLOBAL_STATE
    parser = argparse.ArgumentParser(description="LocalSleep-MoE Remote Web UI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0 for remote access)")
    parser.add_argument("--port", type=int, default=18935, help="Bind port (default: 18935)")
    args = parser.parse_args()

    GLOBAL_STATE = LocalSleepWebState()
    srv = ThreadingHTTPServer((args.host, args.port), LocalSleepWebHandler)
    print(f"🐈‍⬛ [LocalSleep-MoE Web UI] Listening on http://{args.host}:{args.port}")
    print(f"   • Active Profile : {GLOBAL_STATE.profile_name} ({GLOBAL_STATE.model_id} @ {GLOBAL_STATE.server_url})")
    print(f"   • Active Core    : {getattr(GLOBAL_STATE.rust_router, 'loaded_cube_id', 'gen_000_origin')}")
    print(f"   • Sessions Log   : {GLOBAL_STATE.session_logger.md_file if GLOBAL_STATE.session_logger else 'N/A'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down LocalSleep-MoE Web UI...")
        srv.server_close()


if __name__ == "__main__":
    main()
