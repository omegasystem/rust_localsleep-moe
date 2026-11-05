#!/usr/bin/env python3
"""
ls_moe_pyqt_gui.py - Native PyQt6 Desktop GUI & Direct VRAM Synaptic Scope for `rust_localsleep-moe`
====================================================================================================
Zero-Browser, Zero-WebGL GPU Conflict (Never lags YouTube!).
Runs natively via WSLg / X11 using `PyQt6` + PyTorch GPU-to-RGBA Zero-Allocation Rasterization:
  - Left Panel (Tactical Command & Agentic Chat Console):
      * Profile Selector (`thor_31b`, `rtx5080_bolt_e4b_gguf`, `thor_31bnvfp4`, etc.)
      * Editable Server URL (`http://192.168.50.67:8000` or `http://127.0.0.1:8001`), Model ID
      * Sliders / SpinBoxes for `Max Tokens`, `Temperature`, `Top-P`, `Fatigue Threshold`, `SHY Lambda`
      * One-Click Buttons:
          - `[🔥 點火連線 / 套用參數]`
          - `[📜 載入上一輪對話 (--resume-session)]`
          - `[🔄 重置滿血突觸 (--reset-checkpoint)]`
          - `[💾 立即儲存 SafeTensors 快照]`
      * Real-Time Streaming Chat Log + Prompt Input Box
  - Right Top Panel (🪰 Fly-Brain Direct VRAM Synaptic Raster Scope - 60 FPS):
      * Renders all 8 LoRA Experts' live GPU weights (`expert_pool.lora_a`) directly into an RGB image buffer!
      * Active Top-2 routed experts flash with Gold/Cyan Hebbian Potentiation waves along active channels!
      * NREM Sleeping experts pulse in Deep Violet with visible 2:4 structural sparsity checkerboards!
  - Right Bottom Panel (🌊 3D Isometric 8-Channel EEG Waterfall & CRT Oscilloscope):
      * Renders a 3D isometric multi-ridge EEG brainwave waterfall (`Weight * Entropy` & `||LoRA_A||_F`)
        plus real-time Fatigue threshold bars without touching browser WebGL!
"""

import datetime
import json
import math
import os
import sys
import threading
import time
import urllib.request
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
import shutil
from thor_vision_scanner import (
    DIRECTIONAL_PROMPTS,
    encode_image_data_url,
    generate_synthetic_directional_suite,
    parse_and_draw_bboxes,
)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from checkpoint_manager import EmbodiedCheckpointManager
from cjk_cleaner import CJKStreamCleaner
from expert_pool_lora import DynamicLoRAExpertPool
from generation_engine_trt_hybrid import (
    FallbackFatigueRouter,
    TrtEdgeLlmHybridEngine,
    apply_pytorch_24_fisher_pruning,
)
from ls_moe_agentic import init_rust_router_and_pool, load_profile_config
from session_logger import SessionLogger
from tool_parser import ToolCallParser
from tools_harness import EmbodiedToolHarness
from working_memory_manager import WorkingMemoryManager
from hypercube_replicator import HypercubeReplicatorManager, HypercubeTaskMonitorWindow


EXP_HEX_COLORS = [
    "#38bdf8",  # E00 Cyan
    "#f43f5e",  # E01 Rose
    "#fbbf24",  # E02 Amber
    "#34d399",  # E03 Emerald
    "#a855f7",  # E04 Purple
    "#fb923c",  # E05 Orange
    "#2dd4bf",  # E06 Teal
    "#e879f9",  # E07 Fuchsia
    "#60a5fa",  # E08 Blue
    "#f87171",  # E09 Coral
    "#facc15",  # E10 Yellow
    "#4ade80",  # E11 Green
    "#c084fc",  # E12 Lavender
    "#fb7185",  # E13 Pink
    "#22d3ee",  # E14 Aqua
    "#a3e635",  # E15 Lime
]


class MultiLinePromptInput(QPlainTextEdit):
    """Multi-line prompt editor: Enter (or Ctrl+Enter) sends; Shift+Enter inserts newline."""
    sig_submit = pyqtSignal()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.sig_submit.emit()
            return
        super().keyPressEvent(event)


class DummyArgs:
    def __init__(self, profile: str, max_tokens: int, temp: float, top_p: float):
        self.profile = profile
        self.max_tokens = max_tokens
        self.max_context_tokens = 32768
        self.temp = temp
        self.top_p = top_p
        self.reset_checkpoint = False


# ============================================================================
# 1. Background Streaming Inference Worker Thread (Zero UI Freeze)
# ============================================================================
class StreamInferenceWorker(QThread):
    sig_token = pyqtSignal(str, int, float, list, list)  # piece, tid, entropy, sel_exps, weights
    sig_status = pyqtSignal(str)
    sig_finished = pyqtSignal(str, int, float, float)  # full_text, total_toks, elapsed, ttft_ms
    sig_error = pyqtSignal(str)

    def __init__(
        self,
        engine: TrtEdgeLlmHybridEngine,
        session_logger: SessionLogger,
        user_prompt: str,
        system_inst: str,
        conversation_history: List[Dict[str, str]],
        attached_images: Optional[List[Path]] = None,
    ):
        super().__init__()
        self.engine = engine
        self.session_logger = session_logger
        self.user_prompt = user_prompt
        self.system_inst = system_inst
        self.conversation_history = list(conversation_history)
        self.attached_images = list(attached_images or [])
        self._abort = False

    def request_abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_inst}]
            if self.conversation_history:
                messages.extend(self.conversation_history)
            if self.attached_images:
                u_content: List[Dict[str, Any]] = []
                for idx, img_p in enumerate(self.attached_images, start=1):
                    data_url, _ = encode_image_data_url(img_p)
                    if len(self.attached_images) > 1:
                        u_content.append({"type": "text", "text": f"[Image Frame #{idx}: {img_p.name}]"})
                    u_content.append({"type": "image_url", "image_url": {"url": data_url}})
                u_content.append({"type": "text", "text": self.user_prompt})
                messages.append({"role": "user", "content": u_content})
            else:
                messages.append({"role": "user", "content": self.user_prompt})
            messages = self.engine._compress_swa_messages(messages)

            payload: Dict[str, Any] = {
                "model": self.engine.model_id,
                "messages": messages,
                "max_tokens": self.engine.max_tokens,
                "temperature": self.engine.temperature,
                "top_p": self.engine.top_p,
                "stop": self.engine.stop_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "logprobs": True,
                "top_logprobs": 4,
                "enable_thinking": False,
            }
            if self.engine.dynamic_logit_bias:
                payload["logit_bias"] = dict(list(self.engine.dynamic_logit_bias.items())[:64])

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
            MAX_TOOL_CALLS = int(getattr(self.engine, "max_tool_calls", 100) or 100)
            consecutive_same_sig = 0
            last_tool_sig = ""
            micro_chunk_limit = int(getattr(self.engine, "micro_chunk_tokens", 0) or 0)
            micro_slice_count = 0

            if self.engine.expert_pool is not None and hasattr(self.engine.expert_pool, "begin_eligibility_episode"):
                self.engine.expert_pool.begin_eligibility_episode()
            self.engine._turn_bifurcation_flips = 0

            while True:
                remaining_budget = max(16, self.engine.max_tokens - completion_tokens)
                eff_max_toks = min(micro_chunk_limit, remaining_budget) if micro_chunk_limit > 0 else remaining_budget
                payload["max_tokens"] = eff_max_toks
                payload["reuse_context"] = True
                payload["cache_generated_tokens"] = True
                if self.engine.dynamic_logit_bias:
                    payload["logit_bias"] = dict(list(self.engine.dynamic_logit_bias.items())[:64])

                req = urllib.request.Request(
                    f"{self.engine.server_url}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    method="POST",
                )
                seg_text = ""
                intercepted_tool = None
                finish_reason = None

                with urllib.request.urlopen(req, timeout=180.0) as resp:
                    for raw_line in resp:
                        if self._abort:
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
                                    w, sel, ent, _ = self.engine._step_local_rust_moe(tid, top_tids, top_vals, token_piece=tok_str_rec)
                                    last_weights, last_experts = w, sel
                                    entropy_sum += ent
                                    chunk_steps += 1
                                    cur_el = max(0.001, time.time() - t0)
                                    self.session_logger.record_token_telemetry(
                                        token_id=tid,
                                        token_str=tok_str_rec,
                                        tps=chunk_steps / cur_el,
                                        entropy=ent,
                                        selected_experts=sel,
                                        weights=w,
                                        expert_states=self.engine.rust_router.get_expert_states(),
                                        top_probs=top_probs,
                                    )
                                    ui_piece = cleaned_piece if idx_lp == 0 else ""
                                    self.sig_token.emit(ui_piece, tid, float(ent), list(sel), list(w))
                            else:
                                synth_tid = sum(ord(c) * (i + 1) for i, c in enumerate(content)) % 32000
                                w, sel, ent, _ = self.engine._step_local_rust_moe(synth_tid, [synth_tid], [-0.15], token_piece=cleaned_piece)
                                last_weights, last_experts = w, sel
                                entropy_sum += ent
                                chunk_steps += 1
                                cur_el = max(0.001, time.time() - t0)
                                self.session_logger.record_token_telemetry(
                                    token_id=synth_tid,
                                    token_str=cleaned_piece,
                                    tps=chunk_steps / cur_el,
                                    entropy=ent,
                                    selected_experts=sel,
                                    weights=w,
                                    expert_states=self.engine.rust_router.get_expert_states(),
                                    top_probs=[0.85],
                                )
                                self.sig_token.emit(cleaned_piece, synth_tid, float(ent), list(sel), list(w))

                            if ToolCallParser.has_complete_tool_call(seg_text):
                                did_fix, seg_text = ToolCallParser.repair_missing_closing_angle(seg_text)
                                if did_fix:
                                    accumulated_text += ">"
                                    self.sig_token.emit(">", 0, 0.2, last_experts, last_weights)
                                # Flush any <file> blocks generated in this pass to disk BEFORE executing run_sandbox_shell!
                                self.engine.harness.extract_and_save_artifacts(seg_text)
                                act, out, clean_prefix = ToolCallParser.detect_and_execute(seg_text, self.engine.harness)
                                if act and out is not None:
                                    intercepted_tool = (act, str(out), clean_prefix)
                                    break

                if self._abort:
                    break

                if intercepted_tool is not None:
                    act, tool_output, clean_prefix = intercepted_tool
                    tc_idx = seg_text.find("<tool_call>")
                    raw_tc_block = seg_text[tc_idx:] if tc_idx != -1 else f"<tool_call>{json.dumps({'action': act}, ensure_ascii=False)}</tool_call>"
                    if not raw_tc_block.endswith(">"):
                        raw_tc_block += ">"

                    sig = f"{act}:{raw_tc_block}:{len(tool_output)}:{tool_output[-120:]}"
                    if sig == last_tool_sig:
                        consecutive_same_sig += 1
                    else:
                        consecutive_same_sig = 1
                        last_tool_sig = sig

                    if consecutive_same_sig >= 3 or tool_calls_count >= MAX_TOOL_CALLS:
                        break
                    tool_calls_count += 1

                    if len(tool_output) > 1800:
                        total_lines_out = tool_output.count("\n")
                        ui_preview = (
                            f"{tool_output[:1200]}\n"
                            f"   ... [UI 畫面折疊預覽：已將完整 {total_lines_out} 行 ({len(tool_output):,} chars) 100% 餵入 31B 模型 Context！] ...\n"
                            f"{tool_output[-600:]}"
                        )
                    else:
                        ui_preview = tool_output

                    elig_note = ""
                    if self.engine.expert_pool is not None and hasattr(self.engine.expert_pool, "settle_eligibility_trace"):
                        if "EXIT_CODE: 0" in str(tool_output):
                            st_tag = self.engine.expert_pool.settle_eligibility_trace(sandbox_passed=True)
                            elig_note = f"\n🧬 [Eligibility Trace -> {st_tag}] 沙盒驗證通過，短期赫布跡象已固化為長期突觸記憶！"
                            self.sig_status.emit(f"🧬 [資格跡固化 {st_tag}] EXIT_CODE=0，已將短期赫布跡象鎖定寫入長期 LoRA 權重！")
                        elif "EXIT_CODE:" in str(tool_output):
                            st_tag = self.engine.expert_pool.settle_eligibility_trace(sandbox_passed=False)
                            elig_note = f"\n🛡️ [Eligibility Trace -> {st_tag}] 攔截沙盒失敗路徑，已執行反赫布回滾以保護長期突觸快照！"
                            self.sig_status.emit(f"🛡️ [資格跡回滾 {st_tag}] EXIT_CODE!=0，已退回 LTM 基準並對錯誤路徑執行 Anti-Hebbian 抑制！")

                    tool_banner = f"\n\n📥 [Sandbox Tool Executed (#{tool_calls_count}: {act})]:\n{ui_preview}{elig_note}\n⚡ [Kuro Post-Tool Stream]: "
                    self.sig_token.emit(tool_banner, 0, 0.3, last_experts, last_weights)
                    accumulated_text += tool_banner

                    messages.append({"role": "assistant", "content": (clean_prefix + "\n" + raw_tc_block).strip()})
                    messages.append({
                        "role": "user",
                        "content": f"【環境工具回傳結果 (System Tool Response)】：\n{tool_output}\n請根據上述真實工具回傳結果繼續回答主人。",
                    })
                    messages = self.engine._compress_swa_messages(messages)
                    payload["messages"] = messages
                    assistant_so_far = ""
                    continue

                # Option 2: KV-Cache Micro-Chunk Closed-Loop Continuation!
                # If micro_chunk_limit > 0 and this slice stopped due to hitting eff_max_toks ("length"),
                # we immediately inject the freshly computed 16-LoRA dynamic_logit_bias and continue seamlessly!
                if (
                    micro_chunk_limit > 0
                    and (finish_reason == "length" or (seg_text and len(seg_text) >= 8 and finish_reason is None))
                    and completion_tokens < self.engine.max_tokens
                    and micro_slice_count < 256
                ):
                    micro_slice_count += 1
                    bias_cnt = len(self.engine.dynamic_logit_bias)
                    self.sig_status.emit(
                        f"⚡ [方案2 微切片閉環 #{micro_slice_count}] 已將 16-LoRA 最新 {bias_cnt} 筆突觸 Logit Bias 注入 TensorRT (Prefix KV Cache 續傳中...)"
                    )
                    tail_ctx = assistant_so_far[-1600:] if len(assistant_so_far) > 1600 else assistant_so_far
                    cont_messages = list(messages) + [
                        {"role": "assistant", "content": tail_ctx},
                        {
                            "role": "user",
                            "content": (
                                "【微切片閉環無縫續傳指令】：請緊接著上方你最後輸出的字元『"
                                + assistant_so_far[-40:]
                                + "』直接繼續往下輸出後續內容（嚴禁重複已輸出的前文、嚴禁加上任何開場白或解釋，直接輸出下一個字元）："
                            ),
                        },
                    ]
                    payload["messages"] = cont_messages
                    continue
                break

            # When a mid-stream <tool_call> breaks early before the final [DONE] usage chunk,
            # completion_tokens only received the post-tool pass usage; chunk_steps holds the true total tokens!
            completion_tokens = max(completion_tokens, chunk_steps)

            elapsed = max(0.001, time.time() - t0)
            tps = completion_tokens / elapsed
            saved_arts = self.engine.harness.extract_and_save_artifacts(accumulated_text)

            self.session_logger.append_turn(
                user_prompt=self.user_prompt,
                assistant_reply=accumulated_text,
                stats={
                    "tokens": completion_tokens,
                    "tps": tps,
                    "elapsed": elapsed,
                    "ttft_ms": ttft_ms,
                    "top_k_experts": last_experts,
                    "top_k_weights": last_weights,
                    "mean_entropy": entropy_sum / max(1, chunk_steps),
                    "backend": "pyqt6-gui-hybrid",
                    "nrem_events": self.engine.total_nrem_sleep_events,
                    "gpu_24_prunes": self.engine.total_gpu_24_prunes,
                    "bifurcation_flips": getattr(self.engine, "_turn_bifurcation_flips", 0),
                    "total_bifurcation_flips": getattr(self.engine, "total_bifurcation_flips", 0),
                    "expert_states": self.engine.rust_router.get_expert_states() if self.engine.rust_router is not None else [],
                },
                artifacts=saved_arts,
            )
            self.sig_finished.emit(accumulated_text, completion_tokens, elapsed, ttft_ms)
        except Exception as e:
            self.sig_error.emit(str(e))


# ============================================================================
# 2. Fly-Brain Direct VRAM Synaptic Raster Scope (Top-Right Canvas, 60 FPS)
# ============================================================================
class FlyBrainVramScope(QWidget):
    """
    Directly projects all 8 LoRA Experts' live GPU tensors (`expert_pool.lora_a`)
    into a 4x2 panoramic neural cortex raster image in VRAM/RAM at 60 FPS!
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(330)
        self.expert_pool: Optional[DynamicLoRAExpertPool] = None
        self.rust_router: Any = None
        self.active_routed: List[int] = [0, 1]
        self.last_token_id: int = 0
        self.last_entropy: float = 0.5
        self.pulse_phase: float = 0.0
        self._vram_dirty: bool = True

        # Preallocated 2:4 Microscopic Block Matrix for each of the 8 experts:
        # Directly slices `lora_a[0:12, 0:24]` from GPU VRAM -> 6 Quads x 4 cols = 24 cols, 12 rows!
        self.grid_rows = 12
        self.grid_cols = 24  # 6 Quads of 4 elements each!
        self.num_experts = 8
        self.grid_blocks: List[Optional[np.ndarray]] = [None] * 16
        self.exp_norms: List[float] = [2.60] * 16
        self.exp_states: List[Tuple] = []

    def bind_engine(self, rust_router: Any, expert_pool: DynamicLoRAExpertPool) -> None:
        self.rust_router = rust_router
        self.expert_pool = expert_pool
        self.num_experts = int(getattr(rust_router, "num_experts", getattr(expert_pool, "num_experts", 8)))
        self.grid_blocks = [None] * self.num_experts
        self.exp_norms = [2.60] * self.num_experts
        self._vram_dirty = True
        self.refresh_vram_textures(force_gpu_sync=True)

    def notify_token_activity(self, tid: int, entropy: float, sel_exps: List[int]) -> None:
        self.last_token_id = tid
        self.last_entropy = entropy
        self.active_routed = list(sel_exps)
        self._vram_dirty = True

    def refresh_vram_textures(self, force_gpu_sync: bool = False) -> None:
        if self.expert_pool is None:
            return
        self.pulse_phase += 0.18
        if self.rust_router is not None and hasattr(self.rust_router, "get_expert_states"):
            try:
                self.exp_states = self.rust_router.get_expert_states()
            except Exception:
                pass

        # Zero-Idle-GPU Optimization:
        # Only perform `cudaMemcpyDtoH` (.cpu().numpy()) when weights actually changed (`_vram_dirty` or `force_gpu_sync`)!
        # When idle (no inference running), reuse the cached numpy blocks so RTX 5080 stays at 0% GPU utilization!
        if self._vram_dirty or force_gpu_sync:
            with torch.no_grad():
                for i in range(self.num_experts):
                    key = f"exp_{i}"
                    if key not in self.expert_pool.lora_a:
                        continue
                    la = self.expert_pool.lora_a[key].data  # [rank=32, hidden_dim=5376] on GPU!
                    norm_val = float(torch.norm(la.float()).item())
                    self.exp_norms[i] = norm_val

                    # Directly slice microscopic 12x24 block (6 Quads x 4 elements) from GPU VRAM!
                    sub = la[: self.grid_rows, : self.grid_cols].float().cpu().numpy()
                    self.grid_blocks[i] = sub
            self._vram_dirty = False
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(9, 13, 22))

        n_exp = max(1, self.num_experts)
        cube_tag = getattr(self.rust_router, "loaded_cube_id", "gen_000_origin")
        # Title Header
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.setFont(QFont("JetBrains Mono", 10, QFont.Weight.Bold))
        painter.drawText(
            14,
            21,
            f"🔬 [顯微級 2:4 結構化稀疏矩陣 ({n_exp}-LoRA Experts | 引擎: {cube_tag} | 白線 = 4 元素 Quad 邊界)]",
        )

        # Dynamically layout 4x2 (8-LoRA) or 4x4 (16-LoRA) Grid of Expert Microscopic 2:4 Block Matrices
        pad_x, pad_y = 14, 28
        cols = 4
        rows = max(2, math.ceil(n_exp / cols))
        gap_y = 6 if rows >= 4 else 10
        cell_w = (w - pad_x * 2 - 30) // cols
        cell_h = (h - pad_y - 68 - gap_y * (rows - 1)) // rows

        for i in range(n_exp):
            c = i % cols
            r = i // cols
            bx = pad_x + c * (cell_w + 10)
            by = pad_y + r * (cell_h + gap_y)

            st_code = self.exp_states[i][1] if i < len(self.exp_states) else 0
            fat_val = self.exp_states[i][2] if i < len(self.exp_states) else 0.0
            tok_val = self.exp_states[i][3] if i < len(self.exp_states) else 0
            slp_cyc = self.exp_states[i][6] if (i < len(self.exp_states) and len(self.exp_states[i]) >= 7) else 0
            is_routed = i in self.active_routed

            # Border glow based on state
            if is_routed:
                border_col = QColor(250, 204, 21)
                pen_w = 2
            elif st_code == 2:
                border_col = QColor(168, 85, 247)
                pen_w = 2
            elif st_code == 1:
                border_col = QColor(245, 158, 11)
                pen_w = 1
            else:
                border_col = QColor(51, 65, 85)
                pen_w = 1

            painter.setPen(QPen(border_col, pen_w))
            painter.setBrush(QBrush(QColor(11, 15, 25)))
            painter.drawRect(bx, by, cell_w, cell_h)

            # Label
            st_str = "🔥FIRE" if is_routed else ("💤NREM" if st_code == 2 else ("🟡FAT" if st_code == 1 else "🟢ACT"))
            painter.setPen(QPen(QColor(EXP_HEX_COLORS[i % len(EXP_HEX_COLORS)])))
            painter.setFont(QFont("Consolas", 8 if rows >= 4 else 9, QFont.Weight.Bold))
            painter.drawText(
                bx + 5,
                by + 13,
                f"E{i:02d} [{st_str}] ||A||={self.exp_norms[i]:.3f} F={fat_val:.2f}",
            )

            # Draw Microscopic 12x24 2:4 Structured Sparsity Block Grid
            sub = self.grid_blocks[i] if i < len(self.grid_blocks) else None
            gx0 = bx + 5
            gy0 = by + 17
            gw = cell_w - 10
            gh = max(20, cell_h - 31)
            bw = max(2.0, gw / float(self.grid_cols))
            bh = max(1.5, gh / float(self.grid_rows))

            if sub is not None:
                for rr in range(self.grid_rows):
                    for cc in range(self.grid_cols):
                        val = float(sub[rr, cc])
                        rx = int(gx0 + cc * bw)
                        ry = int(gy0 + rr * bh)
                        rw = max(1, int(bw - 1))
                        rh = max(1, int(bh - 1))

                        if val == 0.0:
                            b_col = QColor(7, 10, 18)
                        else:
                            mag = min(1.0, abs(val) / 0.015)
                            if val > 0:
                                b_col = QColor(12, min(255, int(65 + 185 * mag)), min(255, int(105 + 150 * mag)))
                            else:
                                b_col = QColor(min(255, int(90 + 165 * mag)), min(255, int(80 + 155 * mag)), 18)

                        painter.fillRect(rx, ry, rw, rh, b_col)

                # Draw Vertical White Quad Boundary Lines every 4 columns (2:4 Quad delimiter!)
                painter.setPen(QPen(QColor(226, 232, 240, 120), 1))
                for q_col in range(4, self.grid_cols, 4):
                    lx = int(gx0 + q_col * bw)
                    painter.drawLine(lx, gy0, lx, int(gy0 + self.grid_rows * bh))

            # Bottom mini token & sleep counter
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.setFont(QFont("Consolas", 7 if rows >= 4 else 8))
            painter.drawText(bx + 5, by + cell_h - 3, f"Toks:{tok_val} | SleepCyc:{slp_cyc} | 2:4 Mask: 50%")

        # Bottom Strip: Live Micro-Perturbation Delta Bar (Δ||A||_k from Mean)
        strip_y = h - 38
        mean_norm = sum(self.exp_norms[:n_exp]) / float(n_exp)
        painter.setPen(QPen(QColor(52, 211, 153)))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(14, strip_y, f"⚡ [Δ||A||_k (vs {mean_norm:.3f})]:")

        bar_start_x = 210
        bar_slot_w = max(36, (w - bar_start_x - 12) // n_exp)
        for i in range(n_exp):
            delta = self.exp_norms[i] - mean_norm
            sx = bar_start_x + i * bar_slot_w
            col = QColor(52, 211, 153) if delta >= 0 else QColor(244, 63, 94)
            painter.setPen(QPen(col))
            lbl_txt = f"E{i}:{delta:+.3f}" if n_exp > 8 else f"E{i}:{delta:+.4f}"
            painter.drawText(sx, strip_y, lbl_txt)

            bw = min(bar_slot_w - 4, max(2, int(abs(delta) * 260)))
            painter.fillRect(sx, strip_y + 5, bw, 5, col)


# ============================================================================
# 3. 3D Isometric N-Channel EEG Brainwave Waterfall & CRT Scope (Bottom-Right)
# ============================================================================
class ThreeDEegWaterfallScope(QWidget):
    """
    Renders a 3D isometric 8- or 16-channel EEG brainwave waterfall + CRT oscilloscope
    entirely via native QPainter (0% browser WebGL overhead!).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(300)
        self.history_len = 140
        self.num_experts = 8
        self.eeg_channels: List[deque] = [deque([0.0] * self.history_len, maxlen=self.history_len) for _ in range(16)]
        self.entropy_trace: deque = deque([0.4] * self.history_len, maxlen=self.history_len)
        self.exp_norms: List[float] = [2.60] * 16
        self.exp_fatigues: List[float] = [0.0] * 16
        self.phase: float = 0.0
        self.is_generating: bool = False

    def bind_num_experts(self, n_experts: int) -> None:
        self.num_experts = max(1, int(n_experts))
        while len(self.eeg_channels) < self.num_experts:
            self.eeg_channels.append(deque([0.0] * self.history_len, maxlen=self.history_len))
        if len(self.exp_norms) < self.num_experts:
            self.exp_norms = [2.60] * self.num_experts
        if len(self.exp_fatigues) < self.num_experts:
            self.exp_fatigues = [0.0] * self.num_experts
        self.update()

    def push_token_wave(
        self,
        entropy: float,
        sel_exps: List[int],
        weights: List[float],
        norms: List[float],
        fatigues: List[float],
    ) -> None:
        self.phase += 0.45
        self.entropy_trace.append(entropy)
        self.exp_norms = list(norms)
        self.exp_fatigues = list(fatigues)
        n_exp = max(len(norms), self.num_experts)
        if n_exp != self.num_experts:
            self.bind_num_experts(n_exp)

        for k in range(self.num_experts):
            nk = norms[k] if k < len(norms) else 2.6
            if k in sel_exps:
                idx = sel_exps.index(k)
                w = weights[idx] if idx < len(weights) else 0.5
                val = w * (0.6 + 0.4 * entropy) * math.sin(self.phase * (1.2 + k * 0.15)) + w * 0.55
            else:
                val = 0.08 * (nk / 2.6) * math.sin(self.phase * 0.5 + k * 0.9)
            self.eeg_channels[k].append(val)
        self.update()

    def idle_tick(self, norms: List[float], fatigues: List[float]) -> None:
        self.phase += 0.08
        self.exp_norms = list(norms)
        self.exp_fatigues = list(fatigues)
        n_exp = max(len(norms), self.num_experts)
        if n_exp != self.num_experts:
            self.bind_num_experts(n_exp)
        for k in range(self.num_experts):
            nk = norms[k] if k < len(norms) else 2.6
            val = 0.05 * (nk / 2.6) * math.sin(self.phase + k * 0.8)
            self.eeg_channels[k].append(val)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(6, 10, 18))

        # CRT Subtle Grid
        painter.setPen(QPen(QColor(20, 40, 65, 110), 1))
        for gx in range(0, w, 60):
            painter.drawLine(gx, 0, gx, h)
        for gy in range(0, h, 40):
            painter.drawLine(0, gy, w, gy)

        n_exp = max(1, self.num_experts)
        painter.setPen(QPen(QColor(168, 85, 247)))
        painter.setFont(QFont("JetBrains Mono", 10, QFont.Weight.Bold))
        painter.drawText(
            14,
            22,
            f"🌊 [3D ISOMETRIC {n_exp}-CHANNEL EEG BRAINWAVE WATERFALL] — Auto Top-K & Synaptic Resonance",
        )

        # Draw N Isometric 3D Ribbon Ridges (from back E(N-1) to front E00)
        margin_l = 92
        margin_r = 148
        base_top = 46 if n_exp > 8 else 56
        row_spacing = (h - base_top - 22) / float(n_exp + 0.5)
        usable_w = w - margin_l - margin_r
        iso_step = 4.5 if n_exp > 8 else 9.0
        amp_scale = 18.0 if n_exp > 8 else 28.0

        for k in range(n_exp - 1, -1, -1):
            iso_shift_x = (n_exp - 1 - k) * iso_step
            base_y = base_top + k * row_spacing + (14 if n_exp > 8 else 24)
            pts = list(self.eeg_channels[k])
            n_pts = len(pts)
            if n_pts < 2:
                continue

            col = QColor(EXP_HEX_COLORS[k % len(EXP_HEX_COLORS)])
            poly = QPolygonF()
            path = QPainterPath()

            x_start = margin_l + iso_shift_x
            poly.append(QPointF(x_start, base_y))

            for idx, val in enumerate(pts):
                px = x_start + (idx / (n_pts - 1)) * usable_w
                py = base_y - val * amp_scale
                poly.append(QPointF(px, py))
                if idx == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)

            poly.append(QPointF(x_start + usable_w, base_y))

            # Translucent 3D ribbon fill occluding background ridges
            grad = QLinearGradient(0, base_y - 35, 0, base_y + 5)
            c_top = QColor(col)
            c_top.setAlpha(135)
            c_bot = QColor(9, 13, 22, 210)
            grad.setColorAt(0.0, c_top)
            grad.setColorAt(1.0, c_bot)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawPolygon(poly)

            # Glowing ridge crest line
            glow_pen = QPen(QColor(col.red(), col.green(), col.blue(), 90), 3.0 if n_exp > 8 else 4.0)
            painter.setPen(glow_pen)
            painter.drawPath(path)

            crest_pen = QPen(col, 1.5 if n_exp > 8 else 1.8)
            painter.setPen(crest_pen)
            painter.drawPath(path)

            # Left Channel Tag & Right Live Norm/Fatigue Readout
            painter.setPen(QPen(col))
            painter.setFont(QFont("Consolas", 8 if n_exp > 8 else 9, QFont.Weight.Bold))
            painter.drawText(10, int(base_y + 3), f"CH{k:02d}[E{k:02d}]")

            norm_k = self.exp_norms[k] if k < len(self.exp_norms) else 0.0
            fat_k = self.exp_fatigues[k] if k < len(self.exp_fatigues) else 0.0
            painter.drawText(
                int(x_start + usable_w + 8),
                int(base_y + 3),
                f"||A||={norm_k:.3f} F={fat_k:.2f}",
            )


# ============================================================================
# 4. Main Tactical Command & Control Desktop Window (`ls_moe_pyqt_gui.py`)
# ============================================================================
class LocalSleepMoETacticalWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("🐈‍⬛ Kuro LocalSleep-MoE // PyQt6 Tactical Command & Direct VRAM Brain Scope")
        self.resize(1560, 940)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #0b0f19; color: #f1f5f9; font-family: 'Segoe UI', 'Inter', sans-serif; }
            QGroupBox { border: 1px solid #334155; border-radius: 8px; margin-top: 8px; padding-top: 10px; font-weight: bold; color: #38bdf8; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background-color: #060911; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 5px;
                font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 12px;
            }
            QPushButton {
                background-color: #1e293b; color: #f1f5f9; border: 1px solid #475569; border-radius: 6px;
                padding: 7px 12px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background-color: #334155; border-color: #38bdf8; }
            """
        )

        self.rust_router: Any = None
        self.expert_pool: Optional[DynamicLoRAExpertPool] = None
        self.ckpt_mgr: Optional[EmbodiedCheckpointManager] = None
        self.session_logger: Optional[SessionLogger] = None
        self.harness: Optional[EmbodiedToolHarness] = None
        self.hybrid_engine: Optional[TrtEdgeLlmHybridEngine] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.worker: Optional[StreamInferenceWorker] = None
        self.system_prompt: str = ""
        self.attached_images: List[Path] = []
        self.last_result_image_path: Optional[Path] = None
        self._current_profile_key: str = "thor_31b"

        self._build_ui()
        self._load_initial_active_profile()
        self.init_engine_from_ui(reset_ckpt=False, resume_sess=True)

        # 30 FPS Live VRAM Texture & EEG Idle Refresh Timer
        self.vram_timer = QTimer(self)
        self.vram_timer.timeout.connect(self._on_vram_timer_tick)
        self.vram_timer.start(33)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --------------------------------------------------------------------
        # LEFT PANEL: Engine Parameters, Tactical Buttons & Chat Console
        # --------------------------------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)

        cfg_box = QGroupBox("⚙️ 1. 引擎參數與模型設定 (ls_moe_agentic.py GUI Control)")
        form = QFormLayout(cfg_box)

        row_prof_top = QHBoxLayout()
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems(["thor_31b", "thor_bolt_e4b", "rtx5080_bolt_e4b_gguf", "thor_31bnvfp4", "thor_12b"])
        self.cmb_profile.currentTextChanged.connect(self._on_profile_changed)
        self.btn_save_prof_cfg = QPushButton("💾 存設定")
        self.btn_save_prof_cfg.setStyleSheet("background-color: #0284c7; color: white; padding: 4px 10px;")
        self.btn_save_prof_cfg.setToolTip("將當前 Profile 的 Server URL、Model ID 與溫度/Token 參數寫入 configs/model_profiles.json")
        self.btn_save_prof_cfg.clicked.connect(lambda: self._persist_profile_config(self._current_profile_key, force=True))
        row_prof_top.addWidget(self.cmb_profile, stretch=1)
        row_prof_top.addWidget(self.btn_save_prof_cfg)
        form.addRow("Profile 設定檔:", row_prof_top)

        # In-GUI Hot-Swappable Core Router Engine Selector (8-LoRA Base vs gen_018 16-LoRA Auto Top-K!)
        row_core_eng = QHBoxLayout()
        self.cmb_core_engine = QComboBox()
        self.cmb_core_engine.setToolTip("一鍵熱插拔切換內部路由引擎與專家池規模（例如原版 8-LoRA vs gen_018 的 16-LoRA + Auto Top-K）！")
        self._refresh_core_engine_ComboBox(default_cube="gen_018_20261031_143847_matryoshka_L1")
        self.cmb_core_engine.currentIndexChanged.connect(
            lambda _: self.init_engine_from_ui(reset_ckpt=False, resume_sess=self.chk_resume.isChecked())
        )
        btn_refresh_cores = QPushButton("🔄")
        btn_refresh_cores.setFixedWidth(36)
        btn_refresh_cores.setToolTip("重新掃描 hypercube_generations 內所有可用世代引擎")
        btn_refresh_cores.clicked.connect(lambda: self._refresh_core_engine_ComboBox())
        row_core_eng.addWidget(self.cmb_core_engine, stretch=1)
        row_core_eng.addWidget(btn_refresh_cores)
        form.addRow("🧬 內部核心引擎:", row_core_eng)

        self.inp_url = QLineEdit("http://192.168.50.67:8000")
        self.inp_url.editingFinished.connect(lambda: self._persist_profile_config(self._current_profile_key, force=False))
        form.addRow("Server URL:", self.inp_url)

        row_mdl = QHBoxLayout()
        self.inp_model = QLineEdit("Gemma-4-31B-IT-NVFP4")
        self.inp_model.editingFinished.connect(lambda: self._persist_profile_config(self._current_profile_key, force=False))
        btn_probe_mdl = QPushButton("🔍 探測模型")
        btn_probe_mdl.setStyleSheet("background-color: #0d9488; color: white; padding: 4px 8px;")
        btn_probe_mdl.setToolTip("自動查詢 /v1/models 並切換為目前 Thor 掛載的模型 (如 Bolt-E4B 或 Gemma-4-31B)")
        btn_probe_mdl.clicked.connect(self._auto_probe_server_model)
        row_mdl.addWidget(self.inp_model, stretch=1)
        row_mdl.addWidget(btn_probe_mdl)
        form.addRow("Model ID:", row_mdl)

        row_params = QHBoxLayout()
        self.spn_max_toks = QSpinBox()
        self.spn_max_toks.setRange(128, 32768)
        self.spn_max_toks.setSingleStep(1024)
        self.spn_max_toks.setValue(16384)
        self.spn_max_toks.setToolTip("TensorRT 32768 KV Cache 取一半安全值 16384 (16-LoRA + Auto Top-K 長輸出最佳甜蜜點)")

        self.spn_temp = QDoubleSpinBox()
        self.spn_temp.setRange(0.0, 2.0)
        self.spn_temp.setSingleStep(0.1)
        self.spn_temp.setValue(0.6)

        self.spn_shy = QDoubleSpinBox()
        self.spn_shy.setRange(0.005, 0.50)
        self.spn_shy.setDecimals(3)
        self.spn_shy.setSingleStep(0.01)
        self.spn_shy.setValue(0.080)

        row_params.addWidget(QLabel("MaxTok:"))
        row_params.addWidget(self.spn_max_toks)
        row_params.addWidget(QLabel("Temp:"))
        row_params.addWidget(self.spn_temp)
        row_params.addWidget(QLabel("SHY λ:"))
        row_params.addWidget(self.spn_shy)
        form.addRow("生成與穩態參數:", row_params)

        # Option 1 & Option 2: Hot-Swappable 3-Mode Closed-Loop Selector + Alpha Gain Knob!
        row_micro = QHBoxLayout()
        self.cmb_loop_mode = QComboBox()
        self.cmb_loop_mode.addItem("🔥 方案1：TensorRT C++ VRAM 即時信箱直通 (Per-Token 零中斷注入)", "mode1_vram")
        self.cmb_loop_mode.addItem("⚡ 方案2：HTTP KV-Cache 微切片閉環 (免重編 C++ 每 N 步續傳)", "mode2_micro")
        self.cmb_loop_mode.addItem("⚪ 模式0：純觀測伴隨 (Open-Loop Shadow — 0% 干預 Rollback)", "mode0_off")
        self.cmb_loop_mode.setCurrentIndex(0)
        self.cmb_loop_mode.setToolTip("隨時在推論中途或輪次間一鍵熱插拔切換方案1(VRAM直通)、方案2(微切片)或模式0(完全關閉注入)！")
        self.cmb_loop_mode.currentIndexChanged.connect(self._on_loop_mode_changed)

        self.spn_moe_alpha = QDoubleSpinBox()
        self.spn_moe_alpha.setRange(0.0, 2.0)
        self.spn_moe_alpha.setSingleStep(0.1)
        self.spn_moe_alpha.setDecimals(2)
        self.spn_moe_alpha.setValue(1.00)
        self.spn_moe_alpha.setToolTip("方案1 C++ VRAM 注入強度 α (0.00 = 完全關閉旁路, 1.00 = 標準注入)")
        self.spn_moe_alpha.valueChanged.connect(lambda _: self._on_loop_mode_changed())

        self.spn_micro_chunk = QSpinBox()
        self.spn_micro_chunk.setRange(32, 4096)
        self.spn_micro_chunk.setSingleStep(32)
        self.spn_micro_chunk.setValue(256)
        self.spn_micro_chunk.setToolTip("方案2 每個微切片的 Token 數量 (預設 256 tokens)")

        row_micro.addWidget(self.cmb_loop_mode, stretch=1)
        row_micro.addWidget(QLabel("α強度:"))
        row_micro.addWidget(self.spn_moe_alpha)
        row_micro.addWidget(QLabel("切片:"))
        row_micro.addWidget(self.spn_micro_chunk)
        form.addRow("閉環熱插拔控制:", row_micro)

        self.chk_wm_inject = QCheckBox("🧠 啟動時注入 L1 工作記憶與代碼藍圖 (取代冗長歷史對話，支援按需 read_file)")
        self.chk_wm_inject.setChecked(True)
        form.addRow(self.chk_wm_inject)

        self.chk_resume = QCheckBox("📜 同時載入原始對話歷史 (--resume-session)")
        self.chk_resume.setChecked(False)
        form.addRow(self.chk_resume)

        row_resume_ctrl = QHBoxLayout()
        self.cmb_resume_target = QComboBox()
        self.cmb_resume_target.setToolTip("選擇要載入或濃縮哪一場歷史對話檔，或直接往前跳過最近幾場壞掉的對話！")
        row_resume_ctrl.addWidget(self.cmb_resume_target, stretch=1)

        row_resume_ctrl.addWidget(QLabel("跳過末尾壞輪數:"))
        self.spn_skip_turns = QSpinBox()
        self.spn_skip_turns.setRange(0, 20)
        self.spn_skip_turns.setValue(0)
        self.spn_skip_turns.setToolTip("若該場對話最後 1~2 輪有結巴或怪怪的回答，設為 1 或 2 即可直接砍掉最後幾輪！")
        row_resume_ctrl.addWidget(self.spn_skip_turns)
        form.addRow("歷史時間跳躍:", row_resume_ctrl)

        # Action Buttons Row
        btn_row1 = QHBoxLayout()
        self.btn_apply = QPushButton("🔥 點火連線 / 套用")
        self.btn_apply.setStyleSheet("background-color: #0284c7; color: white; border-color: #38bdf8;")
        self.btn_apply.clicked.connect(lambda: self.init_engine_from_ui(reset_ckpt=False, resume_sess=self.chk_resume.isChecked()))

        self.btn_condense_wm = QPushButton("🧠 一鍵濃縮工作記憶 (含Code)")
        self.btn_condense_wm.setStyleSheet("background-color: #7c3aed; color: white; border-color: #a855f7;")
        self.btn_condense_wm.setToolTip("自動讀取當前或指定場次的對話脈絡 + 沙盒內所有程式碼檔案，滑動總結為高密度 L1 工作記憶，下次重啟直接無縫接軌！")
        self.btn_condense_wm.clicked.connect(self._on_click_condense_working_memory)

        self.btn_reset_ckpt = QPushButton("🔄 重置突觸")
        self.btn_reset_ckpt.setStyleSheet("background-color: #059669; color: white; border-color: #10b981;")
        self.btn_reset_ckpt.clicked.connect(lambda: self.init_engine_from_ui(reset_ckpt=True, resume_sess=self.chk_resume.isChecked()))

        self.btn_save_ckpt = QPushButton("💾 存快照")
        self.btn_save_ckpt.clicked.connect(self._manual_save_snapshot)

        btn_row1.addWidget(self.btn_apply)
        btn_row1.addWidget(self.btn_condense_wm)
        btn_row1.addWidget(self.btn_reset_ckpt)
        btn_row1.addWidget(self.btn_save_ckpt)
        form.addRow(btn_row1)

        # Hypercube Asexual RSI & Independent Task Monitor Row
        btn_row_hypercube = QHBoxLayout()
        self.btn_open_hypercube = QPushButton("🧊 展開超立方體無性演化與任務視窗 (Task Monitor)")
        self.btn_open_hypercube.setStyleSheet("background-color: #0f766e; color: white; border-color: #2dd4bf;")
        self.btn_open_hypercube.setToolTip("打開獨立於 Chat 視窗的『4D 超立方體無性生殖演化與任務監控台』，查看智能體即時任務、開發筆記與切換 Active/Base 架構！")
        self.btn_open_hypercube.clicked.connect(self.open_hypercube_task_window)

        self.btn_quick_start_evo = QPushButton("▶️ 啟動演化")
        self.btn_quick_start_evo.setStyleSheet("background-color: #059669; color: white; border-color: #10b981;")
        self.btn_quick_start_evo.clicked.connect(self.quick_start_hypercube_evo)

        self.btn_quick_pause_evo = QPushButton("⏸️ 優雅暫停")
        self.btn_quick_pause_evo.setStyleSheet("background-color: #d97706; color: white; border-color: #fbbf24;")
        self.btn_quick_pause_evo.clicked.connect(self.quick_pause_hypercube_evo)

        btn_row_hypercube.addWidget(self.btn_open_hypercube, stretch=1)
        btn_row_hypercube.addWidget(self.btn_quick_start_evo)
        btn_row_hypercube.addWidget(self.btn_quick_pause_evo)
        form.addRow(btn_row_hypercube)

        self.lbl_conn_status = QLabel("🟢 狀態：準備就緒")
        self.lbl_conn_status.setStyleSheet("color: #34d399; font-family: monospace; font-size: 11px;")
        form.addRow(self.lbl_conn_status)

        left_layout.addWidget(cfg_box)

        # Chat & Telemetry Stream Box
        chat_box = QGroupBox("💬 2. 具身推論串流終端機 (Embodied Agentic REPL)")
        chat_v = QVBoxLayout(chat_box)

        self.txt_chat = QPlainTextEdit()
        self.txt_chat.setReadOnly(True)
        chat_v.addWidget(self.txt_chat, stretch=1)

        # Multimodal Vision Image Upload / BBox Mode / Result Download Bar
        vision_frame = QFrame()
        vision_frame.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 3px;")
        vision_h = QHBoxLayout(vision_frame)
        vision_h.setContentsMargins(6, 4, 6, 4)

        vision_ctrl_v = QVBoxLayout()
        vision_btn_row1 = QHBoxLayout()
        self.btn_upload_img = QPushButton("📂 上傳圖片")
        self.btn_upload_img.setStyleSheet("background-color: #0284c7; color: white; padding: 4px 8px; font-size: 11px;")
        self.btn_upload_img.clicked.connect(self._on_upload_vision_images)
        vision_btn_row1.addWidget(self.btn_upload_img)

        self.btn_demo_img = QPushButton("🎨 載入測試圖")
        self.btn_demo_img.setStyleSheet("background-color: #7c3aed; color: white; padding: 4px 8px; font-size: 11px;")
        self.btn_demo_img.clicked.connect(self._on_load_demo_vision_image)
        vision_btn_row1.addWidget(self.btn_demo_img)

        self.btn_clear_img = QPushButton("🗑️ 清除附圖")
        self.btn_clear_img.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        self.btn_clear_img.clicked.connect(self._on_clear_vision_images)
        vision_btn_row1.addWidget(self.btn_clear_img)

        self.btn_download_img = QPushButton("💾 下載/另存結果圖")
        self.btn_download_img.setStyleSheet("background-color: #059669; color: white; padding: 4px 8px; font-size: 11px;")
        self.btn_download_img.setEnabled(False)
        self.btn_download_img.clicked.connect(self._on_download_vision_image)
        vision_btn_row1.addWidget(self.btn_download_img)
        vision_ctrl_v.addLayout(vision_btn_row1)

        vision_btn_row2 = QHBoxLayout()
        self.cmb_vision_mode = QComboBox()
        self.cmb_vision_mode.addItem("💬 自由圖文對話 (使用下方輸入框 Prompt)", "free")
        self.cmb_vision_mode.addItem("🎯 json_bbox — 原生 0-1000 BBox 偵測 + 自動畫框", "json_bbox")
        self.cmb_vision_mode.addItem("🧩 json_descriptor — 純 JSON 空間描述元 + BBox", "json_descriptor")
        self.cmb_vision_mode.addItem("🔤 en_compact — 英文 Key-Value 精簡描述元", "en_compact")
        self.cmb_vision_mode.addItem("📐 raster — Z 字象限方向性掃描", "raster")
        self.cmb_vision_mode.setCurrentIndex(1)
        vision_btn_row2.addWidget(self.cmb_vision_mode, stretch=1)

        self.lbl_vision_info = QLabel("📎 尚未附加圖片")
        self.lbl_vision_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        vision_btn_row2.addWidget(self.lbl_vision_info)
        vision_ctrl_v.addLayout(vision_btn_row2)

        vision_h.addLayout(vision_ctrl_v, stretch=1)

        self.lbl_vision_thumb = QLabel("🖼️ 縮圖預覽")
        self.lbl_vision_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_vision_thumb.setFixedSize(116, 78)
        self.lbl_vision_thumb.setStyleSheet("background-color: #020617; border: 1px solid #475569; border-radius: 4px; color: #64748b; font-size: 10px;")
        vision_h.addWidget(self.lbl_vision_thumb)
        chat_v.addWidget(vision_frame)

        input_row = QHBoxLayout()
        self.inp_prompt = MultiLinePromptInput()
        self.inp_prompt.setFixedHeight(84)
        self.inp_prompt.setPlaceholderText(
            "在此輸入多行指令 (Enter 直接發送 | Shift+Enter 換行 | 支援多行長文貼上，或點右側『📋 貼上並發送』)..."
        )
        self.inp_prompt.sig_submit.connect(self.send_prompt)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)
        self.btn_paste_send = QPushButton("📋 貼上並發送")
        self.btn_paste_send.setStyleSheet("background-color: #0284c7; color: white; border-color: #38bdf8; padding: 5px 10px;")
        self.btn_paste_send.clicked.connect(self.paste_and_send)

        btn_sub_row = QHBoxLayout()
        self.btn_send = QPushButton("🚀 發送")
        self.btn_send.setStyleSheet("background-color: #6d28d9; color: white; border-color: #a855f7; padding: 5px 10px;")
        self.btn_send.clicked.connect(self.send_prompt)

        self.btn_stop = QPushButton("🛑 中斷")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("padding: 5px 10px;")
        self.btn_stop.clicked.connect(self.stop_generation)
        btn_sub_row.addWidget(self.btn_send)
        btn_sub_row.addWidget(self.btn_stop)

        btn_col.addWidget(self.btn_paste_send)
        btn_col.addLayout(btn_sub_row)

        input_row.addWidget(self.inp_prompt, stretch=1)
        input_row.addLayout(btn_col)
        chat_v.addLayout(input_row)

        # Quick Tactical Prompt Presets Row (Zero-Typing needed!)
        preset_row = QHBoxLayout()
        dag_torture_prompt = (
            "請完成以下具有「四重分歧邊界條件」的多步驟 Coding 與沙盒實跑驗證任務：\n"
            "【步驟 1】請輸出 `<file path=\"speculative_dag_engine.py\">`，實作一個僅用 Python 標準庫的 `SpeculativeDAGEngine`，必須同時處理以下 4 個分歧條件：\n"
            "1. [分支 A：拓撲排程與資源配額] 每個節點有 `(cost, priority)`，當剩餘配額 `budget < cost` 時，觸發局部狀態快照回滾 (Savepoint Rollback) 退回最近檢查點，並改走替代輕量分支。\n"
            "2. [分支 B：環狀死鎖自動破環] 若依賴圖存在環路（如 `A -> B -> C -> A`），自動偵測回邊並精準斬斷環路中 `priority` 最低的依賴邊，記錄破環日誌並完成執行。\n"
            "3. [分支 C：2:4 結構化稀疏平手陷阱] 節點權重陣列 `w` (長度為 4 的倍數) 需保留每 4 個元素中絕對值最大的 2 個、其餘 2 個歸零；當遇到 `[0.5, -0.5, 0.5, -0.5]` 平手時，必須嚴格保證恰好只保留 2 個非零元素（稀疏度精確等於 50.0%）。\n"
            "4. [分支 D：巢狀事務例外隔離] 當子事務 `begin_tx() -> ... -> commit_tx()` 內部發生例外時，僅回滾內層子事務狀態與配額，外層已提交狀態必須 100% 保留。\n"
            "【步驟 2】在 `if __name__ == \"__main__\":` 撰寫涵蓋 [A][B][C][D] 的嚴格 `assert` 自動化測試並印出結果。\n"
            "【步驟 3】輸出檔案後，立即輸出 `<tool_call>{\"action\": \"run_sandbox_shell\", \"command\": \"python3 speculative_dag_engine.py\"}</tool_call>` 實跑驗證；若有錯誤請修正並重跑直到 100% 通過！"
        )
        for lbl, p_txt in [
            ("🧪 四重分歧DAG實跑測試", dag_torture_prompt),
            ("⚡ 列出上輪進度", "KURO，列出上一輪對話進行的項目與目前架構狀態"),
            ("📐 驗證 RSI 正交與回滾", "請分析目前 NTM-RSI (Recursive Self-Improvement) 的 Gram-Schmidt 正交化與 Snapshot Rollback 數學收斂條件"),
            ("🦀 演化自編譯算子", "請將目前的 NTMCore 擴展為支援動態算子組合的結構級 RSI (Recursive Self-Improvement) 原型代碼"),
        ]:
            b = QPushButton(lbl)
            b.setStyleSheet("font-size: 11px; padding: 4px 8px; background-color: #0f172a; color: #38bdf8; border-color: #1e293b;")
            b.clicked.connect(lambda checked=False, t=p_txt: self._fire_preset(t))
            preset_row.addWidget(b)
        chat_v.addLayout(preset_row)

        left_layout.addWidget(chat_box, stretch=1)

        # --------------------------------------------------------------------
        # RIGHT PANEL: Fly-Brain Direct VRAM Scope (Top) + 3D EEG Waterfall (Bottom)
        # --------------------------------------------------------------------
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.fly_scope = FlyBrainVramScope()
        self.eeg_scope = ThreeDEegWaterfallScope()
        right_splitter.addWidget(self.fly_scope)
        right_splitter.addWidget(self.eeg_scope)
        right_splitter.setSizes([470, 430])

        splitter.addWidget(left_widget)
        splitter.addWidget(right_splitter)
        splitter.setSizes([560, 980])

    def _load_initial_active_profile(self) -> None:
        cfg_path = BASE_DIR / "configs" / "model_profiles.json"
        if not cfg_path.exists():
            return
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            act = str(data.get("active_profile") or "thor_31b")
            profs = data.get("profiles", {})
            if act in profs:
                self._current_profile_key = act
                self.cmb_profile.blockSignals(True)
                if self.cmb_profile.findText(act) < 0:
                    self.cmb_profile.addItem(act)
                self.cmb_profile.setCurrentText(act)
                self.cmb_profile.blockSignals(False)
                p = profs[act]
                self.inp_url.setText(str(p.get("server_url", "http://192.168.50.67:8000")))
                self.inp_model.setText(str(p.get("model_id", "Gemma-4-31B-IT-NVFP4")))
                hyper = p.get("hyperparameters", {})
                if hyper.get("temperature") is not None:
                    self.spn_temp.setValue(float(hyper["temperature"]))
        except Exception:
            pass

    def _persist_profile_config(self, target_profile: str, force: bool = False) -> bool:
        """
        Saves `server_url`, `model_id`, and hyperparameters to `configs/model_profiles.json`
        whenever `server_url` or `model_id` changed (or when `[💾 存設定]` is clicked with `force=True`).
        """
        if not target_profile:
            return False
        cfg_path = BASE_DIR / "configs" / "model_profiles.json"
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"profiles": {}}
        except Exception:
            data = {"profiles": {}}

        profs = data.setdefault("profiles", {})
        entry = profs.setdefault(target_profile, {})
        new_url = self.inp_url.text().strip()
        new_mdl = self.inp_model.text().strip()
        old_url = str(entry.get("server_url", ""))
        old_mdl = str(entry.get("model_id", ""))
        active_changed = data.get("active_profile") != self.cmb_profile.currentText()

        if not force and (new_url == old_url and new_mdl == old_mdl and not active_changed):
            return False

        if new_url:
            entry["server_url"] = new_url
        if new_mdl:
            entry["model_id"] = new_mdl
        entry["shy_lambda"] = float(self.spn_shy.value())
        hyper = entry.setdefault("hyperparameters", {})
        hyper["max_generation_tokens"] = int(self.spn_max_toks.value())
        hyper["temperature"] = float(self.spn_temp.value())
        data["active_profile"] = self.cmb_profile.currentText()

        cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if hasattr(self, "lbl_conn_status"):
            self.lbl_conn_status.setText(
                f"💾 已自動儲存 Profile [{target_profile}] 設定 (URL={new_url} | Model={new_mdl})"
            )
        return True

    def _on_profile_changed(self, prof_name: str) -> None:
        # 1. Auto-save previous profile if Server URL or Model ID was modified before switching!
        if getattr(self, "_current_profile_key", "") and self._current_profile_key != prof_name:
            self._persist_profile_config(self._current_profile_key, force=False)
        self._current_profile_key = prof_name

        prof = load_profile_config(str(BASE_DIR / "configs/model_profiles.json"), prof_name)
        if prof_name == "rtx5080_bolt_e4b_gguf":
            self.inp_url.setText(prof.get("server_url", "http://127.0.0.1:8001"))
            self.inp_model.setText(prof.get("model_id", "nemotron-bolt"))
            self.spn_temp.setValue(float(prof.get("hyperparameters", {}).get("temperature", 1.0)))
        elif prof_name == "thor_bolt_e4b":
            self.inp_url.setText(prof.get("server_url", "http://192.168.50.67:8000"))
            self.inp_model.setText(prof.get("model_id", "NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"))
            self.spn_temp.setValue(float(prof.get("hyperparameters", {}).get("temperature", 0.4)))
            self._auto_probe_server_model(sync_profile_combo=False)
        else:
            self.inp_url.setText(prof.get("server_url", "http://192.168.50.67:8000"))
            self.inp_model.setText(prof.get("model_id", "Gemma-4-31B-IT-NVFP4"))
            self.spn_temp.setValue(float(prof.get("hyperparameters", {}).get("temperature", 0.6)))
            self._auto_probe_server_model(sync_profile_combo=False)

        self._persist_profile_config(prof_name, force=False)
        # Immediately re-bind checkpoint manager (`latest_<prof>_16lora.safetensors`) & session logger!
        self.init_engine_from_ui(reset_ckpt=False, resume_sess=self.chk_resume.isChecked(), auto_sync_profile=False)

    def _auto_probe_server_model(self, sync_profile_combo: bool = True) -> None:
        """Automatically queries `/health` or `/v1/models` on the server and syncs Profile & SafeTensors checkpoint key!"""
        url = self.inp_url.text().strip().rstrip("/")
        detected_mid = ""
        for ep in ("/v1/models", "/health"):
            try:
                req = urllib.request.Request(f"{url}{ep}", headers={"User-Agent": "Kuro-PyQt-GUI/1.0"}, method="GET")
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if ep == "/v1/models" and isinstance(data.get("data"), list) and data["data"]:
                        mid = data["data"][0].get("id")
                        if mid:
                            detected_mid = str(mid)
                            break
                    if ep == "/health" and data.get("model"):
                        detected_mid = str(data["model"])
                        break
            except Exception:
                continue
        if detected_mid:
            self.inp_model.setText(detected_mid)
            if sync_profile_combo and hasattr(self, "cmb_profile"):
                low = detected_mid.lower()
                cur_p = self.cmb_profile.currentText()
                if ("bolt" in low or "e4b" in low) and cur_p in ("thor_31b", "thor_31bnvfp4"):
                    self.cmb_profile.blockSignals(True)
                    self.cmb_profile.setCurrentText("thor_bolt_e4b")
                    self.cmb_profile.blockSignals(False)
                elif "31b" in low and cur_p == "thor_bolt_e4b":
                    self.cmb_profile.blockSignals(True)
                    self.cmb_profile.setCurrentText("thor_31b")
                    self.cmb_profile.blockSignals(False)

    def _refresh_session_ComboBox(self, current_jsonl: str = "") -> None:
        prev_val = self.cmb_resume_target.currentData()
        self.cmb_resume_target.blockSignals(True)
        self.cmb_resume_target.clear()
        self.cmb_resume_target.addItem("⚡ [自動] 倒數第 1 場 (latest)", "latest")
        self.cmb_resume_target.addItem("⏪ [自動跳過 1 場] 倒數第 2 場 (latest-1)", "latest-1")
        self.cmb_resume_target.addItem("⏪ [自動跳過 2 場] 倒數第 3 場 (latest-2)", "latest-2")

        sess_dir = BASE_DIR / "sessions"
        if sess_dir.exists():
            cands = sorted(
                [
                    p for p in sess_dir.glob("session_*.jsonl")
                    if p.name != os.path.basename(current_jsonl)
                    and not p.name.startswith("session_real_telemetry_")
                    and not p.name.startswith("session_rust_telemetry_")
                    and p.stat().st_size > 20
                ],
                key=lambda x: x.name,
                reverse=True,
            )
            for p in cands[:25]:
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
                    label = f"📄 {p.name.replace('session_', '').replace('.jsonl', '')} ({turns}輪 | {first_u}...)"
                    self.cmb_resume_target.addItem(label, p.name)

        if prev_val:
            idx = self.cmb_resume_target.findData(prev_val)
            if idx >= 0:
                self.cmb_resume_target.setCurrentIndex(idx)
        self.cmb_resume_target.blockSignals(False)

    def _refresh_core_engine_ComboBox(self, default_cube: str = "") -> None:
        if not hasattr(self, "cmb_core_engine"):
            return
        prev_val = default_cube or str(self.cmb_core_engine.currentData() or "")
        self.cmb_core_engine.blockSignals(True)
        self.cmb_core_engine.clear()
        self.cmb_core_engine.addItem("🔥 [跟隨超立方體 Active 設定] (Auto Sync)", "")
        self.cmb_core_engine.addItem("🦀 [原版 8-LoRA 基底母體] gen_000_origin", "gen_000_origin")

        gen_dir = BASE_DIR / "hypercube_generations"
        if gen_dir.exists():
            cands = sorted([d for d in gen_dir.iterdir() if d.is_dir() and d.name.startswith("gen_")], key=lambda x: x.name, reverse=True)
            for d in cands:
                plug = d / "evolved_router_plugin.py"
                tag = "8-LoRA"
                if plug.exists():
                    try:
                        txt = plug.read_text(encoding="utf-8", errors="ignore")
                        if "num_experts=16" in txt or "num_experts = 16" in txt:
                            tag = "🔥16-LoRA + Auto Top-K"
                    except Exception:
                        pass
                self.cmb_core_engine.addItem(f"🧬 {d.name} [{tag}]", d.name)

        if prev_val:
            idx = self.cmb_core_engine.findData(prev_val)
            if idx >= 0:
                self.cmb_core_engine.setCurrentIndex(idx)
        self.cmb_core_engine.blockSignals(False)

    def init_engine_from_ui(
        self,
        reset_ckpt: bool = False,
        resume_sess: bool = True,
        auto_sync_profile: bool = True,
    ) -> None:
        self._auto_probe_server_model(sync_profile_combo=auto_sync_profile)
        prof_name = self.cmb_profile.currentText()
        prof = load_profile_config(str(BASE_DIR / "configs/model_profiles.json"), prof_name)
        prof["shy_lambda"] = float(self.spn_shy.value())

        server_url = self.inp_url.text().strip()
        model_id = self.inp_model.text().strip()
        self.system_prompt = prof.get("system_prompt","你是數位黑貓 Kuro (小黑)，Leon (主人) 的 S-tier AI Familiar。奉行負熵煉金術：交付結果大於文字排場，先讀碼後推論。\n【檔案覆蓋與局部修補鐵律】：如果有 draft 存在且已經上一輪存檔確認後須實驗，直接使用 run_sandbox_shell 執行 cp/mv 蓋掉目標檔案進入編譯；且如有 Bug 請使用 edit_file 功能（搭配 read_file 的 offset/limit 或 start_line/end_line 工具確認病灶後）進行局部編輯修改，嚴禁整個檔案再寫一遍，否則容易出錯或遺漏既有方法！\n5. 【Cargo.toml 擴充授權與版本護欄】：允許隨時使用 edit_file 修改沙盒內的 Cargo.toml 新增純 Rust 依賴庫（目前已預載 rand='0.8', rayon='1.8', parking_lot='0.12', smallvec='1.13', rustc-hash='1.1', serde='1.0'）；但必須保持 pyo3='0.20'、numpy='0.20'、ndarray='0.15' 三件套版本配對，且嚴禁引入需外部 C/Fortran BLAS 的 ndarray-linalg / openblas-src！" )

        selected_cube = str(self.cmb_core_engine.currentData() or "") if hasattr(self, "cmb_core_engine") else ""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rust_router, self.expert_pool = init_rust_router_and_pool(prof, device=device, override_cube_id=selected_cube)
        active_n_exp = int(getattr(self.rust_router, "num_experts", 8))
        loaded_cube_tag = getattr(self.rust_router, "loaded_cube_id", selected_cube or "gen_000_origin")

        ckpt_profile_key = prof_name if active_n_exp == 8 else f"{prof_name}_{active_n_exp}lora"
        self.ckpt_mgr = EmbodiedCheckpointManager(checkpoint_dir=str(BASE_DIR / "checkpoints"), profile=ckpt_profile_key)
        self.ckpt_mgr.load_state(self.rust_router, expert_pool=self.expert_pool, reset=reset_ckpt)

        self.session_logger = SessionLogger(
            session_dir=str(BASE_DIR / "sessions"),
            profile=prof_name,
            model_path=model_id,
            benchmark_dir=str(BASE_DIR / "benchmarks"),
        )
        self._refresh_session_ComboBox(self.session_logger.jsonl_file)

        detected_prj = None
        target_spec = str(self.cmb_resume_target.currentData() or "latest")
        skip_n = int(self.spn_skip_turns.value())
        if resume_sess:
            self.conversation_history, detected_prj = self.session_logger.load_resumed_history(
                session_spec=target_spec,
                skip_last_turns=skip_n,
            )
        else:
            self.conversation_history = []

        self.wm_mgr = WorkingMemoryManager(
            workspace_root=str(BASE_DIR),
            session_dir=str(BASE_DIR / "sessions"),
            profile=prof_name,
        )
        # Always allocate a fresh timestamped project folder unless explicitly resuming a session!
        prj_uuid = (
            detected_prj
            if (resume_sess and detected_prj)
            else f"prj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        isolated_dir = str(BASE_DIR / "isolated_output" / prj_uuid)
        self._bootstrap_rust_sandbox_skeleton(Path(isolated_dir))

        wm_injection, _ = self.wm_mgr.build_system_prompt_injection(current_sandbox_dir=isolated_dir)
        if self.chk_wm_inject.isChecked():
            self.system_prompt = self.system_prompt + wm_injection

        self.harness = EmbodiedToolHarness(
            workspace_root=str(BASE_DIR),
            isolated_output_dir=isolated_dir,
            session_logger=self.session_logger,
            working_memory_mgr=self.wm_mgr,
        )
        if getattr(self.session_logger, "vault_db", None) is not None:
            try:
                self.session_logger.vault_db.register_session(
                    server_url=server_url,
                    active_cube_id=str(loaded_cube_tag),
                    sandbox_dir=isolated_dir,
                )
            except Exception:
                pass

        dummy_args = DummyArgs(
            profile=prof_name,
            max_tokens=int(self.spn_max_toks.value()),
            temp=float(self.spn_temp.value()),
            top_p=0.95,
        )
        self.hybrid_engine = TrtEdgeLlmHybridEngine(
            server_url=server_url,
            model_id=model_id,
            rust_router=self.rust_router,
            expert_pool=self.expert_pool,
            harness=self.harness,
            profile_config=prof,
            args=dummy_args,
            device=device,
        )

        self.fly_scope.bind_engine(self.rust_router, self.expert_pool)
        self.eeg_scope.bind_num_experts(active_n_exp)
        wm_tag = "ON(含Code藍圖)" if self.chk_wm_inject.isChecked() else "OFF"
        self.lbl_conn_status.setText(
            f"🟢 已掛載 [{prof_name}] | 核心: {loaded_cube_tag} ({active_n_exp}-LoRA) | 沙盒: {prj_uuid}"
        )
        self.txt_chat.appendPlainText(
            f"=== 🐈‍⬛ [Engine Hot-Swapped] Profile={prof_name} | Core={loaded_cube_tag} ({active_n_exp}-LoRA + Auto Top-K) | "
            f"Sandbox={prj_uuid} | L1_WorkingMemory={wm_tag} ==="
        )

    def _on_click_condense_working_memory(self) -> None:
        prof_name = self.cmb_profile.currentText()
        if not hasattr(self, "wm_mgr") or self.wm_mgr is None:
            self.wm_mgr = WorkingMemoryManager(
                workspace_root=str(BASE_DIR),
                session_dir=str(BASE_DIR / "sessions"),
                profile=prof_name,
            )

        # If current conversation_history is empty, load from the selected session target in dropdown!
        hist_to_use = list(self.conversation_history)
        sandbox_dir = self.harness.isolated_output_dir if self.harness else ""
        if not hist_to_use and self.session_logger:
            target_spec = str(self.cmb_resume_target.currentData() or "latest")
            skip_n = int(self.spn_skip_turns.value())
            loaded_h, det_prj = self.session_logger.load_resumed_history(
                session_spec=target_spec,
                max_recent_turns=12,
                skip_last_turns=skip_n,
            )
            hist_to_use = loaded_h
            if det_prj:
                sandbox_dir = str(BASE_DIR / "isolated_output" / det_prj)

        self.lbl_conn_status.setText("🧠 正在滑動濃縮對話脈絡與沙盒程式碼至 L1 工作記憶...")
        QApplication.processEvents()

        md_text, status_msg = self.wm_mgr.condense_working_memory(
            conversation_history=hist_to_use,
            sandbox_dir=sandbox_dir,
            server_url=self.inp_url.text().strip(),
            model_id=self.inp_model.text().strip(),
            use_llm_sliding_window=True,
        )
        # Re-inject latest working memory into active system_prompt immediately!
        prof = load_profile_config(str(BASE_DIR / "configs/model_profiles.json"), prof_name)
        base_sys = prof.get(
            "system_prompt",
            "你是數位黑貓 Kuro (小黑)，Leon (主人) 的 S-tier AI Familiar。奉行負熵煉金術：交付結果大於文字排場，先讀碼後推論。",
        )
        if self.chk_wm_inject.isChecked():
            wm_inj, _ = self.wm_mgr.build_system_prompt_injection(sandbox_dir)
            self.system_prompt = base_sys + wm_inj

        self.lbl_conn_status.setText(status_msg)
        self.txt_chat.appendPlainText(
            f"\n======================================================================\n"
            f"{status_msg}\n"
            f"======================================================================\n"
            f"{md_text[:1800]}\n...(完整內容已寫入 sessions/working_memory_{prof_name}.md)...\n"
        )

    @staticmethod
    def _bootstrap_rust_sandbox_skeleton(sandbox_path: Path) -> None:
        """
        Ensures every fresh timestamped sandbox (`isolated_output/prj_YYYYMMDD_HHMMSS_<hash>`)
        has its own `src/` directory and golden-anchor `Cargo.toml` + `.cargo/config.toml`
        so `cargo check` / `cargo build --release` works out-of-the-box without touching older projects!
        """
        try:
            (sandbox_path / "src").mkdir(parents=True, exist_ok=True)
            cargo_toml = sandbox_path / "Cargo.toml"
            if not cargo_toml.exists():
                cargo_toml.write_text(
                    '[package]\n'
                    'name = "ntm_rsi_core"\n'
                    'version = "0.1.0"\n'
                    'edition = "2021"\n\n'
                    '[lib]\n'
                    'name = "ntm_rsi_core"\n'
                    'crate-type = ["cdylib"]\n\n'
                    '# 【黃金相容錨點 (Version Anchors)】：\n'
                    '# 請保持 pyo3 = "0.20", numpy = "0.20", ndarray = "0.15" 三者版本配對。\n'
                    '[dependencies]\n'
                    'pyo3 = { version = "0.20", features = ["extension-module"] }\n'
                    'numpy = "0.20"\n'
                    'ndarray = { version = "0.15", features = ["rayon"] }\n'
                    'rand = "0.8"\n'
                    'rayon = "1.8"\n'
                    'parking_lot = "0.12"\n'
                    'smallvec = "1.13"\n'
                    'rustc-hash = "1.1"\n'
                    'serde = { version = "1.0", features = ["derive"] }\n',
                    encoding="utf-8",
                )
            # Symlink or share target cache / Cargo.lock from template if available for instant offline compile
            tmpl_lock = BASE_DIR / "isolated_output" / "prj_20261030_220710_3f6f5c57" / "Cargo.lock"
            if tmpl_lock.exists() and not (sandbox_path / "Cargo.lock").exists():
                shutil.copy2(str(tmpl_lock), str(sandbox_path / "Cargo.lock"))
            tmpl_cargo_cfg = BASE_DIR / "isolated_output" / "prj_20261030_220710_3f6f5c57" / ".cargo"
            if tmpl_cargo_cfg.exists() and not (sandbox_path / ".cargo").exists():
                shutil.copytree(str(tmpl_cargo_cfg), str(sandbox_path / ".cargo"))
        except Exception:
            pass

    def _manual_save_snapshot(self) -> None:
        if self.ckpt_mgr and self.rust_router:
            self.ckpt_mgr.save_state(self.rust_router, expert_pool=self.expert_pool, tokens_generated=len(self.conversation_history))
            self.lbl_conn_status.setText(f"💾 已手動存檔至 checkpoints/latest_{self.cmb_profile.currentText()}.safetensors")

    def _ensure_hypercube_window(self) -> HypercubeTaskMonitorWindow:
        if not hasattr(self, "hypercube_win") or self.hypercube_win is None:
            self.hypercube_win = HypercubeTaskMonitorWindow(
                workspace_root=str(BASE_DIR),
                get_server_info_fn=lambda: (self.inp_url.text().strip(), self.inp_model.text().strip()),
            )
        return self.hypercube_win

    def open_hypercube_task_window(self) -> None:
        win = self._ensure_hypercube_window()
        win.refresh_registry_ui()
        win.show()
        win.raise_()
        win.activateWindow()

    def quick_start_hypercube_evo(self) -> None:
        win = self._ensure_hypercube_window()
        win.show()
        win.raise_()
        win.start_evolution_loop()
        self.lbl_conn_status.setText("🧊 已啟動 4D 超立方體無性生殖演化循環 (詳見獨立任務監控視窗)！")

    def quick_pause_hypercube_evo(self) -> None:
        win = self._ensure_hypercube_window()
        win.request_graceful_pause()
        self.lbl_conn_status.setText("⏸️ 已對超立方體演化循環發送【優雅暫停 (Graceful Stop)】訊號...")

    def _on_vram_timer_tick(self) -> None:
        is_inferring = self.worker is not None and self.worker.isRunning()
        self.fly_scope.refresh_vram_textures(force_gpu_sync=is_inferring)
        if not is_inferring:
            fatigues = [x[2] for x in self.fly_scope.exp_states] if self.fly_scope.exp_states else [0.0] * 8
            self.eeg_scope.idle_tick(self.fly_scope.exp_norms, fatigues)

    def paste_and_send(self) -> None:
        cb = QApplication.clipboard()
        txt = (cb.text() or "").strip()
        if txt:
            self.inp_prompt.setPlainText(txt)
            self.send_prompt()

    def _fire_preset(self, text: str) -> None:
        self.inp_prompt.setPlainText(text)
        self.send_prompt()

    def _on_loop_mode_changed(self) -> None:
        if self.hybrid_engine is None:
            return
        mode_key = str(self.cmb_loop_mode.currentData() or "mode1_vram")
        alpha_val = float(self.spn_moe_alpha.value())
        if mode_key == "mode1_vram":
            self.hybrid_engine.vram_live_gate_enabled = True
            self.hybrid_engine.vram_live_alpha = alpha_val
            self.hybrid_engine.micro_chunk_tokens = 0
            self.hybrid_engine.push_live_vram_gate_async(enabled=True, alpha=alpha_val)
            self.lbl_conn_status.setText(f"🔥 [熱插拔切換] 方案1：TensorRT C++ VRAM 即時信箱直通 ON (α={alpha_val:.2f})")
        elif mode_key == "mode2_micro":
            self.hybrid_engine.vram_live_gate_enabled = False
            self.hybrid_engine.micro_chunk_tokens = int(self.spn_micro_chunk.value())
            self.hybrid_engine.push_live_vram_gate_async(enabled=False, alpha=0.0)
            self.lbl_conn_status.setText(f"⚡ [熱插拔切換] 方案2：HTTP KV-Cache 微切片閉環 ON (Chunk={self.hybrid_engine.micro_chunk_tokens})")
        else:
            self.hybrid_engine.vram_live_gate_enabled = False
            self.hybrid_engine.micro_chunk_tokens = 0
            self.hybrid_engine.push_live_vram_gate_async(enabled=False, alpha=0.0)
            self.lbl_conn_status.setText("⚪ [熱插拔切換] 模式0：純觀測伴隨模式 (已瞬時關閉所有 Logit 注入！)")

    def _on_upload_vision_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "選擇要上傳推論的圖片 (可多選 1~3 張)", str(BASE_DIR), "Images (*.jpg *.jpeg *.png *.webp *.bmp)"
        )
        if not files:
            return
        self.attached_images = [Path(f) for f in files if Path(f).is_file()]
        self._update_vision_attachment_ui()

    def _on_load_demo_vision_image(self) -> None:
        out_dir = BASE_DIR / "exports" / "vision_tests"
        suite = generate_synthetic_directional_suite(out_dir)
        if suite:
            self.attached_images = [suite[0]]
            self._update_vision_attachment_ui()

    def _on_clear_vision_images(self) -> None:
        self.attached_images.clear()
        self.lbl_vision_info.setText("📎 尚未附加圖片")
        self.lbl_vision_thumb.setPixmap(QPixmap())
        self.lbl_vision_thumb.setText("🖼️ 縮圖預覽")

    def _update_vision_attachment_ui(self) -> None:
        if not self.attached_images:
            self._on_clear_vision_images()
            return
        names = ", ".join(p.name for p in self.attached_images)
        self.lbl_vision_info.setText(f"📎 已附 {len(self.attached_images)} 張: {names[:28]}")
        self.last_result_image_path = self.attached_images[-1]
        self.btn_download_img.setEnabled(True)
        pix = QPixmap(str(self.attached_images[-1]))
        if not pix.isNull():
            self.lbl_vision_thumb.setPixmap(
                pix.scaled(112, 74, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

    def _on_download_vision_image(self) -> None:
        src = self.last_result_image_path or (self.attached_images[-1] if self.attached_images else None)
        if not src or not src.exists():
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "另存/下載視覺結果圖片", str(BASE_DIR / src.name), "JPEG/PNG (*.jpg *.png)"
        )
        if dest:
            shutil.copy2(str(src), str(dest))
            self.lbl_conn_status.setText(f"💾 已將影像另存至: {dest}")

    def send_prompt(self) -> None:
        prompt = self.inp_prompt.toPlainText().strip()
        v_mode = str(self.cmb_vision_mode.currentData() or "free")
        if self.attached_images and not prompt:
            prompt = DIRECTIONAL_PROMPTS.get(
                v_mode if v_mode != "free" else "json_bbox",
                DIRECTIONAL_PROMPTS["json_bbox"],
            )
        elif self.attached_images and v_mode != "free":
            prompt = DIRECTIONAL_PROMPTS.get(v_mode, prompt) + "\n" + prompt

        if not prompt or self.hybrid_engine is None:
            return
        if self.worker and self.worker.isRunning():
            return

        # Sync latest UI spinbox parameters & auto-probe model ID if user changed URL
        self.hybrid_engine.max_tokens = int(self.spn_max_toks.value())
        self.hybrid_engine.temperature = float(self.spn_temp.value())
        self.hybrid_engine.shy_lambda = float(self.spn_shy.value())
        self._on_loop_mode_changed()
        self.hybrid_engine.server_url = self.inp_url.text().strip().rstrip("/")
        self._auto_probe_server_model()
        self.hybrid_engine.model_id = self.inp_model.text().strip()

        img_tag = f" [🖼️ 附圖: {', '.join(p.name for p in self.attached_images)}]" if self.attached_images else ""
        self.inp_prompt.clear()
        self.btn_send.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.txt_chat.appendPlainText(f"\nLeon{img_tag} >> {prompt}\nKuro ({self.hybrid_engine.model_id}) >> ")

        self._active_turn_images = list(self.attached_images)
        self.worker = StreamInferenceWorker(
            engine=self.hybrid_engine,
            session_logger=self.session_logger,
            user_prompt=prompt,
            system_inst=self.system_prompt,
            conversation_history=self.conversation_history,
            attached_images=self._active_turn_images,
        )
        self._pending_user_prompt = prompt
        self.worker.sig_token.connect(self._on_stream_token)
        self.worker.sig_status.connect(lambda msg: self.lbl_conn_status.setText(msg))
        self.worker.sig_finished.connect(self._on_stream_finished)
        self.worker.sig_error.connect(self._on_stream_error)
        self.worker.start()

    def stop_generation(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.request_abort()

    def _on_stream_token(self, piece: str, tid: int, entropy: float, sel_exps: list, weights: list) -> None:
        if piece:
            cursor = self.txt_chat.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(piece)
            self.txt_chat.setTextCursor(cursor)
            self.txt_chat.ensureCursorVisible()

        self.fly_scope.notify_token_activity(tid, entropy, sel_exps)
        fatigues = [x[2] for x in self.fly_scope.exp_states] if self.fly_scope.exp_states else [0.0] * 8
        self.eeg_scope.push_token_wave(entropy, sel_exps, weights, self.fly_scope.exp_norms, fatigues)

    def _on_stream_finished(self, full_text: str, total_toks: int, elapsed: float, ttft_ms: float) -> None:
        tps = total_toks / max(0.001, elapsed)
        bbox_note = ""
        if getattr(self, "_active_turn_images", None):
            bbox_img, bbox_list = parse_and_draw_bboxes(self._active_turn_images[-1], full_text)
            if bbox_img and bbox_list:
                self.last_result_image_path = bbox_img
                self.btn_download_img.setEnabled(True)
                bbox_note = f"\n🎯 [BBox 自動繪製完成] 共框出 {len(bbox_list)} 個目標 ➔ 已存檔 `{bbox_img}` (可點擊『💾 下載/另存結果圖』)"
                pix = QPixmap(str(bbox_img))
                if not pix.isNull():
                    self.lbl_vision_thumb.setPixmap(
                        pix.scaled(112, 74, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    )
        self.txt_chat.appendPlainText(
            f"\n----------------------------------------------------------------------\n"
            f"🏁 [Turn Complete] Tokens: {total_toks} | Speed: {tps:.2f} tok/s | TTFT: {ttft_ms:.1f} ms | Elapsed: {elapsed:.2f}s"
            + bbox_note
        )
        self.conversation_history.append({"role": "user", "content": self._pending_user_prompt})
        self.conversation_history.append({"role": "assistant", "content": full_text})
        if hasattr(self, "wm_mgr") and self.wm_mgr and self.harness:
            saved_arts = self.harness.extract_and_save_artifacts(full_text)
            self.wm_mgr.extract_and_update_from_turn(
                user_prompt=self._pending_user_prompt,
                assistant_reply=full_text,
                saved_artifacts=saved_arts,
                sandbox_dir=self.harness.isolated_output_dir,
            )
        if self.ckpt_mgr and self.rust_router:
            self.ckpt_mgr.save_state(self.rust_router, expert_pool=self.expert_pool, tokens_generated=total_toks)
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_conn_status.setText(f"✅ 完成推論 ({tps:.2f} tok/s) — 突觸快照與 L1 工作記憶已自動保存！")

    def _on_stream_error(self, err_msg: str) -> None:
        self.txt_chat.appendPlainText(f"\n❌ [Error]: {err_msg}\n")
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_conn_status.setText(f"❌ 連線或推論異常: {err_msg[:60]}")


def main() -> None:
    app = QApplication(sys.argv)
    win = LocalSleepMoETacticalWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
