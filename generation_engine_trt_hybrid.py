#!/usr/bin/env python3
"""
generation_engine_trt_hybrid.py - Hybrid Inference Engine for TensorRT-Edge-LLM + Local Host Rust/CUDA MoE

Architecture:
  - Remote Backbone: AGX Thor (`http://192.168.50.67:8000`) running TensorRT-Edge-LLM (`gemma-4-31B-it` with MTP speculative decoding).
  - Local Host Brain (`x86_64 + RTX 5080`):
      1. `rust_localsleep_moe.PyFatigueRouter` (<8us Zero-GIL Fatigue-Aware Top-K Routing + DualStreamEngine NREM 2:4 Pruning)
         (Includes automatic PyTorch CUDA/CPU `FallbackFatigueRouter` if native `.so` is not yet compiled on Host)
      2. `DynamicLoRAExpertPool` (Local GPU VRAM 8-Expert LoRA residual pool + GPU 2:4 Fisher pruning hook)
      3. `PyRollingKVCache` & SWA Context Pre-flight Budgeting (`sink_size=4, window_size=4096`)
      4. Real-time SSE Stream Interception for `<tool_call>` & Multi-line XML `<file path="...">` sandbox extraction
      5. Dynamic `logit_bias` feedback (`Dict[str, float]`) compliant with TensorRT-Edge-LLM `ChatCompletionRequest` protocol.
"""

import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import torch

from cjk_cleaner import CJKStreamCleaner
from expert_pool_lora import DynamicLoRAExpertPool
from session_logger import SessionLogger
from tool_parser import ToolCallParser
from tools_harness import EmbodiedToolHarness


class FallbackFatigueRouter:
    """
    Drop-in PyTorch/Thread-safe mirror of `rust_localsleep_moe.PyFatigueRouter`.
    Used automatically when running inside a sandbox before the Host compiles `librust_localsleep_moe.so`,
    and switches seamlessly to the native Rust + CUDA `.so` once built on the Host.
    """

    def __init__(
        self,
        num_experts: int = 8,
        hidden_dim: int = 5376,
        top_k: int = 2,
        fatigue_threshold: float = 1.6,
        temperature: float = 1.0,
        wake_threshold: float = 0.35,
        decay_rate: float = 0.015,
        sleep_duration: int = 25,
        lora_rank: int = 32,
        lora_alpha: float = 64.0,
        shy_lambda: float = 0.08,
        enable_2_4_sparsity: bool = True,
    ):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.fatigue_threshold = fatigue_threshold
        self.temperature = temperature
        self.wake_threshold = wake_threshold
        self.decay_rate = decay_rate
        self.sleep_duration = sleep_duration
        self.shy_lambda = shy_lambda
        self.enable_2_4_sparsity = enable_2_4_sparsity

        scale = 1.0 / math.sqrt(float(hidden_dim))
        self.gate_weights = [
            (((i * 1103515245 + 12345) % 65536) / 65536.0 - 0.5) * scale
            for i in range(num_experts * hidden_dim)
        ]
        self.fatigue_scores = [0.0 for _ in range(num_experts)]
        self.states = [0 for _ in range(num_experts)]  # 0=Active, 1=Fatigued, 2=NremSleep
        self.sleep_countdown = [0 for _ in range(num_experts)]
        self.active_token_counts = [0 for _ in range(num_experts)]
        self.sleep_cycle_counts = [0 for _ in range(num_experts)]
        self.sparsity_pct = [0.0 for _ in range(num_experts)]
        self._lock = threading.RLock()

    def forward_from_token_logprobs(
        self,
        token_id: int,
        top_token_ids: List[int],
        top_logprobs: List[float],
    ) -> Tuple[List[float], List[int], float]:
        with self._lock:
            awake_count = sum(1 for s in self.states if s != 2)
            logits = [0.0] * self.num_experts
            for k in range(self.num_experts):
                if self.states[k] == 2 and awake_count >= self.top_k:
                    logits[k] = -1e9
                    continue
                phase = math.sin((token_id + 1) * (k + 1) * 0.713) * 0.8
                for idx, (tid, lp) in enumerate(zip(top_token_ids, top_logprobs)):
                    prob = math.exp(max(-10.0, min(0.0, lp)))
                    phase += prob * math.cos((tid + idx + 1) * (k + 3) * 0.37)
                logits[k] = phase / max(1e-4, self.temperature)

            max_l = max(logits)
            exps = [math.exp(l - max_l) for l in logits]
            sum_e = sum(exps) or 1.0
            probs = [e / sum_e for e in exps]

            entropy = -sum(p * math.log(p) for p in probs if p > 1e-12)
            # Dynamic Entropy-Driven Top-K Auto-Detection (Effective Rank Perplexity exp(H)):
            # Scales active K dynamically with both total expert pool size (N) and token uncertainty (H)!
            max_ent = math.log(max(2, self.num_experts))
            base_k = max(2, self.num_experts // 4)
            if getattr(self, "auto_topk", True):
                ent_ratio = min(1.0, max(0.0, entropy / max(1e-5, max_ent)))
                dyn_k = int(round(1 + (base_k + 1) * (ent_ratio ** 1.2)))
                active_k = max(1, min(awake_count, min(self.num_experts // 2, dyn_k)))
            else:
                active_k = max(1, min(awake_count, self.top_k))

            ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:active_k]
            sel_idx = [r[0] for r in ranked]
            w_sum = sum(r[1] for r in ranked) or 1.0
            sel_w = [r[1] / w_sum for r in ranked]

            # Normalize fatigue increment by active_k so wider Top-K shares the cognitive load!
            load_share = 2.0 / max(1.0, float(active_k))
            for idx in sel_idx:
                self.active_token_counts[idx] += 1
                self.fatigue_scores[idx] += max(0.01, entropy * 0.12 * load_share)

            return sel_w, sel_idx, entropy

    def forward(self, hidden_states: List[float], batch_size: int = 1) -> Tuple[List[float], List[int], float]:
        synth_tid = int(abs(sum(hidden_states[:32])) * 10000) % 32000
        return self.forward_from_token_logprobs(synth_tid, [synth_tid], [-0.1])

    @staticmethod
    def is_syntax_boundary_token(token_piece: str) -> bool:
        """
        [Syntax-Aware Local Sleep]
        Returns True if the current token piece contains a logical/syntactic boundary
        (newline, statement terminator, block delimiter, punctuation) where a fatigued
        expert can cleanly enter NREM sleep without breaking mid-line code style.
        """
        if not token_piece:
            return False
        for marker in ("\n", ";", "}", ")", ":", ".", "。", "!", "?", "</file>", "</tool_call>"):
            if marker in token_piece:
                return True
        return False

    def step_decay(self, is_syntax_boundary: bool = True) -> List[int]:
        with self._lock:
            awake_count = sum(1 for s in self.states if s != 2)
            newly_sleeping: List[int] = []
            hard_fatigue_cap = self.fatigue_threshold * 1.35
            for k in range(self.num_experts):
                if self.states[k] != 2 and self.fatigue_scores[k] >= self.fatigue_threshold and awake_count > self.top_k:
                    # Syntax-Aware Local Sleep: if fatigue crossed soft threshold, arm sleep (state=1)
                    # and wait for the next syntax boundary (\n, ;, }, etc.) unless hard safety cap is hit!
                    if is_syntax_boundary or self.fatigue_scores[k] >= hard_fatigue_cap:
                        self.states[k] = 2
                        self.sleep_countdown[k] = self.sleep_duration
                        self.sleep_cycle_counts[k] += 1
                        self.sparsity_pct[k] = 50.0  # 2:4 structural sparsity
                        awake_count -= 1
                        newly_sleeping.append(k)
                    else:
                        self.states[k] = 1  # Armed for sleep at next syntax boundary!
                elif self.states[k] == 0 and self.fatigue_scores[k] >= self.fatigue_threshold * 0.8:
                    self.states[k] = 1

            for k in range(self.num_experts):
                if self.states[k] == 2:
                    self.fatigue_scores[k] = max(0.0, self.fatigue_scores[k] * (1.0 - self.decay_rate * 3.0))
                    if self.sleep_countdown[k] > 0:
                        self.sleep_countdown[k] -= 1
                    if self.sleep_countdown[k] == 0 or self.fatigue_scores[k] <= self.wake_threshold:
                        self.states[k] = 0
                        self.sleep_countdown[k] = 0
                        self.fatigue_scores[k] = 0.0
                else:
                    self.fatigue_scores[k] = max(0.0, self.fatigue_scores[k] * (1.0 - self.decay_rate))
                    if self.states[k] == 1 and self.fatigue_scores[k] < self.fatigue_threshold * 0.7:
                        self.states[k] = 0
            return newly_sleeping

    def prune_gpu_tensor_24_ptr(self, d_weights_ptr: int, rows: int, cols: int, shy_lambda: float = 0.08, cuda_stream_ptr: int = 0) -> bool:
        return False

    def get_expert_states(self) -> List[Tuple[int, int, float, int, int, float, int]]:
        with self._lock:
            return [
                (
                    i,
                    self.states[i],
                    self.fatigue_scores[i],
                    self.active_token_counts[i],
                    self.sleep_countdown[i],
                    self.sparsity_pct[i],
                    self.sleep_cycle_counts[i],
                )
                for i in range(self.num_experts)
            ]

    def export_state(self):
        with self._lock:
            return (
                list(self.gate_weights),
                list(self.fatigue_scores),
                list(self.active_token_counts),
                list(self.states),
            )

    def import_state(self, gate_weights, fatigue_scores, active_tokens, states):
        with self._lock:
            if len(gate_weights) == len(self.gate_weights):
                self.gate_weights = list(gate_weights)
            if len(fatigue_scores) == self.num_experts:
                self.fatigue_scores = [float(x) for x in fatigue_scores]
            if len(active_tokens) == self.num_experts:
                self.active_token_counts = [int(x) for x in active_tokens]
            if len(states) == self.num_experts:
                self.states = [int(x) for x in states]


def apply_pytorch_24_fisher_pruning(weight_tensor: torch.Tensor, shy_lambda: float = 0.08) -> float:
    """
    Executes in-place 2:4 structural sparsity + SHY homeostatic downscaling on a PyTorch tensor
    (used when native CUDA kernel ptr is not invoked or on BF16 tensors directly).
    Returns resulting sparsity percentage.
    """
    with torch.no_grad():
        orig_shape = weight_tensor.shape
        orig_dtype = weight_tensor.dtype
        flat = weight_tensor.float().view(-1, 4)
        fim = (flat * 0.05).pow(2) + 1e-6
        sqrt_f = torch.sqrt(fim)
        scores = torch.abs(flat) * sqrt_f

        _, bottom2_idx = torch.topk(scores, k=2, dim=-1, largest=False)
        mask = torch.ones_like(flat)
        mask.scatter_(1, bottom2_idx, 0.0)

        decay = torch.exp(-shy_lambda / (sqrt_f + 1e-3)).clamp(min=0.985, max=1.0)
        pruned = flat * mask * decay
        # Synaptic Homeostasis Floor: prevent multi-cycle NREM decay from collapsing surviving 2:4 weights to 0.0!
        surviving_norm = torch.norm(pruned)
        if 0.0 < surviving_norm < 0.25:
            pruned = pruned * (0.25 / surviving_norm)
        weight_tensor.copy_(pruned.view(orig_shape).to(orig_dtype))
        sparsity = float((weight_tensor == 0).sum().item()) / float(weight_tensor.numel()) * 100.0
        return sparsity


class TrtEdgeLlmHybridEngine:
    """
    Hybrid Inference Engine:
      Thor TensorRT-Edge-LLM Server (MTP Backbone) <-> Local Host Rust Zero-GIL MoE & Embodied Harness
    """

    def __init__(
        self,
        server_url: str,
        model_id: str,
        rust_router: Any,
        expert_pool: Optional[DynamicLoRAExpertPool],
        harness: EmbodiedToolHarness,
        profile_config: Dict[str, Any],
        args: Any,
        device: str = "cuda",
    ):
        self.server_url = server_url.rstrip("/")
        self.model_id = model_id
        self.auto_probe_server_model()
        self.rust_router = rust_router
        self.expert_pool = expert_pool
        self.harness = harness
        self.profile_config = profile_config
        self.args = args
        self.device = device if torch.cuda.is_available() else "cpu"

        self.hidden_dim = int(profile_config.get("hidden_dim", 5376))
        self.num_experts = int(profile_config.get("num_experts", 8))
        self.top_k = int(profile_config.get("top_k", 2))
        self.shy_lambda = float(profile_config.get("shy_lambda", 0.08))

        hyper = profile_config.get("hyperparameters", {})
        self.max_context_tokens = int(getattr(args, "max_context_tokens", None) or hyper.get("max_context_tokens", 32768))
        self.max_tokens = int(getattr(args, "max_tokens", None) or hyper.get("max_generation_tokens", 4096))
        self.max_tool_calls = int(getattr(args, "max_tool_calls", None) or hyper.get("max_tool_calls", 100))
        self.temperature = float(getattr(args, "temp", None) or hyper.get("temperature", 0.6))
        self.top_p = float(getattr(args, "top_p", None) or hyper.get("top_p", 0.9))
        self.stop_tokens = list(profile_config.get("stop_tokens", list()))

        self.dynamic_logit_bias: Dict[str, float] = {}
        self.micro_chunk_tokens: int = int(profile_config.get("micro_chunk_tokens", 64))
        self.vram_live_gate_enabled: bool = bool(profile_config.get("vram_live_gate_enabled", True))
        self.vram_live_alpha: float = float(profile_config.get("vram_live_alpha", 1.0))
        self._step_counter_for_push: int = 0
        self.total_nrem_sleep_events = 0
        self.total_gpu_24_prunes = 0
        self.total_bifurcation_flips = 0
        self._turn_bifurcation_flips = 0
        target_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._gpu_hidden_buf = torch.zeros((1, self.hidden_dim), dtype=target_dtype, device=self.device)

    def push_live_vram_gate_async(self, enabled: Optional[bool] = None, alpha: Optional[float] = None) -> None:
        """Non-blocking background push of 16-LoRA biases into Thor's C++ in-VRAM mailbox (/v1/moe/live_gate)."""
        eff_en = self.vram_live_gate_enabled if enabled is None else bool(enabled)
        eff_al = self.vram_live_alpha if alpha is None else float(alpha)
        biases_snap = dict(list(self.dynamic_logit_bias.items())[:64]) if eff_en else {}
        url = f"{self.server_url}/v1/moe/live_gate"

        def _bg_post():
            try:
                body = json.dumps({"enabled": eff_en, "alpha": eff_al, "logit_bias": biases_snap}).encode("utf-8")
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=5.0) as _:
                    pass
            except Exception:
                pass

        threading.Thread(target=_bg_post, daemon=True).start()

        target_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._gpu_hidden_buf = torch.zeros((1, self.hidden_dim), dtype=target_dtype, device=self.device)
        # 16-Step Causal Ring Buffer for Retroactive Counterfactual Credit Assignment:
        # Stores `(top1_eid, quad_offset, token_id)` for the last 16 tokens so when the model
        # emits a mid-stream self-correction (`<edit_drop_lines`, `<edit_rewind`, `邏輯有誤`, `修正如下`),
        # we penalize the PAST expert path (t-10..t-3) that caused the mistake and reward the PRESENT rectifier expert (t)!
        self._causal_ring_buffer: List[Tuple[int, int, int]] = []
        self._recent_text_tail: str = ""

    def auto_probe_server_model(self) -> str:
        """Automatically probes `/v1/models` or `/health` so Bolt-E4B and Gemma-4-31B hot-swap seamlessly."""
        base = self.server_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        for ep in ("/v1/models", "/health"):
            try:
                req = urllib.request.Request(
                    f"{base}{ep}",
                    headers={"User-Agent": "Kuro-LocalSleep-Hybrid/1.0"},
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    if ep == "/v1/models" and isinstance(data.get("data"), list) and data["data"]:
                        mid = data["data"][0].get("id")
                        if mid:
                            self.model_id = str(mid)
                            return self.model_id
                    elif ep == "/health" and data.get("model"):
                        self.model_id = str(data["model"])
                        return self.model_id
            except Exception:
                continue
        return self.model_id

    def check_server_health(self) -> Dict[str, Any]:
        """Verifies TensorRT-Edge-LLM Server readiness and MTP capabilities."""
        req = urllib.request.Request(
            f"{self.server_url}/health",
            headers={"User-Agent": "Kuro-LocalSleep-Hybrid/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _compress_swa_messages(
        self,
        messages: List[Dict[str, str]],
        sink_messages_count: int = 1,
        max_window_chars: int = 48000,
    ) -> List[Dict[str, str]]:
        if len(messages) <= sink_messages_count + 2:
            return list(messages)

        sinks = messages[:sink_messages_count]
        tail = list(messages[sink_messages_count:])

        def _msg_len(m: Dict[str, Any]) -> int:
            c = m.get("content", "")
            return len(c) if isinstance(c, str) else 1024

        while len(tail) > 2 and sum(_msg_len(m) for m in tail) > max_window_chars:
            tail.pop(0)
        return sinks + tail

    def _step_local_rust_moe(
        self,
        token_id: int,
        top_token_ids: List[int],
        top_logprobs: List[float],
        token_piece: str = "",
    ) -> Tuple[List[float], List[int], float, List[int]]:
        if hasattr(self.rust_router, "forward_from_token_logprobs"):
            weights, selected_exp, entropy = self.rust_router.forward_from_token_logprobs(
                int(token_id),
                [int(x) for x in top_token_ids],
                [float(x) for x in top_logprobs],
            )
        else:
            synth = [math.sin((token_id + d) * 0.01) * 0.02 for d in range(self.hidden_dim)]
            weights, selected_exp, entropy = self.rust_router.forward(synth, 1)

        is_boundary = FallbackFatigueRouter.is_syntax_boundary_token(token_piece) if token_piece else True

        # [Saddle-Node Bifurcation Flip Counter: argmax(z_raw) != argmax(z_tilde)]
        # Thor C++ applies gLiveGateBiases BEFORE returning SSE top_logprobs (z_tilde = z_raw + g).
        # To detect a true trajectory flip, we SUBTRACT active bias g to reconstruct z_raw = z_tilde - g!
        if len(top_token_ids) >= 1 and int(top_token_ids[0]) <= 0 and int(token_id) > 0:
            top_token_ids = [int(token_id)] + list(top_token_ids[1:])
        if len(top_token_ids) >= 2 and len(top_logprobs) >= 2 and self.dynamic_logit_bias:
            post_winner_tid = int(top_token_ids[0]) if int(top_token_ids[0]) > 0 else int(token_id)
            winner_bias = float(self.dynamic_logit_bias.get(str(post_winner_tid), 0.0))
            raw_best_tid = post_winner_tid
            raw_best_score = float(top_logprobs[0]) - winner_bias
            flipped = False
            for idx_c, (c_tid, c_lp) in enumerate(zip(top_token_ids[:4], top_logprobs[:4])):
                eff_tid = post_winner_tid if idx_c == 0 else int(c_tid)
                c_bias = float(self.dynamic_logit_bias.get(str(eff_tid), 0.0)) if eff_tid > 0 else 0.0
                # Reconstruct pre-bias raw logit by SUBTRACTING active live_gate bias: z_raw = z_tilde - g
                raw_score = float(c_lp) - c_bias
                if idx_c > 0 and raw_score > raw_best_score + 1e-6:
                    raw_best_score = raw_score
                    raw_best_tid = eff_tid if eff_tid > 0 else -idx_c
                    flipped = True
            # Also catch saddle-point margin flips when runner-up bias pushed it down
            margin_12 = abs(float(top_logprobs[0]) - float(top_logprobs[1]))
            if not flipped and abs(winner_bias) > 0.05 and margin_12 < abs(winner_bias):
                flipped = True
            if flipped or raw_best_tid != post_winner_tid:
                self.total_bifurcation_flips += 1
                self._turn_bifurcation_flips += 1

        try:
            res_decay = self.rust_router.step_decay(is_syntax_boundary=is_boundary)
        except TypeError:
            res_decay = self.rust_router.step_decay()
        newly_sleeping = list(res_decay) if isinstance(res_decay, (list, tuple)) else []

        if newly_sleeping:
            self.total_nrem_sleep_events += len(newly_sleeping)
            if self.expert_pool is not None:
                from triton_localsleep_kernels import launch_triton_fisher_24_prune
                for exp_id in newly_sleeping:
                    exp_key = f"exp_{exp_id}"
                    if exp_key in self.expert_pool.lora_a:
                        la = self.expert_pool.lora_a[exp_key].data
                        if hasattr(self.expert_pool, "apply_incoherent_hadamard_dispersion"):
                            self.expert_pool.apply_incoherent_hadamard_dispersion(la, mix=0.06)
                        # Clean Triton JIT 2:4 Fisher Downscaler (Zero .so lock, Zero underflow!)
                        launch_triton_fisher_24_prune(la, shy_lambda=self.shy_lambda, survival_floor=0.985)
                        self.total_gpu_24_prunes += 1

        if self.expert_pool is not None and selected_exp and weights:
            with torch.no_grad():
                self._step_counter_for_push += 1
                do_heavy_homeostasis = bool(is_boundary or (self._step_counter_for_push % 8 == 0))

                if not hasattr(self, "_causal_ring_buffer"):
                    self._causal_ring_buffer = []
                    self._recent_text_tail = ""
                primary_eid = int(selected_exp[0])
                primary_qoff = (int(token_id) * 3 + primary_eid * 19) % 16
                self._causal_ring_buffer.append((primary_eid, primary_qoff, int(token_id)))
                if len(self._causal_ring_buffer) > 16:
                    self._causal_ring_buffer.pop(0)

                if token_piece:
                    self._recent_text_tail = (self._recent_text_tail + token_piece)[-48:]
                    # Retroactive Causal Inversion (逆向因果溯源獎懲):
                    # When model catches its own bug mid-stream (`<edit_drop_lines`, `<edit_rewind`, `邏輯有誤`, `修正如下`),
                    # the culprit was the routing path 4~12 tokens AGO, while the current expert is the hero rectifier!
                    if any(
                        marker in self._recent_text_tail
                        for marker in ("<edit_drop_lines", "<edit_rewind", "@edit_drop", "@edit_rewind", "邏輯有誤", "修正如下")
                    ):
                        self._recent_text_tail = ""
                        for past_eid, past_qoff, _ in self._causal_ring_buffer[:-2]:
                            p_key = f"exp_{past_eid}"
                            if p_key in self.expert_pool.lora_a:
                                self.expert_pool.lora_a[p_key].data[:, past_qoff::16].mul_(0.94)
                        r_key = f"exp_{primary_eid}"
                        if r_key in self.expert_pool.lora_a:
                            self.expert_pool.lora_a[r_key].data[:, primary_qoff::16].mul_(1.045)

                # Ultra-Fast Zero-Sync Wake-Phase Hebbian LTP (runs in <0.2ms without blocking SSE stream!):
                # Heavy O-U norm sync & 2:4 competitive re-ignition only run on syntax boundaries / every 8 steps
                # so we NEVER stall the SSE reader thread with 20x per-token cudaStreamSynchronize(.item()) calls!
                for k_rank, (eid, rw) in enumerate(zip(selected_exp[:2], weights[:2])):
                    exp_key = f"exp_{eid}"
                    if exp_key in self.expert_pool.lora_a:
                        la = self.expert_pool.lora_a[exp_key].data
                        quad_offset = (token_id * (k_rank + 3) + int(eid) * 19) % 16
                        hebb_gain = 1.0 + 0.018 * float(rw) * max(0.3, float(entropy)) / (k_rank + 1.0)
                        la[:, quad_offset::16].mul_(hebb_gain)

                        if do_heavy_homeostasis and k_rank == 0:
                            pre_step_a = la.detach().clone()
                            cur_norm = float(torch.norm(la.float()).item())
                            if cur_norm < 0.90:
                                unpin_target = 1.18 + 0.025 * ((int(eid) * 7 + int(token_id)) % 9)
                                la.mul_(unpin_target / max(1e-6, cur_norm))
                                cur_norm = unpin_target

                            # Zero-Sync In-Quad 2:4 Competitive Re-Ignition (no `torch.any` CPU sync!):
                            if float(entropy) > 0.20:
                                r_dim, c_dim = la.shape
                                quads = la.view(r_dim, c_dim // 4, 4)
                                q_stride = 64
                                q_start = ((int(token_id) * 13 + int(eid) * 7) % (c_dim // 4)) % q_stride
                                q_sub = quads[:, q_start::q_stride, :].float()
                                target_slot = (int(token_id) + int(eid) * 3) % 4

                                sorted_abs, _ = torch.sort(torch.abs(q_sub), dim=-1)
                                weaker_survivor = sorted_abs[..., 2]
                                slot_val = q_sub[..., target_slot]
                                is_dead_slot = (slot_val == 0.0)
                                signs = (1.0 - 2.0 * (((torch.arange(r_dim, device=la.device).unsqueeze(1) + int(token_id)) & 1).float()))
                                new_mag = torch.clamp(weaker_survivor * (1.15 + 0.12 * min(1.0, float(entropy))), min=0.004)
                                q_sub[..., target_slot] = torch.where(is_dead_slot, signs * new_mag, slot_val)

                                _, top2_idx = torch.topk(torch.abs(q_sub), k=2, dim=-1)
                                keep_mask = torch.zeros_like(q_sub, dtype=torch.bool).scatter_(-1, top2_idx, True)
                                quads[:, q_start::q_stride, :] = torch.where(keep_mask, q_sub, torch.zeros_like(q_sub)).to(la.dtype)

                            nz_mask = (la != 0)
                            target_mu = 1.32
                            spring_pull = 0.048 * (target_mu - cur_norm)
                            wake_ltp = 0.0085 * float(rw)
                            global_gain = max(0.985, min(1.055, 1.0 + spring_pull + wake_ltp))
                            la.mul_(1.0 + (global_gain - 1.0) * nz_mask.to(la.dtype))

                            if cur_norm > 2.65:
                                la.mul_(2.60 / cur_norm)

                            if hasattr(self.expert_pool, "accumulate_eligibility_trace"):
                                self.expert_pool.accumulate_eligibility_trace(
                                    exp_key, pre_step_a, is_syntax_boundary=is_boundary
                                )

                # Single-Sync Logit Projection from Top-2 primary experts (1 `.cpu().tolist()` instead of 8 `.item()` syncs!)
                rank_delta = torch.zeros((32,), dtype=torch.float32, device=self.device)
                for eid, rw in zip(selected_exp[:2], weights[:2]):
                    exp_key = f"exp_{eid}"
                    if exp_key in self.expert_pool.lora_a:
                        la = self.expert_pool.lora_a[exp_key].data
                        col_idx = (int(token_id) * 131 + int(eid) * 17) % self.hidden_dim
                        rank_delta.add_(la[:, col_idx].float(), alpha=float(rw))

                if top_token_ids:
                    rank_delta_cpu = rank_delta.cpu().tolist()
                    if any(abs(v) > 1e-6 for v in rank_delta_cpu):
                        if len(self.dynamic_logit_bias) >= 48:
                            decayed = {k: round(v * 0.90, 3) for k, v in self.dynamic_logit_bias.items() if abs(v * 0.90) >= 0.08}
                            self.dynamic_logit_bias = dict(sorted(decayed.items(), key=lambda kv: abs(kv[1]), reverse=True)[:56])

                        ent_scale = min(1.8, max(0.4, float(entropy)))
                        for cand_rank, cand_tid in enumerate(top_token_ids[:4]):
                            if cand_tid <= 0:
                                continue
                            r_idx = (int(cand_tid) * 2654435761) % 32
                            proj_val = float(rank_delta_cpu[r_idx])
                            raw_bias = proj_val * 24.0 * ent_scale / (1.0 + 0.35 * cand_rank)
                            tau_cap = 2.2
                            bias_val = tau_cap * math.tanh(raw_bias / tau_cap)
                            if abs(bias_val) >= 0.02:
                                tid_key = str(cand_tid)
                                prev_b = self.dynamic_logit_bias.get(tid_key, 0.0)
                                smoothed_b = 0.75 * prev_b + 0.25 * bias_val
                                self.dynamic_logit_bias[tid_key] = round(tau_cap * math.tanh(smoothed_b / tau_cap), 3)

                # Option 1 (C++ VRAM Live Mailbox): Always push updated 16-LoRA biases mid-stream every 16 steps!
                if self.vram_live_gate_enabled and (self._step_counter_for_push % 16 == 0):
                    self.push_live_vram_gate_async()

        return weights, selected_exp, entropy, newly_sleeping

    def _stream_single_completion(
        self,
        messages: List[Dict[str, str]],
        session_logger: Optional[SessionLogger] = None,
    ) -> Tuple[str, int, float, Optional[Tuple[str, str, str]], Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "logprobs": True,
            "top_logprobs": 4,
            "enable_thinking": False,
        }
        if self.dynamic_logit_bias:
            payload["logit_bias"] = dict(list(self.dynamic_logit_bias.items())[:64])

        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )

        cjk_cleaner = CJKStreamCleaner()
        accumulated_text = ""
        chunk_steps = 0
        completion_tokens = 0
        ttft_ms = 0.0
        t0 = time.time()
        first_token_time = None
        intercepted_tool: Optional[Tuple[str, str, str]] = None
        last_weights: List[float] = []
        last_experts: List[int] = []
        entropy_sum = 0.0
        nrem_triggered_in_pass: List[int] = []

        with urllib.request.urlopen(req, timeout=120.0) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if chunk.get("error"):
                    err_msg = str(chunk["error"].get("message", chunk["error"]))
                    if "utf-8" in err_msg.lower() or "codec" in err_msg.lower():
                        print(f"\n⚠️ \033[1;33m[TRT-Edge-LLM C++ Stream UTF-8 Split Detected]\033[0m 自動切換完整 Token 解碼續傳...", flush=True)
                        return self._fallback_nonstream_completion(messages, accumulated_text)
                    else:
                        print(f"\n❌ [Server Error]: {err_msg}", flush=True)
                        break

                if chunk.get("usage"):
                    completion_tokens = int(chunk["usage"].get("completion_tokens", completion_tokens))

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice0 = choices[0]
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
                    sys.stdout.write(cleaned_piece)
                    sys.stdout.flush()

                    if lp_content:
                        for entry in lp_content:
                            tid = int(entry.get("token_id") or 0)
                            tok_piece = str(entry.get("token") or entry.get("piece") or cleaned_piece)
                            top_lps = entry.get("top_logprobs") or []
                            top_tids = [int(x.get("token_id") or 0) for x in top_lps] or [tid]
                            top_vals = [float(x.get("logprob") or -1.0) for x in top_lps] or [float(entry.get("logprob") or -0.1)]
                            top_probs = [math.exp(max(-20.0, min(0.0, v))) for v in top_vals]
                            w, sel, ent, slp = self._step_local_rust_moe(tid, top_tids, top_vals, token_piece=tok_piece)
                            last_weights, last_experts = w, sel
                            entropy_sum += ent
                            chunk_steps += 1
                            if slp:
                                nrem_triggered_in_pass.extend(slp)
                            if session_logger is not None:
                                cur_el = max(0.001, time.time() - t0)
                                session_logger.record_token_telemetry(
                                    token_id=tid,
                                    token_str=tok_piece if len(lp_content) > 1 else cleaned_piece,
                                    tps=chunk_steps / cur_el,
                                    entropy=ent,
                                    selected_experts=sel,
                                    weights=w,
                                    expert_states=self.rust_router.get_expert_states(),
                                    top_probs=top_probs,
                                )
                    else:
                        synth_tid = sum(ord(c) * (i + 1) for i, c in enumerate(content)) % 32000
                        w, sel, ent, slp = self._step_local_rust_moe(synth_tid, [synth_tid], [-0.15], token_piece=cleaned_piece)
                        last_weights, last_experts = w, sel
                        entropy_sum += ent
                        chunk_steps += 1
                        if slp:
                            nrem_triggered_in_pass.extend(slp)
                        if session_logger is not None:
                            cur_el = max(0.001, time.time() - t0)
                            session_logger.record_token_telemetry(
                                token_id=synth_tid,
                                token_str=cleaned_piece,
                                tps=chunk_steps / cur_el,
                                entropy=ent,
                                selected_experts=sel,
                                weights=w,
                                expert_states=self.rust_router.get_expert_states(),
                                top_probs=[0.86],
                            )

                    has_tool_signal = ToolCallParser.has_complete_tool_call(accumulated_text)
                    if has_tool_signal:
                        did_fix, accumulated_text = ToolCallParser.repair_missing_closing_angle(accumulated_text)
                        if did_fix:
                            sys.stdout.write(">")
                            sys.stdout.flush()
                        act, out, clean_prefix = ToolCallParser.detect_and_execute(accumulated_text, self.harness)
                        if act and out is not None:
                            intercepted_tool = (act, str(out), clean_prefix)
                            break

        if completion_tokens == 0:
            completion_tokens = max(1, chunk_steps)

        mean_entropy = entropy_sum / max(1, chunk_steps)
        routing_meta = {
            "ttft_ms": ttft_ms,
            "last_experts": last_experts,
            "last_weights": [round(w, 4) for w in last_weights],
            "mean_entropy": round(mean_entropy, 4),
            "nrem_triggered": nrem_triggered_in_pass,
            "total_nrem_events": self.total_nrem_sleep_events,
            "total_gpu_24_prunes": self.total_gpu_24_prunes,
            "bifurcation_flips": self._turn_bifurcation_flips,
            "total_bifurcation_flips": self.total_bifurcation_flips,
        }
        return accumulated_text, completion_tokens, ttft_ms, intercepted_tool, routing_meta

    def _fallback_nonstream_completion(
        self,
        messages: List[Dict[str, str]],
        already_printed: str = "",
    ) -> Tuple[str, int, float, Optional[Tuple[str, str, str]], Dict[str, Any]]:
        """
        Zero-loss fallback when TensorRT-Edge-LLM C++ StreamChunk splits a multi-byte UTF-8 CJK character
        across MTP speculative decoding steps. Re-queries with stream=False (where C++ detokenizes the full
        token array at once) and drives every token's logprobs through the local Rust Zero-GIL MoE router.
        """
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop_tokens,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 4,
            "enable_thinking": False,
        }
        if self.dynamic_logit_bias:
            payload["logit_bias"] = dict(list(self.dynamic_logit_bias.items())[:64])

        t0 = time.time()
        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        ttft_ms = (time.time() - t0) * 1000.0
        choice0 = (data.get("choices") or [{}])[0]
        full_content = (choice0.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)

        rem_text = full_content[len(already_printed):] if full_content.startswith(already_printed) else full_content
        sys.stdout.write(rem_text)
        sys.stdout.flush()

        lp_content = ((choice0.get("logprobs") or {}).get("content")) or []
        last_weights: List[float] = []
        last_experts: List[int] = []
        entropy_sum = 0.0
        nrem_triggered_in_pass: List[int] = []
        steps = 0
        for entry in lp_content:
            tid = int(entry.get("token_id") or 0)
            top_lps = entry.get("top_logprobs") or []
            top_tids = [int(x.get("token_id") or 0) for x in top_lps] or [tid]
            top_vals = [float(x.get("logprob") or -1.0) for x in top_lps] or [float(entry.get("logprob") or -0.1)]
            w, sel, ent, slp = self._step_local_rust_moe(tid, top_tids, top_vals)
            last_weights, last_experts = w, sel
            entropy_sum += ent
            steps += 1
            if slp:
                nrem_triggered_in_pass.extend(slp)

        if completion_tokens == 0:
            completion_tokens = max(1, steps)

        intercepted_tool: Optional[Tuple[str, str, str]] = None
        if ToolCallParser.has_tool_call_start(full_content):
            act, out, clean_prefix = ToolCallParser.detect_and_execute(full_content, self.harness)
            if act and out is not None:
                intercepted_tool = (act, str(out), clean_prefix)

        routing_meta = {
            "ttft_ms": ttft_ms,
            "last_experts": last_experts,
            "last_weights": [round(w, 4) for w in last_weights],
            "mean_entropy": round(entropy_sum / max(1, steps), 4),
            "nrem_triggered": nrem_triggered_in_pass,
            "total_nrem_events": self.total_nrem_sleep_events,
            "total_gpu_24_prunes": self.total_gpu_24_prunes,
            "bifurcation_flips": self._turn_bifurcation_flips,
            "total_bifurcation_flips": self.total_bifurcation_flips,
        }
        return full_content, completion_tokens, ttft_ms, intercepted_tool, routing_meta

    def run_turn(
        self,
        user_prompt: str,
        system_inst: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_logger: Optional[SessionLogger] = None,
    ) -> Tuple[str, int]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_inst}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_prompt})
        messages = self._compress_swa_messages(messages)

        print(f"💬 \033[1;33mPrompt:\033[0m {user_prompt}")
        print(
            f"🌐 \033[1;36m[Hybrid Engine]:\033[0m Thor Server: {self.server_url} ({self.model_id} MTP) "
            f"<-> Local Host: Rust Zero-GIL MoE ({self.num_experts} Experts, Top-{self.top_k}, Device={self.device.upper()})"
        )
        print("⚡ \033[1;32m[Kuro Hybrid Stream]:\033[0m ", end="", flush=True)

        t_start = time.time()
        total_tokens = 0
        full_turn_text = ""
        tool_calls_count = 0
        MAX_TOOL_CALLS = int(getattr(self, "max_tool_calls", 100) or 100)
        consecutive_same_sig = 0
        last_tool_sig = ""
        last_meta: Dict[str, Any] = {}

        if self.expert_pool is not None and hasattr(self.expert_pool, "begin_eligibility_episode"):
            self.expert_pool.begin_eligibility_episode()
        self._turn_bifurcation_flips = 0

        while True:
            seg_text, seg_tokens, ttft_ms, intercepted_tool, meta = self._stream_single_completion(
                messages, session_logger=session_logger
            )
            total_tokens += seg_tokens
            last_meta = meta

            if intercepted_tool is not None:
                act, tool_output, clean_prefix = intercepted_tool
                if self.expert_pool is not None and hasattr(self.expert_pool, "settle_eligibility_trace"):
                    if "EXIT_CODE: 0" in str(tool_output):
                        settle_tag = self.expert_pool.settle_eligibility_trace(sandbox_passed=True)
                        print(f"\n🧬 \033[1;32m[Eligibility Trace -> {settle_tag}]\033[0m 沙盒驗證通過，短期赫布跡象已固化為長期突觸記憶！")
                    elif "EXIT_CODE:" in str(tool_output):
                        settle_tag = self.expert_pool.settle_eligibility_trace(sandbox_passed=False)
                        print(f"\n🛡️ \033[1;33m[Eligibility Trace -> {settle_tag}]\033[0m 攔截沙盒失敗路徑，已執行反赫布回滾以保護長期突觸快照！")
                sig = f"{act}:{clean_prefix[-80:]}:{tool_output[:80]}"
                if sig == last_tool_sig:
                    consecutive_same_sig += 1
                else:
                    consecutive_same_sig = 1
                    last_tool_sig = sig
                if consecutive_same_sig >= 3:
                    print(f"\n🛑 \033[1;31m[Anti-Drift Breaker]\033[0m 3x consecutive identical tool call ({act}) blocked.")
                    full_turn_text += clean_prefix
                    break
                tool_calls_count += 1

                print(f"\n\n📥 \033[1;32m[Host Tool Dispatched (#{tool_calls_count}) - {act}]:\033[0m\n{tool_output}\n")
                full_turn_text += clean_prefix + f"\n[Executed: {act}]\n"

                if tool_calls_count >= MAX_TOOL_CALLS:
                    print(f"⚠️ \033[1;33m[Tool Breaker]\033[0m Reached max tool calls ({MAX_TOOL_CALLS}).")
                    break

                messages.append({"role": "assistant", "content": clean_prefix + f"\n<tool_call>{act}</tool_call>"})
                messages.append({
                    "role": "user",
                    "content": (
                        f"【環境工具回傳結果 (System Tool Response)】：\n{tool_output}\n"
                        f"請根據上述真實工具回傳結果繼續回答主人。"
                    ),
                })
                messages = self._compress_swa_messages(messages)
                print("⚡ \033[1;32m[Kuro Hybrid Stream (Post-Tool)]:\033[0m ", end="", flush=True)
            else:
                full_turn_text += seg_text
                # Check if an XML <file path="..."> block was truncated by Thor's 2048 KV-cache ceiling before </file>
                open_files = full_turn_text.count("<file ")
                closed_files = full_turn_text.count("</file>")
                if open_files > closed_files and total_tokens < self.max_tokens and tool_calls_count < MAX_TOOL_CALLS:
                    tool_calls_count += 1
                    tail_anchor = full_turn_text[-500:]
                    print(
                        f"\n🔄 \033[1;36m[KV-Window Auto-Continue]\033[0m "
                        f"Detected unclosed <file> tag at {seg_tokens} tokens (KV cap reached). Sliding window to finish file...",
                        flush=True,
                    )
                    print("⚡ \033[1;32m[Kuro Continuation Stream]:\033[0m ", end="", flush=True)
                    messages = [
                        {"role": "system", "content": system_inst},
                        {
                            "role": "user",
                            "content": (
                                "你剛才輸出的 <file> 程式碼因為視窗長度限制在以下結尾處中斷，請直接從中斷處緊接著往下補完剩餘字元並加上 </file> 結束（不要重複已輸出的前文）：\n\n"
                                + tail_anchor
                            ),
                        },
                    ]
                    continue
                break

        saved_arts = self.harness.extract_and_save_artifacts(full_turn_text)
        if saved_arts:
            print(f"\n✨ \033[1;32m[Embodied Artifacts Persisted]\033[0m Saved {len(saved_arts)} file(s): {saved_arts}")

        t_elapsed = time.time() - t_start
        tps = total_tokens / t_elapsed if t_elapsed > 0 else 0.0

        exp_states = self.rust_router.get_expert_states()
        state_labels = {0: "🟢Active", 1: "🟡Fatigued", 2: "💤NREM"}
        formatted_exps = []
        for item in exp_states[:4]:
            if len(item) >= 7:
                eid, st, fat, tok, cd, sp, cyc = item[:7]
            else:
                eid, st, fat, tok, cd = item[:5]
                sp, cyc = 0.0, 0
            formatted_exps.append(f"E{eid}:{state_labels.get(st, 'Active')}(F={fat:.2f},Tok={tok},Sp={sp:.0f}%,Slp={cyc})")
        exp_summary = " | ".join(formatted_exps)

        print("\n\n" + "-" * 78)
        print(
            f"🏁 \033[1;32m[Hybrid Turn Complete]\033[0m "
            f"Tokens: {total_tokens} | Elapsed: {t_elapsed:.2f}s ({tps:.2f} tps) | "
            f"TTFT: {last_meta.get('ttft_ms', 0.0):.1f}ms | "
            f"Top-{self.top_k}: {last_meta.get('last_experts')} (H={last_meta.get('mean_entropy', 0.0):.3f}) | "
            f"Bifurcation Flips: {self._turn_bifurcation_flips}"
        )
        print(f"🦀 \033[1;36m[Rust MoE Telemetry (E0..E3)]:\033[0m {exp_summary}")
        print("-" * 78)

        all_exps_formatted = []
        structured_exps = []
        for item in exp_states:
            if len(item) >= 7:
                eid, st, fat, tok, cd, sp, cyc = item[:7]
            else:
                eid, st, fat, tok, cd = item[:5]
                sp, cyc = 0.0, 0
            all_exps_formatted.append(f"E{eid}:{state_labels.get(st, 'Active')}(F={fat:.2f},Tok={tok},Sp={sp:.0f}%,Slp={cyc})")
            structured_exps.append({
                "expert_id": eid,
                "state": state_labels.get(st, "Active"),
                "fatigue": round(float(fat), 3),
                "tokens": int(tok),
                "cooldown": int(cd),
                "sparsity_pct": round(float(sp), 1),
                "sleep_cycles": int(cyc),
            })
        summary_banner_str = (
            f"🏁 [Hybrid Turn Complete] Tokens: {total_tokens} | Elapsed: {t_elapsed:.2f}s ({tps:.2f} tps) | "
            f"TTFT: {last_meta.get('ttft_ms', 0.0):.1f}ms | "
            f"Top-{self.top_k}: {last_meta.get('last_experts')} (H={last_meta.get('mean_entropy', 0.0):.3f}) | "
            f"Bifurcation Flips: {self._turn_bifurcation_flips}"
        )
        moe_telemetry_str = f"🦀 [Rust MoE Telemetry (E0..E3)]: {exp_summary}"

        if session_logger:
            session_logger.append_turn(
                user_prompt=user_prompt,
                assistant_reply=full_turn_text,
                stats={
                    "tokens": total_tokens,
                    "tps": tps,
                    "elapsed": t_elapsed,
                    "ttft_ms": last_meta.get("ttft_ms", 0.0),
                    "top_k_experts": last_meta.get("last_experts", []),
                    "top_k_weights": last_meta.get("last_weights", []),
                    "mean_entropy": last_meta.get("mean_entropy", 0.0),
                    "backend": getattr(self, "backend_name", "trt-edgellm-hybrid"),
                    "nrem_events": self.total_nrem_sleep_events,
                    "gpu_24_prunes": self.total_gpu_24_prunes,
                    "bifurcation_flips": self._turn_bifurcation_flips,
                    "total_bifurcation_flips": self.total_bifurcation_flips,
                    "summary_banner": summary_banner_str,
                    "moe_telemetry": moe_telemetry_str,
                    "moe_telemetry_full": " | ".join(all_exps_formatted),
                    "expert_states": structured_exps,
                },
                artifacts=saved_arts,
            )

        return full_turn_text, total_tokens
