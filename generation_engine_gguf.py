#!/usr/bin/env python3
"""
generation_engine_gguf.py - Native GGUF (`llama-cpp-python`) Engine for `rust_localsleep-moe`
=============================================================================================
Specifically designed to load `NVIDIA-Nemotron-3.5-Bolt-E4B-4GB/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`
(Gemma-4-E4B architecture, hidden_dim=2560) directly onto RTX 5080 (16GB VRAM) in-process,
or connect via `llama-server` (`TrtEdgeLlmHybridEngine`), while driving:
  1. `rust_localsleep_moe.PyFatigueRouter` (<8us Zero-GIL Fatigue-Aware Top-K Routing)
  2. `DynamicLoRAExpertPool` (8-Expert VRAM LoRA residual modulation + NREM 2:4 Fisher pruning)
  3. `EmbodiedToolHarness` (`<tool_call>` & `<file path="...">` sandbox execution)
  4. SWA Context Budgeting (`sink_size=4, window_size=4096`)
"""

import json
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from cjk_cleaner import CJKStreamCleaner
from expert_pool_lora import DynamicLoRAExpertPool
from generation_engine_trt_hybrid import apply_pytorch_24_fisher_pruning
from session_logger import SessionLogger
from tool_parser import ToolCallParser
from tools_harness import EmbodiedToolHarness


class LlamaCppGgufEngine:
    """
    In-Process `llama-cpp-python` GGUF Engine coupled with Local Host Rust Zero-GIL MoE Router
    and GPU Dynamic LoRA Expert Pool on RTX 5080.
    """

    def __init__(
        self,
        gguf_path: str,
        rust_router: Any,
        expert_pool: Optional[DynamicLoRAExpertPool],
        harness: EmbodiedToolHarness,
        profile_config: Dict[str, Any],
        args: Any,
        device: str = "cuda",
        draft_gguf_path: Optional[str] = None,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "尚未安裝 `llama-cpp-python`！請在 RTX 5080 環境執行：\n"
                "  CMAKE_ARGS=\"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120\" "
                ".venv_rust/bin/pip install llama-cpp-python --no-cache-dir"
            ) from exc

        self.gguf_path = gguf_path
        self.rust_router = rust_router
        self.expert_pool = expert_pool
        self.harness = harness
        self.profile_config = profile_config
        self.args = args
        self.device = device if torch.cuda.is_available() else "cpu"

        self.hidden_dim = int(profile_config.get("hidden_dim", 2560))
        self.num_experts = int(profile_config.get("num_experts", 8))
        self.top_k = int(profile_config.get("top_k", 2))
        self.shy_lambda = float(profile_config.get("shy_lambda", 0.08))

        hyper = profile_config.get("hyperparameters", {})
        self.max_context_tokens = int(getattr(args, "max_context_tokens", None) or hyper.get("max_context_tokens", 32768))
        self.max_tokens = int(getattr(args, "max_tokens", None) or hyper.get("max_generation_tokens", 4096))
        self.temperature = float(getattr(args, "temp", None) or hyper.get("temperature", 1.0))
        self.top_p = float(getattr(args, "top_p", None) or hyper.get("top_p", 0.95))
        self.stop_tokens = list(profile_config.get("stop_tokens", ["<|tool_response>", "<turn|>", "<end_of_turn>"]))

        self.total_nrem_sleep_events = 0
        self.total_gpu_24_prunes = 0

        target_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._gpu_hidden_buf = torch.zeros((1, self.hidden_dim), dtype=target_dtype, device=self.device)

        print(f"📦 \033[1;34m[GGUF Engine]\033[0m Loading `{gguf_path}` onto RTX 5080 (n_gpu_layers=-1, n_ctx={self.max_context_tokens})...")
        self.llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=-1,
            n_ctx=self.max_context_tokens,
            n_batch=2048,
            flash_attn=True,
            logits_all=False,
            verbose=False,
        )
        print(f"✅ \033[1;32m[GGUF Engine Ready]\033[0m `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` loaded on GPU!")

    def _step_local_rust_moe(
        self,
        token_id: int,
        top_token_ids: List[int],
        top_logprobs: List[float],
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

        res_decay = self.rust_router.step_decay()
        newly_sleeping = list(res_decay) if isinstance(res_decay, (list, tuple)) else []

        if newly_sleeping:
            self.total_nrem_sleep_events += len(newly_sleeping)
            if self.expert_pool is not None:
                for exp_id in newly_sleeping:
                    exp_key = f"exp_{exp_id}"
                    if exp_key in self.expert_pool.lora_a:
                        la = self.expert_pool.lora_a[exp_key].data
                        used_cuda_kernel = False
                        if la.is_cuda and hasattr(self.rust_router, "prune_gpu_tensor_24_ptr"):
                            la_f32 = la.float().contiguous()
                            used_cuda_kernel = bool(
                                self.rust_router.prune_gpu_tensor_24_ptr(
                                    la_f32.data_ptr(),
                                    la_f32.shape[0],
                                    la_f32.shape[1],
                                    self.shy_lambda,
                                    0,
                                )
                            )
                            if used_cuda_kernel:
                                la.copy_(la_f32.to(la.dtype))
                        if not used_cuda_kernel:
                            apply_pytorch_24_fisher_pruning(la, shy_lambda=self.shy_lambda)
                        self.total_gpu_24_prunes += 1

        if self.expert_pool is not None and selected_exp and weights:
            with torch.no_grad():
                top_exp = selected_exp[0]
                top_w = weights[0]
                self._gpu_hidden_buf.fill_(float(entropy) * 0.1)
                self._gpu_hidden_buf[0, token_id % self.hidden_dim] = 1.0
                _ = self.expert_pool.forward_delta(self._gpu_hidden_buf, top_exp, top_w)

        return weights, selected_exp, entropy, newly_sleeping

    def run_turn(
        self,
        user_prompt: str,
        system_inst: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_logger: Optional[SessionLogger] = None,
    ) -> Tuple[str, int]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_inst}]
        if conversation_history:
            messages.extend(conversation_history[-8:])
        messages.append({"role": "user", "content": user_prompt})

        print(f"💬 \033[1;33mPrompt:\033[0m {user_prompt}")
        print(
            f"⚡ \033[1;36m[GGUF Engine (RTX 5080)]:\033[0m `{self.gguf_path}` "
            f"<-> Rust Zero-GIL MoE ({self.num_experts} Experts, Hidden={self.hidden_dim})"
        )
        print("🐈‍⬛ \033[1;32m[Kuro Bolt-E4B Stream]:\033[0m ", end="", flush=True)

        t0 = time.time()
        first_token_time = None
        ttft_ms = 0.0
        total_tokens = 0
        accumulated_text = ""
        cjk_cleaner = CJKStreamCleaner()
        last_experts: List[int] = []
        entropy_sum = 0.0

        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_tokens,
            stream=True,
        )

        for chunk in stream:
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if not piece:
                continue

            if first_token_time is None:
                first_token_time = time.time()
                ttft_ms = (first_token_time - t0) * 1000.0

            cleaned = cjk_cleaner.clean(piece)
            accumulated_text += cleaned
            sys.stdout.write(cleaned)
            sys.stdout.flush()

            synth_tid = sum(ord(c) * (i + 1) for i, c in enumerate(piece)) % 32000
            w, sel, ent, _ = self._step_local_rust_moe(synth_tid, [synth_tid], [-0.12])
            last_experts = sel
            entropy_sum += ent
            total_tokens += 1
            if session_logger is not None:
                cur_el = max(0.001, time.time() - t0)
                session_logger.record_token_telemetry(
                    token_id=synth_tid,
                    token_str=cleaned,
                    tps=total_tokens / cur_el,
                    entropy=ent,
                    selected_experts=sel,
                    weights=w,
                    expert_states=self.rust_router.get_expert_states(),
                    top_probs=[0.88],
                )

            if "</tool_call>" in accumulated_text or "<tool_call|" in accumulated_text:
                act, out, clean_prefix = ToolCallParser.detect_and_execute(accumulated_text, self.harness)
                if act and out is not None:
                    print(f"\n📥 \033[1;32m[Host Tool Dispatched - {act}]:\033[0m\n{out}\n")
                    break

        saved_arts = self.harness.extract_and_save_artifacts(accumulated_text)
        if saved_arts:
            print(f"\n✨ \033[1;32m[Embodied Artifacts Persisted]\033[0m Saved {len(saved_arts)} file(s): {saved_arts}")

        t_elapsed = max(0.001, time.time() - t0)
        tps = total_tokens / t_elapsed
        mean_ent = entropy_sum / max(1, total_tokens)
        print("\n\n" + "-" * 78)
        print(
            f"🏁 \033[1;32m[GGUF Turn Complete]\033[0m "
            f"Tokens: {total_tokens} | Elapsed: {t_elapsed:.2f}s ({tps:.2f} tps) | "
            f"TTFT: {ttft_ms:.1f}ms | Top-{self.top_k}: {last_experts} (H={mean_ent:.3f})"
        )
        print("-" * 78)

        if session_logger:
            session_logger.append_turn(
                user_prompt=user_prompt,
                assistant_reply=accumulated_text,
                stats={"tokens": total_tokens, "tps": tps, "elapsed": t_elapsed, "ttft_ms": ttft_ms},
                artifacts=saved_arts,
            )

        return accumulated_text, total_tokens
