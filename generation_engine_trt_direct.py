#!/usr/bin/env python3
"""
generation_engine_trt_direct.py - In-Process TensorRT-Edge-LLM Direct Engine for AGX Thor

Links TensorRT-Edge-LLM (`_edgellm_runtime` / `experimental.server.runtime.engine.LLM`)
and `rust_localsleep_moe.so` inside a SINGLE process and SINGLE CUDA context on AGX Thor:
  - No `ignite_31B_Assis.sh` / HTTP `llm-server` required.
  - Zero HTTP/SSE/JSON serialization overhead.
  - Reads raw `py::bytes` from C++ `StreamChannel` (`LogprobEntry.piece`) + incremental UTF-8
    decoder, 100% bypassing the `StreamChunk.text` UTF-8 split bug natively.
"""

import codecs
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from cjk_cleaner import CJKStreamCleaner
from generation_engine_trt_hybrid import TrtEdgeLlmHybridEngine
from session_logger import SessionLogger
from tool_parser import ToolCallParser
from tools_harness import EmbodiedToolHarness


def _ensure_edgellm_importable(edgellm_root: Optional[str] = None) -> None:
    candidates = []
    if edgellm_root:
        candidates.append(Path(edgellm_root))
    if os.environ.get("EDGELLM_ROOT"):
        candidates.append(Path(os.environ["EDGELLM_ROOT"]))
    candidates.extend([
        Path("/home/omegacd/TensorRT-Edge-LLM"),
        Path("/home/omegacd/projects/tensorRT/TensorRT-Edge-LLM"),
        Path("/home/omegacd/.openclaw/workspace/projects/tensorRT/TensorRT-Edge-LLM"),
        Path("/home/node/.openclaw/workspace/projects/tensorRT/TensorRT-Edge-LLM"),
        Path(__file__).resolve().parent.parent / "tensorRT" / "TensorRT-Edge-LLM",
    ])
    for cand in candidates:
        if cand.is_dir() and (cand / "experimental" / "server" / "runtime" / "engine.py").is_file():
            cand_str = str(cand.resolve())
            if cand_str not in sys.path:
                sys.path.insert(0, cand_str)
            break


class TrtEdgeLlmDirectEngine(TrtEdgeLlmHybridEngine):
    """
    In-Process Direct Engine running TensorRT-Edge-LLM C++ runtime (`_edgellm_runtime`)
    and `rust_localsleep_moe` in the same process on AGX Thor.
    """

    def __init__(
        self,
        model_path: str,
        rust_router: Any,
        expert_pool: Any,
        harness: EmbodiedToolHarness,
        profile_config: Dict[str, Any],
        args: Any,
        device: str = "cuda",
        edgellm_root: Optional[str] = None,
        cache_dir: str = "",
    ):
        super().__init__(
            server_url="in-process://_edgellm_runtime",
            model_id=os.path.basename(model_path.rstrip("/")) or "gemma-4-31B-it",
            rust_router=rust_router,
            expert_pool=expert_pool,
            harness=harness,
            profile_config=profile_config,
            args=args,
            device=device,
        )
        self.model_path = model_path
        _ensure_edgellm_importable(edgellm_root or getattr(args, "edgellm_root", None))

        from experimental.server.runtime.engine import LLM, SamplingParams
        self._SamplingParams = SamplingParams

        spec_cfg = profile_config.get("speculative_config", {
            "method": getattr(args, "spec_method", None) or "mtp",
            "num_speculative_tokens": int(getattr(args, "num_spec_tokens", None) or 5),
        })
        if getattr(args, "spec_method", None) == "none":
            spec_cfg = None

        max_input_len = int(profile_config.get("max_input_len", 8192))
        max_kv_cap = int(profile_config.get("max_kv_cache_capacity", self.max_context_tokens))
        resolved_cache_dir = cache_dir or getattr(args, "cache_dir", None) or profile_config.get("cache_dir", "")

        print(
            f"🚀 \033[1;36m[TRT-Direct In-Process Init]\033[0m Loading TensorRT-Edge-LLM directly into process:\n"
            f"   • Model: {self.model_path}\n"
            f"   • SpecDecode: {spec_cfg}\n"
            f"   • MaxInput: {max_input_len} | MaxKVCache: {max_kv_cap}"
        )
        self.llm = LLM(
            model=self.model_path,
            cache_dir=resolved_cache_dir,
            max_input_len=max_input_len,
            max_batch_size=1,
            max_kv_cache_capacity=max_kv_cap,
            speculative_config=spec_cfg,
            enable_in_flight_batching=False,
        )

    @staticmethod
    def _decode_chunk_lossless(chunk: Any, utf8_dec: codecs.IncrementalDecoder) -> str:
        """
        Decodes text from C++ StreamChunk losslessly across MTP steps:
          1. Tries `chunk.text` first; if prior chunk left trailing UTF-8 bytes in `utf8_dec`,
             feeds the raw bytes into `utf8_dec`.
          2. If `chunk.text` raises `UnicodeDecodeError` (split 3-byte CJK char), extracts the
             exact C++ raw `bytes` from `ude.object` and feeds `utf8_dec` without dropping a single byte.
        """
        try:
            raw_str = chunk.text
            if utf8_dec.getstate()[0]:
                return utf8_dec.decode(raw_str.encode("utf-8", errors="surrogateescape"), final=bool(chunk.finished))
            return raw_str
        except UnicodeDecodeError as ude:
            raw_b = ude.object if isinstance(ude.object, bytes) else b""
            return utf8_dec.decode(raw_b, final=bool(chunk.finished))

    def _stream_single_completion(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, int, float, Optional[Tuple[str, str, str]], Dict[str, Any]]:
        """
        Streams tokens directly from in-process C++ `_edgellm_runtime.StreamChannel`
        and drives `rust_localsleep_moe` in the same address space.
        """
        logit_bias_int: Optional[Dict[int, float]] = None
        if self.dynamic_logit_bias:
            logit_bias_int = {int(k): float(v) for k, v in list(self.dynamic_logit_bias.items())[:64]}

        params = self._SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_tokens,
            num_logprobs=4,
            enable_thinking=False,
            logit_bias=logit_bias_int,
        )

        self.llm._ensure_open()
        channel = self.llm._rt.StreamChannel.create()
        channel.set_skip_special_tokens(params.skip_special_tokens)

        request = self.llm._make_generation_request(
            messages,
            params,
            stream_channel=channel,
        )

        error_holder: List[Optional[BaseException]] = [None]

        def _worker():
            try:
                self.llm._handle_request(request)
            except BaseException as exc:
                error_holder[0] = exc
                channel.cancel()

        worker = threading.Thread(target=_worker, daemon=True)
        t0 = time.time()
        worker.start()

        utf8_dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        cjk_cleaner = CJKStreamCleaner()
        accumulated_text = ""
        chunk_steps = 0
        completion_tokens = 0
        ttft_ms = 0.0
        first_token_time = None
        intercepted_tool: Optional[Tuple[str, str, str]] = None
        last_weights: List[float] = []
        last_experts: List[int] = []
        entropy_sum = 0.0
        nrem_triggered_in_pass: List[int] = []

        try:
            while True:
                chunk = channel.wait_pop(timeout_ms=200)
                if chunk is None:
                    if channel.is_finished() or channel.is_cancelled():
                        chunk = channel.try_pop()
                        if chunk is None:
                            break
                    continue

                piece_text = self._decode_chunk_lossless(chunk, utf8_dec)
                token_ids = list(chunk.token_ids)
                completion_tokens += len(token_ids)

                if piece_text:
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft_ms = (first_token_time - t0) * 1000.0
                    accumulated_text += piece_text
                    clean_delta = cjk_cleaner.push(piece_text)
                    if clean_delta:
                        sys.stdout.write(clean_delta)
                        sys.stdout.flush()

                # Feed in-process C++ logprobs / token_ids directly to Rust Zero-GIL MoE router
                raw_lps = getattr(chunk, "logprobs", None) or []
                if raw_lps:
                    for pos_idx, pos_candidates in enumerate(raw_lps):
                        cands = list(pos_candidates) if not isinstance(pos_candidates, (list, tuple)) else pos_candidates
                        if not cands:
                            continue
                        chosen_tid = token_ids[pos_idx] if pos_idx < len(token_ids) else int(cands[0].token_id)
                        top_tids = [int(c.token_id) for c in cands]
                        top_vals = [float(c.logprob) for c in cands]
                        w, sel, ent, slp = self._step_local_rust_moe(chosen_tid, top_tids, top_vals)
                        last_weights, last_experts = w, sel
                        entropy_sum += ent
                        chunk_steps += 1
                        if slp:
                            nrem_triggered_in_pass.extend(slp)
                else:
                    for tid in token_ids:
                        w, sel, ent, slp = self._step_local_rust_moe(int(tid), [int(tid)], [-0.1])
                        last_weights, last_experts = w, sel
                        entropy_sum += ent
                        chunk_steps += 1
                        if slp:
                            nrem_triggered_in_pass.extend(slp)

                if "<tool_call>" in accumulated_text or "<call:fs:" in accumulated_text:
                    act, out, clean_prefix = ToolCallParser.detect_and_execute(accumulated_text, self.harness)
                    if act and out is not None:
                        intercepted_tool = (act, str(out), clean_prefix)
                        channel.cancel()
                        break

                if chunk.finished:
                    break
        finally:
            if not (channel.is_finished() or channel.is_cancelled()):
                channel.cancel()
            worker.join(timeout=5.0)

        if error_holder[0] is not None and intercepted_tool is None:
            raise error_holder[0]

        tail = cjk_cleaner.flush()
        if tail and intercepted_tool is None:
            sys.stdout.write(tail)
            sys.stdout.flush()

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
        }
        return accumulated_text, completion_tokens, ttft_ms, intercepted_tool, routing_meta
