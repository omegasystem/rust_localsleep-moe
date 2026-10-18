#!/usr/bin/env python3
"""
ls_moe_agentic.py - Unified Agentic Embodied Entrypoint for rust_localsleep-moe

Supports Three Execution Backends:
  1. `trt-edgellm` (Default Hybrid Mode):
     - Remote Backbone: AGX Thor (`http://192.168.50.67:8000`) running TensorRT-Edge-LLM (`gemma-4-31B-it` + MTP speculative decoding)
     - Local Host (`x86_64 + RTX 5080`): Rust Zero-GIL Fatigue Router (<8us) + DualStreamEngine NREM 2:4 Pruning + Dynamic LoRA Expert Pool + SWA + Embodied Tool Harness (`<tool_call>` & `<file path="...">` sandbox)
  2. `local-hf`:
     - Local HuggingFace Transformers model + Rust Zero-GIL Router + Dynamic LoRA Pool + SWA (`GenerationLoop`)
  3. `vllm`:
     - Local/Async vLLM Engine (`VllmGenerationLoop`)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch

from checkpoint_manager import EmbodiedCheckpointManager
from expert_pool_lora import DynamicLoRAExpertPool
from generation_engine_trt_hybrid import FallbackFatigueRouter, TrtEdgeLlmHybridEngine
from session_logger import SessionLogger
from tools_harness import EmbodiedToolHarness

try:
    import rust_localsleep_moe
    HAS_RUST_MOE = True
except ImportError:
    HAS_RUST_MOE = False


def load_profile_config(config_path: str, profile_name: str) -> Dict[str, Any]:
    candidates = [
        config_path,
        os.path.join(os.path.dirname(__file__), config_path),
        os.path.join(os.path.dirname(__file__), "configs/model_profiles.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profiles = data.get("profiles", data)
            if profile_name in profiles:
                return profiles[profile_name]
    return {}


def init_rust_router_and_pool(profile: Dict[str, Any], device: str = "cuda"):
    """Initializes the local Host Rust Zero-GIL Fatigue Router (or PyTorch Fallback before Host compile) and GPU Dynamic LoRA Expert Pool."""
    hidden_dim = int(profile.get("hidden_dim", 5376))
    num_experts = int(profile.get("num_experts", 8))
    top_k = int(profile.get("top_k", 2))
    lora_rank = int(profile.get("lora_rank", 32))
    lora_alpha = float(profile.get("lora_alpha", 64.0))
    fatigue_threshold = float(profile.get("fatigue_threshold", 1.6))
    wake_threshold = float(profile.get("wake_threshold", 0.35))
    decay_rate = float(profile.get("decay_rate", 0.015))
    sleep_duration = int(profile.get("sleep_duration", 25))
    shy_lambda = float(profile.get("shy_lambda", 0.08))
    enable_2_4 = bool(profile.get("enable_2_4_sparsity", True))

    if HAS_RUST_MOE:
        try:
            rust_router = rust_localsleep_moe.PyFatigueRouter(
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                top_k=top_k,
                fatigue_threshold=fatigue_threshold,
                temperature=1.0,
                wake_threshold=wake_threshold,
                decay_rate=decay_rate,
                sleep_duration=sleep_duration,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                shy_lambda=shy_lambda,
                enable_2_4_sparsity=enable_2_4,
            )
        except TypeError:
            rust_router = rust_localsleep_moe.PyFatigueRouter(
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                top_k=top_k,
                fatigue_threshold=fatigue_threshold,
                temperature=1.0,
            )
        print("🦀 \033[1;32m[Native Rust Zero-GIL Router Loaded]\033[0m (`rust_localsleep_moe.PyFatigueRouter`)")
    else:
        print("⚡ \033[1;33m[Host .so Pending Compile]\033[0m Using thread-safe `FallbackFatigueRouter` (Once container restarts with .cargo mounted, build with `maturin develop --release --features python,cuda`).")
        rust_router = FallbackFatigueRouter(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            top_k=top_k,
            fatigue_threshold=fatigue_threshold,
            temperature=1.0,
            wake_threshold=wake_threshold,
            decay_rate=decay_rate,
            sleep_duration=sleep_duration,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            shy_lambda=shy_lambda,
            enable_2_4_sparsity=enable_2_4,
        )

    target_device = device if torch.cuda.is_available() else "cpu"
    expert_pool = DynamicLoRAExpertPool(
        num_experts=num_experts,
        in_dim=hidden_dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        device=target_device,
        dtype=torch.bfloat16 if target_device == "cuda" else torch.float32,
    )
    return rust_router, expert_pool


def run_trt_hybrid_mode(args, profile: Dict[str, Any], harness: EmbodiedToolHarness, session_logger: SessionLogger):
    """Runs the Hybrid Inference Mode (Thor TensorRT-Edge-LLM Server + Local Host Rust/CUDA MoE)."""
    server_url = args.server_url or profile.get("server_url", "http://192.168.50.67:8000")
    model_id = args.model_id or profile.get("model_id", "gemma-4-31B-it")
    system_prompt = profile.get(
        "system_prompt",
        "你是數位黑貓 Kuro (小黑)，Leon (主人) 的 S-tier AI Familiar。奉行負熵煉金術：交付結果大於文字排場，先讀碼後推論。",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rust_router, expert_pool = init_rust_router_and_pool(profile, device=device)

    ckpt_mgr = EmbodiedCheckpointManager(checkpoint_dir="./checkpoints", profile=args.profile)
    ckpt_mgr.load_state(rust_router, expert_pool=expert_pool, reset=args.reset_checkpoint)

    hybrid_engine = TrtEdgeLlmHybridEngine(
        server_url=server_url,
        model_id=model_id,
        rust_router=rust_router,
        expert_pool=expert_pool,
        harness=harness,
        profile_config=profile,
        args=args,
        device=device,
    )

    try:
        health = hybrid_engine.check_server_health()
        caps = health.get("capabilities", {})
        print(
            f"✅ \033[1;32m[Thor TensorRT-Edge-LLM Connected]\033[0m "
            f"Model: {health.get('model')} | SpecDecode: {caps.get('speculative_method')} "
            f"| MaxLen: {caps.get('max_model_len')} | KV: {caps.get('kv_cache_dtype')}"
        )
    except Exception as e:
        print(f"❌ \033[1;31m[Thor Server Unreachable at {server_url}]\033[0m: {e}")
        sys.exit(1)

    conversation_history: List[Dict[str, str]] = []

    if args.prompt:
        reply, total_toks = hybrid_engine.run_turn(
            user_prompt=args.prompt,
            system_inst=system_prompt,
            conversation_history=conversation_history,
            session_logger=session_logger,
        )
        ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=total_toks)
        if not args.interactive:
            return

    print(f"\n🐈‍⬛ \033[1;36m[Kuro Hybrid REPL Ready | Profile: {args.profile}]\033[0m (輸入 'exit' 離開，'/status' 檢視專家疲勞度)")
    while True:
        try:
            user_input = input("\nLeon >> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=len(conversation_history))
                break
            if user_input.lower() == "/status":
                for item in rust_router.get_expert_states():
                    eid, st, fat, tok, cd = item[:5]
                    sp = item[5] if len(item) >= 7 else 0.0
                    cyc = item[6] if len(item) >= 7 else 0
                    st_name = {0: "🟢Active", 1: "🟡Fatigued", 2: "💤NremSleep"}.get(st, "Active")
                    print(
                        f"  Expert #{eid:02d} | {st_name:<12} | Fatigue: {fat:.3f} | "
                        f"Tokens: {tok:5d} | Sparsity(2:4): {sp:5.1f}% | SleepCycles: {cyc} | CD: {cd}"
                    )
                continue

            reply, total_toks = hybrid_engine.run_turn(
                user_prompt=user_input,
                system_inst=system_prompt,
                conversation_history=conversation_history,
                session_logger=session_logger,
            )
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": reply})
            ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=total_toks)

        except (KeyboardInterrupt, EOFError):
            print("\n[!] Exiting Hybrid REPL and saving synaptic checkpoint...")
            ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=len(conversation_history))
            break


def main():
    parser = argparse.ArgumentParser(description="LocalSleep-MoE Unified Agentic Pipeline (TRT-Edge-LLM Hybrid / HF / vLLM)")
    parser.add_argument("--config", type=str, default="configs/model_profiles.json")
    parser.add_argument("--profile", type=str, default="thor_31b")
    parser.add_argument("--backend", type=str, choices=["auto", "trt-edgellm", "local-hf", "vllm"], default="auto")
    parser.add_argument("--server-url", type=str, default=None, help="TensorRT-Edge-LLM Server URL (e.g. http://192.168.50.67:8000)")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Run a single prompt turn immediately")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive REPL mode")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--enable-swa", action="store_true", default=True)
    parser.add_argument("--disable-swa", action="store_true", default=False)
    parser.add_argument("--swa-sink-size", type=int, default=4)
    parser.add_argument("--swa-window-size", type=int, default=4096)
    parser.add_argument("--reset-checkpoint", action="store_true", help="Ignore existing checkpoint and start fresh")
    parser.add_argument("--session-dir", type=str, default="./sessions")
    args = parser.parse_args()

    profile = load_profile_config(args.config, args.profile)
    backend = args.backend if args.backend != "auto" else profile.get("backend", "trt-edgellm")
    model_label = args.model_id or profile.get("model_id") or profile.get("model_path", "gemma-4-31B-it")

    session_logger = SessionLogger(session_dir=args.session_dir, profile=args.profile, model_path=model_label)
    harness = EmbodiedToolHarness(workspace_root=".", isolated_output_dir="./isolated_output", session_logger=session_logger)

    if backend == "trt-edgellm":
        run_trt_hybrid_mode(args, profile, harness, session_logger)
    else:
        print(f"[!] Selected backend '{backend}'. Use '--backend trt-edgellm' for Thor TensorRT-Edge-LLM Hybrid Mode.")


if __name__ == "__main__":
    main()
