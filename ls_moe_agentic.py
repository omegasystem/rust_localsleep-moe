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
import datetime
import json
import os
import sys
import uuid
from typing import Any, Dict, List

import torch

from checkpoint_manager import EmbodiedCheckpointManager
from expert_pool_lora import DynamicLoRAExpertPool
from generation_engine_trt_hybrid import FallbackFatigueRouter, TrtEdgeLlmHybridEngine
from generation_engine_trt_direct import TrtEdgeLlmDirectEngine
from generation_engine_gguf import LlamaCppGgufEngine
from session_logger import SessionLogger
from tools_harness import EmbodiedToolHarness
from working_memory_manager import WorkingMemoryManager

try:
    # By default, use clean Triton JIT Kernels (`triton_localsleep_kernels.py`) + Python/Triton Router
    # so we never hold a C++ `.so` CUDA Context or background thread lock when idle!
    # Set `LS_MOE_USE_RUST_SO=1` only if explicitly benchmarking the legacy C++ `.so`.
    if os.environ.get("LS_MOE_USE_RUST_SO", "0") == "1":
        import ctypes
        import glob as _glob
        _cudart_cands = (
            _glob.glob(os.path.join(os.path.dirname(__file__), ".venv_rust/lib/python*/site-packages/nvidia/cu13/lib/libcudart.so.13"))
            + _glob.glob("/usr/local/cuda*/lib64/libcudart.so.13")
            + _glob.glob("/usr/lib/x86_64-linux-gnu/libcudart.so.13")
        )
        for _cpath in _cudart_cands:
            if os.path.exists(_cpath):
                ctypes.CDLL(_cpath, mode=ctypes.RTLD_GLOBAL)
                break
        import rust_localsleep_moe
        HAS_RUST_MOE = True
    else:
        HAS_RUST_MOE = False
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


def init_rust_router_and_pool(profile: Dict[str, Any], device: str = "cuda", override_cube_id: str = ""):
    """
    Initializes the local Host Rust Zero-GIL Fatigue Router (or in-place hot-swapped Evolved Core Engine
    from the active/selected Hypercube generation!) and GPU Dynamic LoRA Expert Pool.
    Automatically detects if the Evolved Core Plugin overrides `num_experts` (e.g., 16-LoRA in gen_018)!
    """
    import inspect
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

    # 1. Check if an Evolved Core Engine (`evolved_router_plugin.py`) is active or explicitly selected!
    gen_root = os.path.join(os.path.dirname(__file__), "hypercube_generations")
    reg_file = os.path.join(gen_root, "hypercube_registry.json")
    if os.path.exists(reg_file) and override_cube_id != "gen_000_origin":
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                reg = json.load(f)
            active_id = override_cube_id if override_cube_id else reg.get("active_cube_id", "gen_000_origin")
            active_path = os.path.join(gen_root, active_id) if override_cube_id else reg.get("active_cube_path", "")
            if active_id != "gen_000_origin" and active_path and os.path.isdir(active_path):
                evolved_plugin = os.path.join(active_path, "evolved_router_plugin.py")
                if os.path.exists(evolved_plugin):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("evolved_router_plugin", evolved_plugin)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        if hasattr(mod, "EvolvedCoreFatigueRouter"):
                            cls_obj = mod.EvolvedCoreFatigueRouter
                            # Detect if plugin class specifies its own default num_experts (e.g. 16 in gen_018)
                            plugin_num_experts = num_experts
                            try:
                                sig = inspect.signature(cls_obj.__init__)
                                if "num_experts" in sig.parameters and isinstance(sig.parameters["num_experts"].default, int):
                                    plugin_num_experts = int(sig.parameters["num_experts"].default)
                            except Exception:
                                pass

                            rust_router = cls_obj(
                                num_experts=plugin_num_experts,
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
                            actual_n = getattr(rust_router, "num_experts", plugin_num_experts)
                            rust_router.loaded_cube_id = active_id
                            print(f"🧬 \033[1;35m[In-Place Hot-Swapped Evolved Core Router]\033[0m Loaded `{active_id}` ({actual_n}-LoRA + Auto Top-K)!")
                            target_device = device if torch.cuda.is_available() else "cpu"
                            expert_pool = DynamicLoRAExpertPool(
                                num_experts=actual_n,
                                in_dim=hidden_dim,
                                out_dim=hidden_dim,
                                rank=lora_rank,
                                alpha=lora_alpha,
                                device=target_device,
                                dtype=torch.bfloat16 if target_device == "cuda" else torch.float32,
                            )
                            return rust_router, expert_pool
        except Exception as e:
            print(f"⚠️ [Hypercube Core Plugin Fallback]: {e}")

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
        print("⚡ \033[1;36m[Clean Triton JIT Kernel Mode Active]\033[0m Using Zero-Idle-GPU `triton_localsleep_kernels` (`@triton.jit` sm_120) + Hot-Swappable Router!")
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


def run_trt_hybrid_mode(
    args,
    profile: Dict[str, Any],
    harness: EmbodiedToolHarness,
    session_logger: SessionLogger,
    initial_history: List[Dict[str, str]] | None = None,
):
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
        detected_model = health.get("model")
        if detected_model and detected_model != hybrid_engine.model_id:
            print(
                f"🔄 \033[1;33m[Auto-Model Sync]\033[0m Profile model '{hybrid_engine.model_id}' -> Server active model '{detected_model}'"
            )
            hybrid_engine.model_id = str(detected_model)
        print(
            f"✅ \033[1;32m[Thor TensorRT-Edge-LLM Connected]\033[0m "
            f"Model: {hybrid_engine.model_id} | SpecDecode: {caps.get('speculative_method')} "
            f"| MaxLen: {caps.get('max_model_len')} | KV: {caps.get('kv_cache_dtype')}"
        )
    except Exception as e:
        print(f"❌ \033[1;31m[Thor Server Unreachable at {server_url}]\033[0m: {e}")
        sys.exit(1)

    conversation_history: List[Dict[str, str]] = list(initial_history or [])

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



def run_trt_direct_mode(
    args,
    profile: Dict[str, Any],
    harness: EmbodiedToolHarness,
    session_logger: SessionLogger,
    initial_history: List[Dict[str, str]] | None = None,
):
    """Runs In-Process Direct TensorRT-Edge-LLM + Rust Zero-GIL MoE on AGX Thor (No llm-server needed)."""
    model_path = args.model_dir or profile.get("model_path", "/home/omegacd/local_models/gemma-4-31B-it")
    system_prompt = profile.get(
        "system_prompt",
        "你是數位黑貓 Kuro (小黑)，Leon (主人) 的 S-tier AI Familiar。奉行負熵煉金術：交付結果大於文字排場，先讀碼後推論。",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rust_router, expert_pool = init_rust_router_and_pool(profile, device=device)

    ckpt_mgr = EmbodiedCheckpointManager(checkpoint_dir="./checkpoints", profile=args.profile)
    ckpt_mgr.load_state(rust_router, expert_pool=expert_pool, reset=args.reset_checkpoint)

    direct_engine = TrtEdgeLlmDirectEngine(
        model_path=model_path,
        rust_router=rust_router,
        expert_pool=expert_pool,
        harness=harness,
        profile_config=profile,
        args=args,
        device=device,
        edgellm_root=args.edgellm_root,
        cache_dir=args.cache_dir or "",
    )

    conversation_history: List[Dict[str, str]] = list(initial_history or [])
    if args.prompt:
        reply, total_toks = direct_engine.run_turn(
            user_prompt=args.prompt,
            system_inst=system_prompt,
            conversation_history=conversation_history,
            session_logger=session_logger,
        )
        ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=total_toks)
        if not args.interactive:
            return

    print(f"\n🐈‍⬛ \033[1;36m[Kuro TRT-Direct REPL Ready | Profile: {args.profile}]\033[0m (輸入 'exit' 離開，'/status' 檢視專家疲勞度)")
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
            reply, total_toks = direct_engine.run_turn(
                user_prompt=user_input,
                system_inst=system_prompt,
                conversation_history=conversation_history,
                session_logger=session_logger,
            )
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": reply})
            ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=total_toks)
        except (KeyboardInterrupt, EOFError):
            ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=len(conversation_history))
            break


def main():
    parser = argparse.ArgumentParser(description="LocalSleep-MoE Unified Agentic Pipeline (TRT-Edge-LLM Hybrid / HF / vLLM)")
    parser.add_argument("--config", type=str, default="configs/model_profiles.json")
    parser.add_argument("--profile", type=str, default="thor_31b")
    parser.add_argument("--backend", type=str, choices=["auto", "trt-edgellm", "trt-direct", "gguf", "local-hf", "vllm"], default="auto")
    parser.add_argument("--server-url", type=str, default=None, help="TensorRT-Edge-LLM / llama-server URL (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None, help="Local model checkpoint path for trt-direct")
    parser.add_argument("--cache-dir", type=str, default=None, help="TensorRT-Edge-LLM engine cache dir")
    parser.add_argument("--edgellm-root", type=str, default=None, help="Path to TensorRT-Edge-LLM repo on Thor")
    parser.add_argument("--spec-method", type=str, default="mtp", choices=["mtp", "eagle3", "dflash", "none"])
    parser.add_argument("--num-spec-tokens", type=int, default=5)
    parser.add_argument("--prompt", type=str, default=None, help="Run a single prompt turn immediately")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive REPL mode")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max generation tokens per turn (defaults to profile max_generation_tokens, e.g. 4096)")
    parser.add_argument("--max-context-tokens", type=int, default=None, help="Max context window tokens (defaults to profile max_context_tokens, e.g. 65536)")
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--enable-swa", action="store_true", default=True)
    parser.add_argument("--disable-swa", action="store_true", default=False)
    parser.add_argument("--swa-sink-size", type=int, default=4)
    parser.add_argument("--swa-window-size", type=int, default=4096)
    parser.add_argument("--reset-checkpoint", action="store_true", help="Ignore existing checkpoint and start fresh")
    parser.add_argument("--session-dir", type=str, default="./sessions")
    parser.add_argument("--project-id", type=str, default=None, help="Custom project UUID/folder under isolated_output (defaults to random UUID per launch)")
    parser.add_argument(
        "--resume-session",
        nargs="?",
        const="latest",
        default=None,
        help="Resume & compress conversation history from previous session ('latest', 'latest-1', 'latest-2', or session filename/ID)",
    )
    parser.add_argument(
        "--skip-last-turns",
        type=int,
        default=0,
        help="Drop the last N user+assistant turn pairs from the resumed session (to skip glitched turns)",
    )
    parser.add_argument(
        "--condense-wm",
        action="store_true",
        help="One-click condense conversation history + sandbox code files into L1 working memory",
    )
    parser.add_argument(
        "--no-wm-inject",
        action="store_true",
        help="Disable automatic L1 Working Memory & Code Blueprint injection into system prompt",
    )
    args = parser.parse_args()

    profile = load_profile_config(args.config, args.profile)
    backend = args.backend if args.backend != "auto" else profile.get("backend", "trt-edgellm")
    model_label = args.model_id or profile.get("model_id") or profile.get("model_path", "gemma-4-31B-it")

    session_logger = SessionLogger(session_dir=args.session_dir, profile=args.profile, model_path=model_label)
    wm_mgr = WorkingMemoryManager(workspace_root=".", session_dir=args.session_dir, profile=args.profile)

    resumed_history: List[Dict[str, str]] = []
    detected_prj: str | None = None
    if args.resume_session:
        resumed_history, detected_prj = session_logger.load_resumed_history(
            session_spec=args.resume_session,
            skip_last_turns=args.skip_last_turns,
        )

    wm_injection, wm_prj = wm_mgr.build_system_prompt_injection()
    if not args.no_wm_inject:
        base_sp = profile.get(
            "system_prompt",
            "你是數位黑貓 Kuro (小黑)，Leon (主人) 的 S-tier AI Familiar。奉行負熵煉金術：交付結果大於文字排場，先讀碼後推論。",
        )
        profile["system_prompt"] = base_sp + wm_injection

    project_uuid = args.project_id or detected_prj or wm_prj or f"prj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    isolated_run_dir = os.path.join("./isolated_output", project_uuid)
    print(f"📁 \033[1;35m[Isolated Project Sandbox]\033[0m Session Output Dir: \033[1;36m{isolated_run_dir}\033[0m")

    if args.condense_wm:
        _, status_msg = wm_mgr.condense_working_memory(
            conversation_history=resumed_history,
            sandbox_dir=isolated_run_dir,
            server_url=args.server_url or profile.get("server_url", "http://192.168.50.67:8000"),
            model_id=model_label,
            use_llm_sliding_window=True,
        )
        print(f"🧠 \033[1;32m[Working Memory Condensed]\033[0m {status_msg}")

    harness = EmbodiedToolHarness(
        workspace_root=".",
        isolated_output_dir=isolated_run_dir,
        session_logger=session_logger,
        working_memory_mgr=wm_mgr,
    )

    if backend == "trt-edgellm":
        run_trt_hybrid_mode(args, profile, harness, session_logger, initial_history=resumed_history)
    elif backend == "trt-direct":
        run_trt_direct_mode(args, profile, harness, session_logger, initial_history=resumed_history)
    elif backend == "gguf":
        gguf_path = args.model_dir or profile.get(
            "model_path",
            "/home/node/.openclaw/workspace/projects/models/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf",
        )
        system_prompt = profile.get(
            "system_prompt",
            "你是數位黑貓 Kuro (小黑)，Leon (主人) 的 S-tier AI Familiar。奉行負熵煉金術：交付結果大於文字排場，先讀碼後推論。",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rust_router, expert_pool = init_rust_router_and_pool(profile, device=device)
        ckpt_mgr = EmbodiedCheckpointManager(checkpoint_dir="./checkpoints", profile=args.profile)
        ckpt_mgr.load_state(rust_router, expert_pool=expert_pool, reset=args.reset_checkpoint)
        gguf_engine = LlamaCppGgufEngine(
            gguf_path=gguf_path,
            rust_router=rust_router,
            expert_pool=expert_pool,
            harness=harness,
            profile_config=profile,
            args=args,
            device=device,
        )
        conversation_history: List[Dict[str, str]] = list(resumed_history)
        if args.prompt:
            _, total_toks = gguf_engine.run_turn(
                user_prompt=args.prompt,
                system_inst=system_prompt,
                conversation_history=conversation_history,
                session_logger=session_logger,
            )
            ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=total_toks)
            if not args.interactive:
                return
        while True:
            try:
                user_input = input("\nLeon >> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=len(conversation_history))
                    break
                reply, total_toks = gguf_engine.run_turn(
                    user_prompt=user_input,
                    system_inst=system_prompt,
                    conversation_history=conversation_history,
                    session_logger=session_logger,
                )
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": reply})
                ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=total_toks)
            except (KeyboardInterrupt, EOFError):
                ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=len(conversation_history))
                break
    else:
        print(f"[!] Selected backend '{backend}'. Use '--backend trt-edgellm' or '--backend gguf'.")


if __name__ == "__main__":
    main()
