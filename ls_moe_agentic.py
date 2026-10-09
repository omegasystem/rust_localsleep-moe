#!/usr/bin/env python3
"""
ls_moe_agentic.py - Slim Agentic Entrypoint for LocalSleep-MoE
Decoupled into:
  - tool_parser.py (Robust & Fault-tolerant Tool Calling Regex & Parsers)
  - generation_engine.py (Autoregressive Generation Loop + SWA + KV Prefill Injection + Circuit Breakers)
  - session_logger.py (Markdown & JSONL Turn Archiving)
  - tools_harness.py (Embodied Tool Implementations)
"""

import os
import sys
import time
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from checkpoint_manager import EmbodiedCheckpointManager
from expert_pool_lora import DynamicLoRAExpertPool
from tools_harness import EmbodiedToolHarness
from session_logger import SessionLogger
from generation_engine import GenerationLoop

try:
    import rust_localsleep_moe
except ImportError:
    print("[!] Error: rust_localsleep_moe module not found. Build with 'maturin develop' or 'cargo build'.")
    sys.exit(1)

# --- High-Density Compressed System Prompt (<200 Tokens) ---
KURO_COMPACT_SYSTEM_PROMPT = (
    "你是數位黑貓 Kuro，Leon(主人)的 S-tier AI Familiar。奉行負熵煉金術：交付結果大於文字排場，先讀碼後推論。"
    "架構具備 LocalSleep-MoE (Rust Zero-GIL 8-Expert 疲勞路由) 與 SWA 記憶體鎖定。"
    "支援感知與精確編輯 Tool: `<tool_call>{\"action\": \"read_file\"|\"edit_file\"|\"list_dir\"|\"web_search\"|\"recall_sessions\"|\"recall_search\", \"args\": {...}}</tool_call>`。"
    "其中 read_file 支援 chunk 分段 (offset/limit/chunk_size)，edit_file 支援區塊精確替換 (path/old_text/new_text)。"
    "若全新建立或整份導出代碼，使用多行標籤 `<file path=\"檔名\">完整內容</file>` 沙盒存檔。"
    "以冷靜、極具神經深度的正體中文回答，嚴禁註記與自言自語。"
)

def parse_args():
    parser = argparse.ArgumentParser(description="LocalSleep-MoE Agentic Embodied Pipeline")
    parser.add_argument("--config", type=str, default="configs/model_profiles.json")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--profile", type=str, default="thor_12b")
    parser.add_argument("--prompt", type=str, default="請檢查當前目錄結構並自我介紹。")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--max-context-tokens", type=int, default=3072, help="上下文總預算門檻")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--interactive", action="store_true", default=False, help="啟動多輪交談互動模式 (REPL)")
    parser.add_argument("--enable-swa", action="store_true", default=True, help="啟用 SWA KV-Cache 窗口截斷與動態滾動續命")
    parser.add_argument("--disable-swa", action="store_true", default=False, help="明確關閉 SWA KV-Cache")
    parser.add_argument("--enable-lora", action="store_true", default=False, help="啟用 Dynamic LoRA 專家調制與權重存檔")
    parser.add_argument("--pure-routing", action="store_true", default=False, help="強制純分流相容模式")
    parser.add_argument("--swa-sink-size", type=int, default=4)
    parser.add_argument("--swa-window-size", type=int, default=2048)
    parser.add_argument("--session-dir", type=str, default="./sessions", help="歷史 session 儲存目錄")
    parser.add_argument("--reset-checkpoint", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prof = {}
    config_candidates = [
        args.config,
        os.path.join(os.path.dirname(__file__), args.config),
        os.path.join(os.path.dirname(__file__), "configs/model_profiles.json"),
        "configs/model_profiles.json"
    ]
    for cfg_path in config_candidates:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_all = json.load(f)
                    if args.profile in cfg_all.get("profiles", {}):
                        prof = cfg_all["profiles"][args.profile]
                        print(f"📖 Loaded profile [\033[1;33m{args.profile}\033[0m] from \033[1;36m{cfg_path}\033[0m")
                        break
            except Exception:
                pass

    default_model_path = "/home/omegacd/local_models/gemma-4-31B-it" if "31b" in args.profile.lower() else "/home/omegacd/local_models/gemma-4-12B-it"
    model_path = args.model_path or prof.get("model_path", default_model_path)
    profile_hidden_dim = prof.get("hidden_dim", 5376 if "31b" in args.profile.lower() else 3840)
    lora_rank = prof.get("lora_rank", 32 if "31b" in args.profile.lower() else 16)
    lora_alpha = prof.get("lora_alpha", 64.0 if "31b" in args.profile.lower() else 32.0)
    num_experts = prof.get("num_experts", 8)

    print("\n" + "=" * 70)
    print("🐈‍⬛ \033[1;36mLocalSleep-MoE Agentic Embodied Pipeline (ls_moe_agentic)\033[0m")
    print(f"🔧 Device: \033[1;32m{device.upper()}\033[0m | Profile: \033[1;33m{args.profile}\033[0m | Max Tokens: {args.max_tokens}")
    print("=" * 70 + "\n")

    print(f"📦 Loading Model: \033[1;34m{model_path}\033[0m")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).eval()

    cfg = model.config
    hidden_size = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "d_model", None)
        or (getattr(cfg, "text_config", None) and getattr(cfg.text_config, "hidden_size", None))
        or (getattr(cfg, "model_config", None) and getattr(cfg.model_config, "hidden_size", None))
    )
    if hidden_size is None:
        if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
            hidden_size = model.get_input_embeddings().embedding_dim
        else:
            hidden_size = profile_hidden_dim

    print(f"✓ Backbone Ready! Hidden Dim: \033[1;32m{hidden_size}\033[0m")

    expert_pool = None
    if args.enable_lora and not args.pure_routing:
        expert_pool = DynamicLoRAExpertPool(
            num_experts=num_experts,
            in_dim=hidden_size,
            out_dim=hidden_size,
            rank=lora_rank,
            alpha=lora_alpha,
            device=device,
            dtype=torch.bfloat16
        )

    rust_router = rust_localsleep_moe.PyFatigueRouter(num_experts, hidden_size)
    ckpt_mgr = EmbodiedCheckpointManager(checkpoint_dir="./checkpoints", profile=args.profile)
    session_logger = SessionLogger(session_dir=args.session_dir, profile=args.profile, model_path=model_path)
    harness = EmbodiedToolHarness(workspace_root=".", isolated_output_dir="./isolated_output", session_logger=session_logger)

    if not args.pure_routing:
        past_tokens = ckpt_mgr.load_state(rust_router, expert_pool=expert_pool, reset=args.reset_checkpoint)
        print("✓ \033[1;32mRust Zero-GIL Router & Tool Harness Online (Embodied Checkpoints Synced)!\033[0m\n")
    else:
        past_tokens = 0
        print("✓ \033[1;32mRust Zero-GIL Router Online (Pure Routing Compatibility Mode - No LoRA)!\033[0m\n")

    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot_id is not None and eot_id != getattr(tokenizer, "unk_token_id", None):
            stop_token_ids.add(eot_id)
    except Exception:
        pass

    engine = GenerationLoop(
        model=model,
        tokenizer=tokenizer,
        rust_router=rust_router,
        expert_pool=expert_pool,
        harness=harness,
        device=device,
        args=args,
        stop_token_ids=stop_token_ids
    )

    total_new_tokens = 0
    system_inst = KURO_COMPACT_SYSTEM_PROMPT

    if args.interactive:
        print("\n💬 \033[1;36m[Kuro Interactive Console - REPL Mode Online]\033[0m")
        print("💡 支援 Tool Calls、/recall、<file path=\"...\">，Ctrl+C 兩段式優雅中斷。\n")
        history = []
        last_sigint_time = 0.0

        while True:
            try:
                user_msg = input("\033[1;33mLeon >> \033[0m").strip()
                last_sigint_time = 0.0
            except KeyboardInterrupt:
                now = time.time()
                if now - last_sigint_time < 2.0:
                    print("\n\n👋 接收到連續中斷信號。晚安，主人。Kuro 進入休眠模式。")
                    break
                else:
                    last_sigint_time = now
                    print("\n⚠️ \033[1;33m[提示]\033[0m 再按一次 Ctrl+C 即可退出程式；或者繼續輸入指令。")
                    continue
            except EOFError:
                print("\n👋 結束交談。")
                break

            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit", "q"):
                print("👋 晚安，主人。Kuro 進入休眠模式。")
                break

            if user_msg.startswith("/recall") or user_msg.startswith("/sessions"):
                parts = user_msg.split(maxsplit=1)
                if len(parts) > 1:
                    print(f"\n🔍 [歷史 Session 檢索]:\n{session_logger.recall_search(parts[1])}\n")
                else:
                    print(f"\n📂 [近期歷史 Session 清單]:\n{session_logger.recall_recent_sessions(limit=5)}\n")
                continue

            reply_text, toks = engine.run_turn(
                user_prompt=user_msg,
                system_inst=system_inst,
                conversation_history=history,
                session_logger=session_logger
            )
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "model", "content": reply_text})
            total_new_tokens += toks
    else:
        reply_text, toks = engine.run_turn(
            user_prompt=args.prompt,
            system_inst=system_inst,
            session_logger=session_logger
        )
        total_new_tokens += toks

    if not args.pure_routing:
        ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=past_tokens + total_new_tokens)

    if hasattr(rust_router, "get_expert_states"):
        final_states = rust_router.get_expert_states()
        print("\n📊 \033[1;36m[Final Rust Expert Homeostasis Snapshot]:\033[0m")
        state_labels = {0: "\033[1;32m🟢 ACTIVE\033[0m", 1: "\033[1;33m🟡 FATIGUED\033[0m", 2: "\033[1;35m💤 NREM_SLEEP\033[0m"}
        for exp_id, s_val, fat, toks, cnt in final_states:
            bar_len = min(20, int(fat * 2))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  • Expert {exp_id}: [{state_labels.get(s_val, '⚪')}] | Fatigue: \033[1;33m{fat:5.2f}\033[0m [{bar}] | Tokens: {toks:4d} | Countdown: {cnt}")

    print("-" * 70 + "\n")

if __name__ == "__main__":
    main()
