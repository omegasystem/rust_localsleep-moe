#!/usr/bin/env python3
"""
ls_moe_agentic.py - Agentic Embodied Inference Pipeline for LocalSleep-MoE
Integrates:
  - Rust Zero-GIL MoE Router (<8μs dispatch)
  - 8-Expert Dynamic LoRA Synaptic Pool in GPU VRAM
  - Sink-Aware Sliding Window Attention (SWA bounded KV cache & Dynamic Rolling)
  - Pre-flight Context Budget & Dynamic Sliding Window Rolling
  - Multi-Turn Session History & Turn-by-Turn Markdown/JSONL Archiving (SessionLogger)
  - Historical Session Recall & Search Tools (recall_sessions, recall_search)
  - Two-Stage Interactive Ctrl+C Interruption (1st Ctrl+C stops generation; 2nd Ctrl+C exits REPL)
  - Embodied Tool Harness (read_file, list_dir, web_search + XML multi-line <file path="..."> isolated write)
  - Negentropy Compressed System Prompt (<200 tokens)
  - SafeTensors Continuous Embodied State Persistence
"""

import os
import sys
import time
import json
import re
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from checkpoint_manager import EmbodiedCheckpointManager
from expert_pool_lora import DynamicLoRAExpertPool
from swa_attention import trim_past_key_values_swa
from cjk_cleaner import CJKStreamCleaner
from tools_harness import EmbodiedToolHarness
from session_logger import SessionLogger

try:
    import rust_localsleep_moe
except ImportError:
    print("[!] Error: rust_localsleep_moe module not found. Build with 'maturin develop' or 'cargo build'.")
    sys.exit(1)

# --- High-Density Compressed System Prompt (<200 Tokens) ---
# 轉移舊版優點：支援原生多行 XML 寫檔 <file path="...">代碼</file>，避免單行 JSON 逸出 \n 破壞格式！
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
    parser.add_argument("--max-context-tokens", type=int, default=3072, help="上下文總預算門檻，超出即在寫 Code / 推論前執行壓縮")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--interactive", action="store_true", default=False, help="啟動多輪交談互動模式 (REPL)")
    parser.add_argument("--enable-swa", action="store_true", default=True, help="啟用 SWA KV-Cache 窗口截斷與動態滾動續命 (預設開啟)")
    parser.add_argument("--disable-swa", action="store_true", default=False, help="明確關閉 SWA KV-Cache")
    parser.add_argument("--enable-lora", action="store_true", default=False, help="啟用 Dynamic LoRA 專家調制與權重存檔 (預設關閉走純分流相容模式)")
    parser.add_argument("--pure-routing", action="store_true", default=False, help="強制純分流相容模式 (無 LoRA 侵入修改與存檔，完全保持原生注意力)")
    parser.add_argument("--swa-sink-size", type=int, default=4)
    parser.add_argument("--swa-window-size", type=int, default=2048)
    parser.add_argument("--session-dir", type=str, default="./sessions", help="歷史 session 儲存目錄 (Markdown 與 JSONL)")
    parser.add_argument("--reset-checkpoint", action="store_true")
    return parser.parse_args()

def execute_tool_calls(text: str, harness: EmbodiedToolHarness):
    """Detects and executes <tool_call> JSON blocks, returning tool outputs."""
    tool_pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(tool_pattern, text, re.DOTALL)
    if not matches:
        return None, text

    results = []
    for m in matches:
        try:
            call_obj = json.loads(m.strip())
            act = call_obj.get("action")
            args = call_obj.get("args", {})
            print(f"\n⚡ \033[1;35m[Tool Dispatched]\033[0m: \033[1;33m{act}\033[0m with args: {args}")
            out = harness.dispatch(act, args)
            results.append(f"<tool_response action=\"{act}\">\n{out}\n</tool_response>")
        except Exception as e:
            results.append(f"<tool_response error=\"{e}\" />")
    return "\n".join(results), re.sub(tool_pattern, "", text, flags=re.DOTALL)

def prune_and_compress_history(conversation_history: list, tokenizer, max_allowed_tokens: int = 1500) -> list:
    """
    動態對話歷史修剪與壓縮：
    保留最近的對話輪次，並確保歷史總 Token 不超過預算，為即將生成的大型 Code 預留足夠上下文窗口！
    """
    if not conversation_history:
        return []

    compressed_history = list(conversation_history)
    while len(compressed_history) > 2:
        test_text = "".join([m["content"] for m in compressed_history])
        num_toks = len(tokenizer.encode(test_text, add_special_tokens=False))
        if num_toks <= max_allowed_tokens:
            break
        compressed_history.pop(0)
        if compressed_history:
            compressed_history.pop(0)

    return compressed_history

def run_generation_turn(
    user_prompt: str,
    system_inst: str,
    model,
    tokenizer,
    rust_router,
    expert_pool,
    harness: EmbodiedToolHarness,
    device,
    args,
    stop_token_ids,
    conversation_history=None,
    session_logger: SessionLogger = None
):
    """Executes a single model generation turn with streaming output, tool call injection, circuit breakers, and Ctrl+C interruptibility."""
    safe_history = conversation_history
    if conversation_history:
        history_tokens = len(tokenizer.encode("".join([m["content"] for m in conversation_history]), add_special_tokens=False))
        user_tokens = len(tokenizer.encode(user_prompt, add_special_tokens=False))
        total_pre_tokens = history_tokens + user_tokens

        max_prompt_budget = max(1024, args.max_context_tokens - args.max_tokens)
        if total_pre_tokens > max_prompt_budget:
            print(f"⚠️ \033[1;33m[Context Pre-flight Alert]\033[0m: Prompt ({total_pre_tokens} toks) > Budget ({max_prompt_budget} toks). 正在自動壓縮歷史視窗...")
            safe_history = prune_and_compress_history(conversation_history, tokenizer, max_allowed_tokens=max_prompt_budget - user_tokens)

    if "12b" in args.profile.lower():
        if safe_history:
            formatted_messages = list(safe_history) + [{"role": "user", "content": user_prompt}]
        else:
            formatted_messages = [{"role": "user", "content": f"{system_inst}\n\n{user_prompt}"}]
        try:
            prompt_text = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = f"<start_of_turn>user\n{system_inst}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    else:
        try:
            if safe_history:
                messages = [{"role": "system", "content": system_inst}] + list(safe_history) + [{"role": "user", "content": user_prompt}]
            else:
                messages = [{"role": "system", "content": system_inst}, {"role": "user", "content": user_prompt}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_messages = [{"role": "user", "content": f"{system_inst}\n\n{user_prompt}"}]
            prompt_text = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prefill_len = input_ids.shape[1]

    print(f"💬 \033[1;33mPrompt:\033[0m {user_prompt}")
    print(f"📊 \033[1;36m[Context Budget]:\033[0m Prefill: {prefill_len} toks | Max Gen: {args.max_tokens} | Total Window Limit: {args.max_context_tokens}")
    print("⚡ \033[1;32m[Kuro Agentic Stream (Rust Zero-GIL & Tools)]:\033[0m (按 Ctrl+C 可暫停此輪推論)\n")

    generated_ids = input_ids.clone()
    curr_input_ids = input_ids
    past_key_values = None
    t_start = time.time()
    tokens_generated = 0
    cjk_cleaner = CJKStreamCleaner()
    accumulated_full_text = ""
    was_interrupted = False
    tool_calls_count = 0
    consecutive_tool_errors = 0
    MAX_TOOL_CALLS_PER_TURN = 3

    STOP_PHRASES = [
        "</thought>", "**thought", "<thought>",
        "*註記", "*(note", "(note:", "(wait,", "(self-correction", "(final correction",
        "**修正後", "修正後的純粹", "kuro: ai familiar 核心宣告"
    ]

    use_swa = (args.enable_swa and not args.disable_swa)

    with torch.no_grad():
        for step in range(args.max_tokens):
            try:
                outputs = model(
                    curr_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
            except KeyboardInterrupt:
                print("\n\n⏸️ \033[1;33m[Generation Paused by Leon via Ctrl+C]\033[0m 此輪推論已暫停，保留已生成內容。")
                was_interrupted = True
                break

            past_key_values = outputs.past_key_values

            # SWA 滑動視窗修剪機制 (保留首部 Sink + 尾部 Window，實現無限 Token 續命推論)
            if use_swa and past_key_values is not None:
                past_key_values = trim_past_key_values_swa(
                    past_key_values,
                    sink_size=args.swa_sink_size,
                    window_size=args.swa_window_size
                )

            logits = outputs.logits[:, -1, :].clone()
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            # 1. Rust Zero-GIL Routing (<8μs dispatch & telemetry)
            hidden_list = last_hidden.squeeze(0).float().cpu().tolist()
            weights, selected_exp, entropy = rust_router.forward(hidden_list, 1)
            rust_router.step_decay()

            # 2. Dynamic LoRA Expert Modulation
            if expert_pool is not None and not args.pure_routing and len(selected_exp) > 0 and len(weights) > 0:
                top_exp = selected_exp[0]
                top_weight = weights[0]
                delta_h = expert_pool.forward_delta(last_hidden, top_exp, top_weight)
                if hasattr(model, "lm_head"):
                    mod_logits = model.lm_head(last_hidden + delta_h)
                    logits = 0.5 * logits + 0.5 * mod_logits

            # 3. Repetition Penalty
            if args.repetition_penalty != 1.0 and generated_ids.shape[1] > 1:
                prev_tokens = torch.unique(generated_ids)
                prev_logits = logits[:, prev_tokens]
                logits[:, prev_tokens] = torch.where(
                    prev_logits < 0,
                    prev_logits * args.repetition_penalty,
                    prev_logits / args.repetition_penalty
                )

            # 4. Anti-Loop Stutter Breaker
            if generated_ids.shape[1] >= 3:
                last3 = generated_ids[0, -3:]
                if last3[0] == last3[1] == last3[2]:
                    logits[:, last3[0]] = -1e9

            # 5. Sampling
            scaled_logits = logits / args.temp if args.temp > 0 else logits
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                scaled_logits[:, sorted_indices[sorted_indices_to_remove]] = -float("Inf")

            probs = torch.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            curr_input_ids = next_token_id
            tokens_generated += 1

            raw_piece = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            cleaned_piece = cjk_cleaner.clean(raw_piece)
            accumulated_full_text += cleaned_piece

            sys.stdout.write(cleaned_piece)
            sys.stdout.flush()

            # EOS & Drama Terminator
            if next_token_id[0].item() in stop_token_ids:
                break

            tail_lower = accumulated_full_text[-60:].lower()
            stop_phrases_filtered = [p for p in STOP_PHRASES if p != "<start_of_turn>"]
            if any(phrase in tail_lower or phrase in cleaned_piece.lower() for phrase in stop_phrases_filtered):
                break

            if len(accumulated_full_text) > 200:
                recent_150 = accumulated_full_text[-150:]
                earlier_text = accumulated_full_text[:-150]
                if recent_150 in earlier_text:
                    print("\n⚠️ \033[1;33m[Loop Circuit Breaker Triggered]\033[0m")
                    break

            # Check for completed tool call block
            if "</tool_call>" in accumulated_full_text:
                tool_output, _ = execute_tool_calls(accumulated_full_text, harness)
                if tool_output:
                    tool_calls_count += 1
                    if "File not found" in tool_output or "error" in tool_output.lower():
                        consecutive_tool_errors += 1
                    else:
                        consecutive_tool_errors = 0

                    print(f"\n📥 \033[1;32m[Tool Result Injected (Call #{tool_calls_count})]:\033[0m\n{tool_output}\n")

                    # 防爆斷路器 (Circuit Breaker)
                    if tool_calls_count >= MAX_TOOL_CALLS_PER_TURN or consecutive_tool_errors >= 2:
                        cb_msg = (
                            f"\n\n⚠️ \033[1;31m[Tool Circuit Breaker Triggered]\033[0m: "
                            f"已達到單輪工具呼叫上限 ({tool_calls_count}/{MAX_TOOL_CALLS_PER_TURN}) 或連續遭遇錯誤。"
                            f"強制截斷自動調度，防止注意力崩潰與死循環。\n"
                        )
                        print(cb_msg)
                        accumulated_full_text += f"\n<circuit_breaker reason=\"max_calls_or_repeated_errors\" />\n"
                        break

                    tool_ids = tokenizer(f"\n{tool_output}\n", return_tensors="pt").input_ids.to(device)
                    generated_ids = torch.cat([generated_ids, tool_ids], dim=-1)
                    curr_input_ids = tool_ids[:, -1:]
                    accumulated_full_text = ""

    # 自動掃描並抽取多行 XML Artifacts (<file path="...">...) 存入沙盒
    saved_arts = harness.extract_and_save_artifacts(accumulated_full_text)
    if saved_arts:
        print(f"\n✨ \033[1;32m[Embodied Artifacts Persisted]\033[0m Total files saved: {len(saved_arts)}")

    t_elapsed = time.time() - t_start
    tps = tokens_generated / t_elapsed if t_elapsed > 0 else 0
    status_str = "Turn Interrupted" if was_interrupted else "Turn Complete"
    print("\n\n" + "-" * 70)
    print(f"🏁 \033[1;32m{status_str}:\033[0m Generated {tokens_generated} tokens in {t_elapsed:.2f}s ({tps:.2f} tps)")

    # 轉移舊版優點：儲存歷史 Session Markdown 與 JSONL
    if session_logger:
        session_logger.append_turn(
            user_prompt=user_prompt,
            assistant_reply=accumulated_full_text,
            stats={"tokens": tokens_generated, "tps": tps, "elapsed": t_elapsed, "interrupted": was_interrupted},
            artifacts=saved_arts
        )

    return accumulated_full_text, tokens_generated

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Multi-path config resolution
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
            except Exception as e:
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

    # 1. Load Tokenizer & Base Backbone
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
        or (getattr(cfg, "vision_config", None) and getattr(cfg.text_config, None) and getattr(cfg.text_config, "hidden_size", None))
    )
    if hidden_size is None:
        if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
            hidden_size = model.get_input_embeddings().embedding_dim
        else:
            hidden_size = profile_hidden_dim

    print(f"✓ Backbone Ready! Hidden Dim: \033[1;32m{hidden_size}\033[0m")

    # 2. Dynamic LoRA Pool & Rust Router
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
        print("✓ \033[1;32mRust Zero-GIL Router Online (Pure Routing Compatibility Mode - No LoRA / No Disk State)!\033[0m\n")

    # Collect EOS and EOT tokens
    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot_id is not None and eot_id != getattr(tokenizer, "unk_token_id", None):
            stop_token_ids.add(eot_id)
    except Exception:
        pass

    total_new_tokens = 0
    system_inst = KURO_COMPACT_SYSTEM_PROMPT

    if args.interactive:
        print("\n💬 \033[1;36m[Kuro Interactive Console - REPL Mode Online]\033[0m")
        print("💡 輸入指令交談（支援 Tool Calls、/recall、與 <file path=\"...\">）。")
        print("🛡️ 【兩段式中斷防護】：推論生成中按 Ctrl+C 僅暫停當前輸出；在等待輸入提示符下連續按 Ctrl+C 兩次退出。\n")
        history = []
        last_sigint_time = 0.0

        while True:
            try:
                user_msg = input("\033[1;33mLeon >> \033[0m").strip()
                last_sigint_time = 0.0  # 正常輸入成功則重置中斷計時器
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

            # 快捷指令支援：/recall 或 /sessions 查詢歷史
            if user_msg.startswith("/recall") or user_msg.startswith("/sessions"):
                parts = user_msg.split(maxsplit=1)
                if len(parts) > 1:
                    query = parts[1]
                    res = session_logger.recall_search(query)
                    print(f"\n🔍 \033[1;36m[歷史 Session 檢索結果 - 關鍵字: {query}]:\033[0m\n{res}\n")
                else:
                    res = session_logger.recall_recent_sessions(limit=5)
                    print(f"\n📂 \033[1;36m[近期歷史 Session 清單]:\033[0m\n{res}\n")
                continue

            reply_text, toks = run_generation_turn(
                user_prompt=user_msg,
                system_inst=system_inst,
                model=model,
                tokenizer=tokenizer,
                rust_router=rust_router,
                expert_pool=expert_pool,
                harness=harness,
                device=device,
                args=args,
                stop_token_ids=stop_token_ids,
                conversation_history=history,
                session_logger=session_logger
            )
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "model", "content": reply_text})
            total_new_tokens += toks
    else:
        reply_text, toks = run_generation_turn(
            user_prompt=args.prompt,
            system_inst=system_inst,
            model=model,
            tokenizer=tokenizer,
            rust_router=rust_router,
            expert_pool=expert_pool,
            harness=harness,
            device=device,
            args=args,
            stop_token_ids=stop_token_ids,
            session_logger=session_logger
        )
        total_new_tokens += toks

    # 6. Auto-Save Embodied State + 8 Experts via SafeTensors
    if not args.pure_routing:
        ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=past_tokens + total_new_tokens)

    # 7. Render final expert states summary
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
