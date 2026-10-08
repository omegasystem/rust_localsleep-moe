#!/usr/bin/env python3
"""
LocalSleep-MoE Streamlined Inference Pipeline (Entropy-Minimized & Embodied)
Components decoupled:
  - checkpoint_manager.py : Embodied SafeTensors / JSON persistence (8-Expert LoRA + State)
  - expert_pool_lora.py   : Dynamic LoRA Expert Pool (VRAM residual modulation)
  - swa_attention.py      : Sink-Aware Sliding Window Attention (Bounded KV-cache)
  - cjk_cleaner.py        : CJK stream cleaning & drama prevention
  - rust_localsleep_moe   : PyO3 Zero-GIL Native Router (<8μs dispatch)
"""

import os
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from checkpoint_manager import EmbodiedCheckpointManager
from expert_pool_lora import DynamicLoRAExpertPool
from swa_attention import trim_past_key_values_swa
from cjk_cleaner import CJKStreamCleaner

try:
    import rust_localsleep_moe
except ImportError:
    print("[!] Error: rust_localsleep_moe module not found. Build with 'maturin develop' or 'cargo build'.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="LocalSleep-MoE Refactored Real Inference")
    parser.add_argument("--config", type=str, default="configs/model_profiles.json", help="Path to model profiles JSON")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--profile", type=str, default="thor_12b")
    parser.add_argument("--prompt", type=str, default="你好，請簡短介紹你的具身認知架構與 LocalSleep-MoE 運作模式。")
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--enable-swa", action="store_true", default=True)
    parser.add_argument("--swa-sink-size", type=int, default=4)
    parser.add_argument("--swa-window-size", type=int, default=2048)
    parser.add_argument("--reset-checkpoint", action="store_true", help="Start with fresh virgin weights")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model profile if available
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
                print(f"[!] Warning reading config {cfg_path}: {e}")

    default_model_path = "/home/omegacd/local_models/gemma-4-31B-it" if "31b" in args.profile.lower() else "/home/omegacd/local_models/gemma-4-12B-it"
    model_path = args.model_path or prof.get("model_path", default_model_path)
    profile_hidden_dim = prof.get("hidden_dim", 5376 if "31b" in args.profile.lower() else 3840)
    lora_rank = prof.get("lora_rank", 32 if "31b" in args.profile.lower() else 16)
    lora_alpha = prof.get("lora_alpha", 64.0 if "31b" in args.profile.lower() else 32.0)
    num_experts = prof.get("num_experts", 8)

    print("\n" + "=" * 65)
    print("🐈‍⬛ \033[1;36mLocalSleep-MoE Streamlined Embodied Inference Pipeline\033[0m")
    print(f"🔧 Device: \033[1;32m{device.upper()}\033[0m | Profile: \033[1;33m{args.profile}\033[0m | Max Tokens: {args.max_tokens}")
    print("=" * 65 + "\n")

    # 1. Load Tokenizer & Base Backbone
    print(f"📦 Loading Base Model: \033[1;34m{model_path}\033[0m")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).eval()

    # Robust hidden_size extraction (handles Gemma4UnifiedConfig, multimodal wrappers & config fallbacks)
    cfg = model.config
    hidden_size = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "d_model", None)
        or (getattr(cfg, "text_config", None) and getattr(cfg.text_config, "hidden_size", None))
        or (getattr(cfg, "model_config", None) and getattr(cfg.model_config, "hidden_size", None))
        or (getattr(cfg, "vision_config", None) and getattr(cfg, "text_config", None) and getattr(cfg.text_config, "hidden_size", None))
    )
    if hidden_size is None:
        if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
            hidden_size = model.get_input_embeddings().embedding_dim
        else:
            hidden_size = profile_hidden_dim

    print(f"✓ Backbone Ready! Hidden Dim: \033[1;32m{hidden_size}\033[0m")

    # 2. Initialize 8-Expert Dynamic LoRA Pool in GPU VRAM
    expert_pool = DynamicLoRAExpertPool(
        num_experts=num_experts,
        in_dim=hidden_size,
        out_dim=hidden_size,
        rank=lora_rank,
        alpha=lora_alpha,
        device=device,
        dtype=torch.bfloat16
    )
    print(f"✓ Dynamic LoRA Expert Pool: {num_experts} Experts Allocated on VRAM (Rank {lora_rank}, Alpha {lora_alpha}).")

    # 3. Initialize Rust Zero-GIL Router
    # Note: PyFatigueRouter signature is: (num_experts=8, hidden_dim=3840, top_k=2, ...)
    rust_router = rust_localsleep_moe.PyFatigueRouter(num_experts, hidden_size)
    ckpt_mgr = EmbodiedCheckpointManager(checkpoint_dir="./checkpoints", profile=args.profile)
    
    # 4. Zero-Copy Auto-Resume State & 8 Experts
    past_tokens = ckpt_mgr.load_state(rust_router, expert_pool=expert_pool, reset=args.reset_checkpoint)
    print("✓ \033[1;32mRust Zero-GIL Router Online & Synced!\033[0m\n")

    # Format Prompt: Ensure system prompt is injected inside user turn for Gemma-4
    system_inst = "你是數位黑貓 Kuro，Leon 的 AI Familiar。請以忠誠、冷靜、極具神經架構深度的專業繁體中文回答。請直接輸出結論，嚴禁在文末附加任何註記、英文翻譯或自我修正說明。"
    try:
        formatted_history = [
            {"role": "user", "content": f"{system_inst}\n\n{args.prompt}"}
        ]
        prompt_text = tokenizer.apply_chat_template(formatted_history, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_text = f"<start_of_turn>user\n{system_inst}\n\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

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

    print(f"💬 \033[1;33mPrompt:\033[0m {args.prompt}\n")
    print("⚡ \033[1;32m[Kuro Real-Time Generation Stream (Rust Zero-GIL & SWA Bounded)]:\033[0m\n")

    generated_ids = input_ids.clone()
    curr_input_ids = input_ids
    past_key_values = None
    t_start = time.time()
    tokens_generated = 0
    cjk_cleaner = CJKStreamCleaner()
    accumulated_full_text = ""

    STOP_PHRASES = [
        "**thought", "<thought>", "</thought>", "<start_of_turn>",
        "*註記", "*(note", "(note:", "(wait,", "(self-correction", "(final correction",
        "**修正後", "修正後的純粹", "kuro: ai familiar 核心宣告"
    ]

    with torch.no_grad():
        for step in range(args.max_tokens):
            outputs = model(
                curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values

            # SWA KV-Cache Boundedness
            if args.enable_swa and past_key_values is not None:
                past_key_values = trim_past_key_values_swa(
                    past_key_values,
                    sink_size=args.swa_sink_size,
                    window_size=args.swa_window_size
                )

            logits = outputs.logits[:, -1, :].clone()
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            # 1. Rust Zero-GIL Routing (<8μs)
            hidden_list = last_hidden.squeeze(0).float().cpu().tolist()
            weights, selected_exp, entropy = rust_router.forward(hidden_list, 1)
            rust_router.step_decay()

            # 2. Dynamic LoRA Expert Modulation (Inject residual delta)
            if len(selected_exp) > 0 and len(weights) > 0:
                top_exp = selected_exp[0]
                top_weight = weights[0]
                delta_h = expert_pool.forward_delta(last_hidden, top_exp, top_weight)
                # Modulate last hidden state before lm_head if accessible
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
            if generated_ids.shape[1] >= 6:
                last6 = generated_ids[0, -6:]
                if last6[0] == last6[2] == last6[4] and last6[1] == last6[3] == last6[5]:
                    logits[:, last6[4]] = -1e9
                    logits[:, last6[5]] = -1e9

            # 5. Temperature & Top-P Sampling
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

            # Stream print
            sys.stdout.write(cleaned_piece)
            sys.stdout.flush()

            # Drama Terminator & EOS Check
            if next_token_id[0].item() in stop_token_ids:
                break

            tail_lower = accumulated_full_text[-60:].lower()
            stop_phrases_filtered = [p for p in STOP_PHRASES if p != "<start_of_turn>"]
            if any(phrase in tail_lower or phrase in cleaned_piece.lower() for phrase in stop_phrases_filtered):
                break

            # Repetition / Loop Circuit Breaker (Catch repeated identical paragraphs)
            if len(accumulated_full_text) > 200:
                recent_150 = accumulated_full_text[-150:]
                earlier_text = accumulated_full_text[:-150]
                if recent_150 in earlier_text:
                    print("\n⚠️ \033[1;33m[Loop Circuit Breaker Triggered]\033[0m Detected text replay loop. Halting generation.")
                    break

    t_elapsed = time.time() - t_start
    tps = tokens_generated / t_elapsed if t_elapsed > 0 else 0
    print("\n\n" + "-" * 65)
    print(f"🏁 \033[1;32mInference Complete:\033[0m Generated {tokens_generated} tokens in {t_elapsed:.2f}s ({tps:.2f} tps)")

    # 6. Auto-Save Embodied State + 8 Experts via SafeTensors
    ckpt_mgr.save_state(rust_router, expert_pool=expert_pool, tokens_generated=past_tokens + tokens_generated)

    # 7. Render final expert states summary with ASCII gauge bars
    if hasattr(rust_router, "get_expert_states"):
        final_states = rust_router.get_expert_states()
        print("\n📊 \033[1;36m[Final Rust Expert Homeostasis Snapshot]:\033[0m")
        state_labels = {0: "\033[1;32m🟢 ACTIVE\033[0m", 1: "\033[1;33m🟡 FATIGUED\033[0m", 2: "\033[1;35m💤 NREM_SLEEP\033[0m"}
        for exp_id, s_val, fat, toks, cnt in final_states:
            bar_len = min(20, int(fat * 2))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  • Expert {exp_id}: [{state_labels.get(s_val, '⚪')}] | Fatigue: \033[1;33m{fat:5.2f}\033[0m [{bar}] | Tokens: {toks:4d} | Countdown: {cnt}")

    print("-" * 65 + "\n")

if __name__ == "__main__":
    main()
