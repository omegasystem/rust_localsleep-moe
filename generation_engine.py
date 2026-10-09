import time
import sys
import re
import torch
from typing import List, Dict, Tuple, Optional, Set, Any
from transformers import AutoTokenizer

from tools_harness import EmbodiedToolHarness
from tool_parser import ToolCallParser
from session_logger import SessionLogger
from swa_attention import trim_past_key_values_swa
from cjk_cleaner import CJKStreamCleaner

def prune_and_compress_history(conversation_history: list, tokenizer: AutoTokenizer, max_allowed_tokens: int = 1500) -> list:
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

def format_prompt(
    user_prompt: str,
    system_inst: str,
    tokenizer: AutoTokenizer,
    profile: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """建構符合 Gemma Chat Template 的多輪或單輪輸入 Prompt"""
    if "12b" in profile.lower():
        if conversation_history:
            formatted_messages = list(conversation_history) + [{"role": "user", "content": user_prompt}]
        else:
            formatted_messages = [{"role": "user", "content": f"{system_inst}\n\n{user_prompt}"}]
        try:
            return tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"<start_of_turn>user\n{system_inst}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    else:
        try:
            if conversation_history:
                messages = [{"role": "system", "content": system_inst}] + list(conversation_history) + [{"role": "user", "content": user_prompt}]
            else:
                messages = [{"role": "system", "content": system_inst}, {"role": "user", "content": user_prompt}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_messages = [{"role": "user", "content": f"{system_inst}\n\n{user_prompt}"}]
            return tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)

class GenerationLoop:
    """
    單輪推論生成循環核心：
    封裝 Rust Zero-GIL Routing、Dynamic LoRA、SWA 修剪、Tool 容錯攔截與抗漂移斷路器。
    """
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        rust_router,
        expert_pool,
        harness: EmbodiedToolHarness,
        device: str,
        args,
        stop_token_ids: Set[int]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.rust_router = rust_router
        self.expert_pool = expert_pool
        self.harness = harness
        self.device = device
        self.args = args
        self.stop_token_ids = stop_token_ids

        self.STOP_PHRASES = [
            "</thought>", "**thought", "<thought>",
            "*註記", "*(note", "(note:", "(wait,", "(self-correction", "(final correction",
            "**修正後", "修正後的純粹", "kuro: ai familiar 核心宣告"
        ]

    def run_turn(
        self,
        user_prompt: str,
        system_inst: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_logger: Optional[SessionLogger] = None
    ) -> Tuple[str, int]:
        safe_history = conversation_history
        if conversation_history:
            history_tokens = len(self.tokenizer.encode("".join([m["content"] for m in conversation_history]), add_special_tokens=False))
            user_tokens = len(self.tokenizer.encode(user_prompt, add_special_tokens=False))
            total_pre_tokens = history_tokens + user_tokens

            max_prompt_budget = max(1024, self.args.max_context_tokens - self.args.max_tokens)
            if total_pre_tokens > max_prompt_budget:
                print(f"⚠️ \033[1;33m[Context Pre-flight Alert]\033[0m: Prompt ({total_pre_tokens} toks) > Budget ({max_prompt_budget} toks). 自動壓縮歷史視窗...")
                safe_history = prune_and_compress_history(conversation_history, self.tokenizer, max_allowed_tokens=max_prompt_budget - user_tokens)

        prompt_text = format_prompt(user_prompt, system_inst, self.tokenizer, self.args.profile, safe_history)
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        prefill_len = input_ids.shape[1]

        print(f"💬 \033[1;33mPrompt:\033[0m {user_prompt}")
        print(f"📊 \033[1;36m[Context Budget]:\033[0m Prefill: {prefill_len} toks | Max Gen: {self.args.max_tokens} | Window: {self.args.max_context_tokens}")
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
        recent_tool_actions = []

        use_swa = (self.args.enable_swa and not self.args.disable_swa)

        with torch.no_grad():
            for step in range(self.args.max_tokens):
                try:
                    outputs = self.model(
                        curr_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True
                    )
                except KeyboardInterrupt:
                    print("\n\n⏸️ \033[1;33m[Generation Paused by Leon via Ctrl+C]\033[0m 此輪推論已暫停。")
                    was_interrupted = True
                    break

                past_key_values = outputs.past_key_values

                # SWA 滑動視窗修剪機制 (保留首部 Sink + 尾部 Window)
                if use_swa and past_key_values is not None:
                    past_key_values = trim_past_key_values_swa(
                        past_key_values,
                        sink_size=self.args.swa_sink_size,
                        window_size=self.args.swa_window_size
                    )

                logits = outputs.logits[:, -1, :].clone()
                last_hidden = outputs.hidden_states[-1][:, -1, :]

                # 1. Rust Zero-GIL Routing (<8μs dispatch)
                hidden_list = last_hidden.squeeze(0).float().cpu().tolist()
                weights, selected_exp, entropy = self.rust_router.forward(hidden_list, 1)
                self.rust_router.step_decay()

                # 2. Dynamic LoRA Modulation
                if self.expert_pool is not None and not self.args.pure_routing and len(selected_exp) > 0 and len(weights) > 0:
                    top_exp = selected_exp[0]
                    top_weight = weights[0]
                    delta_h = self.expert_pool.forward_delta(last_hidden, top_exp, top_weight)
                    if hasattr(self.model, "lm_head"):
                        mod_logits = self.model.lm_head(last_hidden + delta_h)
                        logits = 0.5 * logits + 0.5 * mod_logits

                # 3. Repetition Penalty
                if self.args.repetition_penalty != 1.0 and generated_ids.shape[1] > 1:
                    prev_tokens = torch.unique(generated_ids)
                    prev_logits = logits[:, prev_tokens]
                    logits[:, prev_tokens] = torch.where(
                        prev_logits < 0,
                        prev_logits * self.args.repetition_penalty,
                        prev_logits / self.args.repetition_penalty
                    )

                # 4. Anti-Loop Stutter Breaker
                if generated_ids.shape[1] >= 3:
                    last3 = generated_ids[0, -3:]
                    if last3[0] == last3[1] == last3[2]:
                        logits[:, last3[0]] = -1e9

                # 5. Sampling
                scaled_logits = logits / self.args.temp if self.args.temp > 0 else logits
                if self.args.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.args.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    scaled_logits[:, sorted_indices[sorted_indices_to_remove]] = -float("Inf")

                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                curr_input_ids = next_token_id
                tokens_generated += 1

                raw_piece = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                cleaned_piece = cjk_cleaner.clean(raw_piece)
                accumulated_full_text += cleaned_piece

                sys.stdout.write(cleaned_piece)
                sys.stdout.flush()

                # EOS & Stop Tokens
                if next_token_id[0].item() in self.stop_token_ids:
                    break

                tail_lower = accumulated_full_text[-60:].lower()
                stop_phrases_filtered = [p for p in self.STOP_PHRASES if p != "<start_of_turn>"]
                if any(phrase in tail_lower or phrase in cleaned_piece.lower() for phrase in stop_phrases_filtered):
                    break

                # 迴圈重複斷路器
                if len(accumulated_full_text) > 200:
                    recent_150 = accumulated_full_text[-150:]
                    earlier_text = accumulated_full_text[:-150]
                    if recent_150 in earlier_text:
                        print("\n⚠️ \033[1;33m[Loop Circuit Breaker Triggered]\033[0m")
                        break

                # 6. Tool Call 容錯偵測與即時攔截
                # 只要出現閉合標籤或換行/turn結束符號，立即執行 Tool 判定
                has_tool_signal = (
                    "</tool_call>" in accumulated_full_text
                    or ("<tool_call>" in accumulated_full_text and (
                        "</tool_call" in accumulated_full_text
                        or "<end_of_turn>" in accumulated_full_text
                        or accumulated_full_text.endswith("}\n")
                    ))
                    or ("<call:fs:" in accumulated_full_text and "/>" in accumulated_full_text)
                )

                if has_tool_signal:
                    action, tool_output, clean_text = ToolCallParser.detect_and_execute(accumulated_full_text, self.harness)
                    if action and tool_output is not None:
                        # 動作防漂移熔斷判定
                        norm_act_sig = f"{action}:{tool_output[:40]}"
                        if norm_act_sig in recent_tool_actions:
                            print(f"\n\n🛑 \033[1;31m[Anti-Drift Breaker]\033[0m 重複執行同一工具動作，強制終止當前輪次。")
                            accumulated_full_text = clean_text + f"\n<circuit_breaker reason=\"duplicate_tool_action\" />\n"
                            break
                        recent_tool_actions.append(norm_act_sig)

                        tool_calls_count += 1
                        if "File not found" in tool_output or "error" in tool_output.lower():
                            consecutive_tool_errors += 1
                        else:
                            consecutive_tool_errors = 0

                        print(f"\n📥 \033[1;32m[Tool Dispatched & Injected (#{tool_calls_count}) - {action}]:\033[0m\n{tool_output}\n")

                        # 工具次數上限熔斷
                        if tool_calls_count >= MAX_TOOL_CALLS_PER_TURN or consecutive_tool_errors >= 2:
                            print(f"\n⚠️ \033[1;31m[Tool Breaker]\033[0m 達到單輪調用死限 ({tool_calls_count}/{MAX_TOOL_CALLS_PER_TURN})，停止推論。")
                            accumulated_full_text = clean_text + f"\n<circuit_breaker reason=\"max_tool_calls_reached\" />\n"
                            break

                        # 角色閉環 Turn 切換與 KV-Cache 重新 Prefill 注入
                        injection_text = (
                            f"<end_of_turn>\n"
                            f"<start_of_turn>user\n"
                            f"【環境工具回傳結果 (System Tool Response)】：\n"
                            f"{tool_output}\n"
                            f"請根據上述真實工具回傳的內容繼續回答或執行下一步。\n"
                            f"<end_of_turn>\n"
                            f"<start_of_turn>model\n"
                        )
                        tool_ids = self.tokenizer(injection_text, return_tensors="pt").input_ids.to(self.device)

                        # 真正 Prefill 進 KV-Cache
                        prefill_outputs = self.model(
                            tool_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_hidden_states=True
                        )
                        past_key_values = prefill_outputs.past_key_values
                        if use_swa and past_key_values is not None:
                            past_key_values = trim_past_key_values_swa(
                                past_key_values,
                                sink_size=self.args.swa_sink_size,
                                window_size=self.args.swa_window_size
                            )

                        generated_ids = torch.cat([generated_ids, tool_ids], dim=-1)
                        curr_input_ids = tool_ids[:, -1:]
                        accumulated_full_text = clean_text + f"\n[Executed: {action}]\n"

        # 抽取多行 XML Artifacts 存入沙盒
        saved_arts = self.harness.extract_and_save_artifacts(accumulated_full_text)
        if saved_arts:
            print(f"\n✨ \033[1;32m[Embodied Artifacts Persisted]\033[0m Total files saved: {len(saved_arts)}")

        t_elapsed = time.time() - t_start
        tps = tokens_generated / t_elapsed if t_elapsed > 0 else 0
        status_str = "Turn Interrupted" if was_interrupted else "Turn Complete"
        print("\n\n" + "-" * 70)
        print(f"🏁 \033[1;32m{status_str}:\033[0m Generated {tokens_generated} tokens in {t_elapsed:.2f}s ({tps:.2f} tps)")

        if session_logger:
            session_logger.append_turn(
                user_prompt=user_prompt,
                assistant_reply=accumulated_full_text,
                stats={"tokens": tokens_generated, "tps": tps, "elapsed": t_elapsed, "interrupted": was_interrupted},
                artifacts=saved_arts
            )

        return accumulated_full_text, tokens_generated
