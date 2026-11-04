import os
import glob
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class SessionLogger:
    """
    通用型 Session Archiving, Markdown Turn Logger 與累計效能見證紀錄器 (Optimization Witness Logger)。
    自動保留每次推論的完整內容、速度指標 (`🏁 [Hybrid Turn Complete] ...`) 與 Rust MoE 專家狀態 (`🦀 [Rust MoE Telemetry] ...`)。
    """
    def __init__(self, session_dir="./sessions", profile="default", model_path="", benchmark_dir="./benchmarks"):
        self.session_dir = os.path.abspath(session_dir)
        self.benchmark_dir = os.path.abspath(benchmark_dir)
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.benchmark_dir, exist_ok=True)
        self.profile = profile
        self.model_path = model_path

        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{profile}_{session_ts}"
        self.md_file = os.path.join(self.session_dir, f"session_{self.session_id}.md")
        self.jsonl_file = os.path.join(self.session_dir, f"session_{self.session_id}.jsonl")
        self.realtime_telemetry_file = os.path.join(self.session_dir, f"session_real_telemetry_{self.session_id}.jsonl")
        self._rt_step = 0
        self.witness_md = os.path.join(self.benchmark_dir, "OPTIMIZATION_WITNESS_LOG.md")
        self.witness_jsonl = os.path.join(self.benchmark_dir, "optimization_witness_history.jsonl")

        self._init_logs()

    def _init_logs(self):
        with open(self.md_file, "w", encoding="utf-8") as f:
            f.write(f"# Agentic Session Log: {self.session_id}\n")
            f.write(f"- **Profile:** `{self.profile}`\n")
            f.write(f"- **Model:** `{self.model_path}`\n")
            f.write(f"- **Start Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")

    def record_event(self, event_type: str, data: dict):
        """JSONL 微秒級遙測紀錄 (供後續除錯、強化學習回放與效能演進對比)"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "type": event_type,
            "data": data,
        }
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_token_telemetry(
        self,
        token_id: int,
        token_str: str,
        tps: float,
        entropy: float,
        selected_experts: List[int],
        weights: List[float],
        expert_states: List[Tuple],
        top_probs: Optional[List[float]] = None,
    ) -> None:
        """
        Writes a real-time per-token telemetry row to `session_real_telemetry_<session_id>.jsonl`
        using the exact schema required by the 3D Lorenz Attractor Player (`player.html`)!
        Includes both `routed_experts`/`routing_weights` and `selected_experts`/`weights` keys.
        """
        self._rt_step += 1
        exps_formatted = []
        for item in expert_states:
            eid, st, fat, tok, cd = item[:5]
            sp = item[5] if len(item) >= 7 else 50.0
            exps_formatted.append({
                "id": int(eid),
                "state": int(st),
                "fatigue": round(float(fat), 4),
                "tokens": int(tok),
                "countdown": int(cd),
                "norm": round(max(0.25, 2.65 - float(fat) * 0.4), 3),
                "sparsity": round(float(sp), 1),
            })

        row = {
            "ts": datetime.now().isoformat(),
            "step": self._rt_step,
            "token_id": int(token_id),
            "token_str": str(token_str),
            "tps": round(float(tps), 2),
            "entropy": round(float(entropy), 4),
            "routed_experts": [int(x) for x in selected_experts],
            "routing_weights": [round(float(w), 4) for w in weights],
            "selected_experts": [int(x) for x in selected_experts],
            "weights": [round(float(w), 4) for w in weights],
            "top_probs": [round(float(p), 4) for p in (top_probs or [])],
            "experts": exps_formatted,
        }
        with open(self.realtime_telemetry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_turn(
        self,
        role: Optional[str] = None,
        content: Optional[str] = None,
        stats: Optional[dict] = None,
        artifacts: Optional[list] = None,
        user_prompt: Optional[str] = None,
        assistant_reply: Optional[str] = None,
    ):
        if user_prompt is not None or assistant_reply is not None:
            if user_prompt:
                self.append_turn(role="user", content=user_prompt)
            if assistant_reply is not None:
                self.append_turn(role="assistant", content=assistant_reply, stats=stats, artifacts=artifacts)
            return

        role = role or "assistant"
        content = content or ""
        now_dt = datetime.now()
        cur_time = now_dt.strftime("%H:%M:%S")

        role_icons = {
            "user": "💬 **USER**",
            "assistant": "🧠 **KURO / MODEL**",
            "tool": "🛠️ **TOOL EXECUTION**",
            "system": "⚙️ **SYSTEM**",
        }
        header = role_icons.get(role, f"🔹 **{role.upper()}**")

        with open(self.md_file, "a", encoding="utf-8") as f:
            f.write(f"\n### {header} ({cur_time})\n\n")
            f.write(f"{content}\n\n")

            if stats:
                tokens = stats.get("tokens", 0)
                tps = stats.get("tps", 0.0)
                elapsed = stats.get("elapsed", 0.0)
                ttft_ms = stats.get("ttft_ms", 0.0)
                top_k_experts = stats.get("top_k_experts", [])
                mean_entropy = stats.get("mean_entropy", 0.0)
                nrem_events = stats.get("nrem_events", 0)
                gpu_24_prunes = stats.get("gpu_24_prunes", nrem_events)
                summary_banner = stats.get(
                    "summary_banner",
                    f"🏁 [Hybrid Turn Complete] Tokens: {tokens} | Elapsed: {elapsed:.2f}s ({tps:.2f} tps) | "
                    f"TTFT: {ttft_ms:.1f}ms | Top-2: {top_k_experts} (H={mean_entropy:.3f})",
                )
                moe_telemetry = stats.get("moe_telemetry", "")
                moe_full = stats.get("moe_telemetry_full", "")

                f.write("```text\n")
                f.write(f"{summary_banner}\n")
                if moe_telemetry:
                    f.write(f"{moe_telemetry}\n")
                if moe_full and moe_full != moe_telemetry:
                    f.write(f"🦀 [Rust MoE Full States (E0..E7)]: {moe_full}\n")
                f.write("```\n\n")

                f.write(
                    f"> ⚡ **Metrics:** `{tokens}` tokens (`{len(content)}` chars) | "
                    f"`{tps:.2f}` tok/s | `{elapsed:.2f}` s | TTFT: `{ttft_ms:.1f}` ms | "
                    f"Top-K: `{top_k_experts}` (`H={mean_entropy:.3f}`) | NREM 2:4 Prunes: `{gpu_24_prunes}`\n\n"
                )

                # Also append to cumulative Optimization Witness Log
                witness_entry = {
                    "timestamp": now_dt.isoformat(timespec="seconds"),
                    "session_id": self.session_id,
                    "backend": stats.get("backend", "trt-edgellm-hybrid"),
                    "tokens": tokens,
                    "chars": len(content),
                    "elapsed_s": round(elapsed, 2),
                    "tps": round(tps, 2),
                    "ttft_ms": round(ttft_ms, 1),
                    "top_k_experts": top_k_experts,
                    "mean_entropy": round(mean_entropy, 3),
                    "nrem_events": nrem_events,
                    "gpu_24_prunes": gpu_24_prunes,
                    "summary_banner": summary_banner,
                    "moe_telemetry": moe_telemetry,
                    "expert_states": stats.get("expert_states", list()),
                }
                with open(self.witness_jsonl, "a", encoding="utf-8") as wf:
                    wf.write(json.dumps(witness_entry, ensure_ascii=False) + "\n")

                if os.path.exists(self.witness_md):
                    with open(self.witness_md, "a", encoding="utf-8") as wmd:
                        wmd.write(
                            f"\n### 🔹 Run `{now_dt.strftime('%Y-%m-%d %H:%M:%S')}` (`{self.session_id}` | `{stats.get('backend', 'hybrid')}`)\n"
                            f"```text\n{summary_banner}\n{moe_telemetry}\n```\n"
                        )

            if artifacts:
                f.write("📦 **Saved Artifacts:**\n")
                for item in artifacts:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        f.write(f"- `{item[0]}` ({item[1]} bytes)\n")
                    else:
                        f.write(f"- `{item}`\n")
                f.write("\n")

            f.write("---\n")

        self.record_event("turn_complete", {
            "role": role,
            "content_length": len(content),
            "content": content,
            "stats": stats or {},
            "artifacts": artifacts or [],
            "artifacts_count": len(artifacts) if artifacts else 0,
        })

    def list_available_sessions(self) -> List[Dict[str, Any]]:
        """
        Lists all non-empty conversation session `.jsonl` files (newest first) with turn counts and first user prompt preview.
        """
        all_cands = sorted(
            [
                p for p in glob.glob(os.path.join(self.session_dir, "session_*.jsonl"))
                if os.path.basename(p) != os.path.basename(self.jsonl_file)
                and not os.path.basename(p).startswith("session_real_telemetry_")
                and not os.path.basename(p).startswith("session_rust_telemetry_")
                and os.path.getsize(p) > 20
            ],
            reverse=True,
        )
        out: List[Dict[str, Any]] = []
        for p in all_cands:
            turns = 0
            first_user = ""
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if rec.get("type") == "turn_complete":
                            data = rec.get("data") or {}
                            if data.get("role") == "user":
                                turns += 1
                                if not first_user:
                                    first_user = (data.get("content") or "").replace("\n", " ").strip()[:36]
            except Exception:
                continue
            if turns > 0:
                out.append({
                    "filename": os.path.basename(p),
                    "path": os.path.abspath(p),
                    "turns": turns,
                    "preview": first_user,
                })
        return out

    def load_resumed_history(
        self,
        session_spec: str = "latest",
        max_recent_turns: int = 4,
        max_total_chars: int = 10000,
        skip_last_turns: int = 0,
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        Loads and compresses conversation history from a previous `session_*.jsonl` file
        so `--resume-session` can seamlessly continue multi-turn tasks (e.g., NTM & RSI)!
        Supports:
          - `session_spec`: `"latest"`, `"latest-1"` (skip latest session file), or exact `session_*.jsonl` filename.
          - `skip_last_turns`: drops the last N user+assistant turn pairs inside the chosen session (to skip glitched turns!).
        """
        target_jsonl = None
        if session_spec and not session_spec.startswith("latest"):
            cand = session_spec
            if not cand.endswith(".jsonl"):
                cand = f"session_{cand}.jsonl" if not cand.startswith("session_") else f"{cand}.jsonl"
            for p in [cand, os.path.join(self.session_dir, cand)]:
                if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(self.jsonl_file):
                    target_jsonl = os.path.abspath(p)
                    break
        if not target_jsonl:
            # Strict profile isolation: only resume sessions belonging to `self.profile`
            avail = [
                x["path"]
                for x in self.list_available_sessions()
                if x["filename"].startswith(f"session_{self.profile}_")
            ]
            if avail:
                offset_idx = 0
                if session_spec and session_spec.startswith("latest-"):
                    try:
                        offset_idx = int(session_spec.split("-")[1])
                    except Exception:
                        offset_idx = 0
                offset_idx = max(0, min(len(avail) - 1, offset_idx))
                target_jsonl = avail[offset_idx]

        if not target_jsonl or not os.path.exists(target_jsonl):
            return [], None

        raw_messages: List[Dict[str, str]] = []
        detected_project_dir: Optional[str] = None
        with open(target_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") != "turn_complete":
                        continue
                    data = rec.get("data") or {}
                    role = data.get("role")
                    content = (data.get("content") or "").strip()
                    if role in ("user", "assistant") and content:
                        raw_messages.append({"role": role, "content": content})
                    for art in data.get("artifacts") or []:
                        if isinstance(art, dict) and art.get("rel_path"):
                            parts = art["rel_path"].replace("\\", "/").split("/")
                            if len(parts) >= 2 and parts[0] == "isolated_output" and parts[1].startswith("prj_"):
                                detected_project_dir = parts[1]
                except Exception:
                    continue

        if skip_last_turns > 0 and len(raw_messages) >= skip_last_turns * 2:
            raw_messages = raw_messages[: len(raw_messages) - skip_last_turns * 2]

        if not raw_messages:
            return [], None

        # Keep last `max_recent_turns * 2` messages verbatim; compact earlier turns into a digest!
        keep_count = max_recent_turns * 2
        if len(raw_messages) > keep_count:
            older_msgs = raw_messages[:-keep_count]
            recent_msgs = raw_messages[-keep_count:]
            digest_lines = [
                f"[Prior Session Context Summary ({os.path.basename(target_jsonl)})]:"
            ]
            for m in older_msgs:
                r_tag = "User" if m["role"] == "user" else "Kuro"
                snippet = m["content"].replace("\n", " ")[:260]
                digest_lines.append(f"- {r_tag}: {snippet}...")
            digest_msg = "\n".join(digest_lines)
            resumed = [
                {"role": "user", "content": digest_msg},
                {"role": "assistant", "content": "了解，我已載入先前對話的上下文摘要與架構進度。"},
            ] + recent_msgs
        else:
            resumed = list(raw_messages)

        # Enforce total character budget
        while len(resumed) > 2 and sum(len(m["content"]) for m in resumed) > max_total_chars:
            resumed[0]["content"] = resumed[0]["content"][:1500] + "\n...(earlier context compressed)..."
            if sum(len(m["content"]) for m in resumed) > max_total_chars and len(resumed) > 4:
                resumed.pop(2)
                resumed.pop(2)
            else:
                break

        print(
            f"📜 \033[1;32m[Session Context Resumed]\033[0m Loaded {len(raw_messages)} messages "
            f"(compressed to {len(resumed)} turns, {sum(len(m['content']) for m in resumed)} chars) "
            f"from \033[1;36m{os.path.basename(target_jsonl)}\033[0m"
        )
        return resumed, detected_project_dir
