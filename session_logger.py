import os
import json
import time
from datetime import datetime
from typing import Optional


class SessionLogger:
    """
    通用型 Session Archiving and Markdown Turn Logger。
    同時相容新版 `append_turn(role, content, ...)` 與舊版 `append_turn(user_prompt=..., assistant_reply=...)` 呼叫介面。
    """
    def __init__(self, session_dir="./sessions", profile="default", model_path=""):
        self.session_dir = os.path.abspath(session_dir)
        os.makedirs(self.session_dir, exist_ok=True)
        self.profile = profile
        self.model_path = model_path

        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{profile}_{session_ts}"
        self.md_file = os.path.join(self.session_dir, f"session_{self.session_id}.md")
        self.jsonl_file = os.path.join(self.session_dir, f"session_{self.session_id}.jsonl")

        self._init_logs()

    def _init_logs(self):
        with open(self.md_file, "w", encoding="utf-8") as f:
            f.write(f"# Agentic Session Log: {self.session_id}\n")
            f.write(f"- **Profile:** `{self.profile}`\n")
            f.write(f"- **Model:** `{self.model_path}`\n")
            f.write(f"- **Start Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")

    def record_event(self, event_type: str, data: dict):
        """JSONL 微秒級遙測紀錄 (供後續除錯或強化學習回放)"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_turn(
        self,
        role: Optional[str] = None,
        content: Optional[str] = None,
        stats: Optional[dict] = None,
        artifacts: Optional[list] = None,
        user_prompt: Optional[str] = None,
        assistant_reply: Optional[str] = None,
    ):
        """
        支援任意 Role (user, assistant, tool, system) 的 Markdown 追加，
        並向下相容 `append_turn(user_prompt=..., assistant_reply=..., stats=..., artifacts=...)`。
        """
        if user_prompt is not None or assistant_reply is not None:
            if user_prompt:
                self.append_turn(role="user", content=user_prompt)
            if assistant_reply is not None:
                self.append_turn(role="assistant", content=assistant_reply, stats=stats, artifacts=artifacts)
            return

        role = role or "assistant"
        content = content or ""
        cur_time = datetime.now().strftime("%H:%M:%S")

        role_icons = {
            "user": "💬 **USER**",
            "assistant": "🧠 **KURO / MODEL**",
            "tool": "🛠️ **TOOL EXECUTION**",
            "system": "⚙️ **SYSTEM**"
        }
        header = role_icons.get(role, f"🔹 **{role.upper()}**")

        with open(self.md_file, "a", encoding="utf-8") as f:
            f.write(f"\n### {header} ({cur_time})\n\n")
            f.write(f"{content}\n\n")

            if stats:
                tokens = stats.get("tokens", 0)
                tps = stats.get("tps", 0.0)
                elapsed = stats.get("elapsed", 0.0)
                f.write(f"> ⚡ **Metrics:** `{tokens}` tokens | `{tps:.2f}` tok/s | `{elapsed:.2f}` s\n\n")

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
            "stats": stats or {},
            "artifacts_count": len(artifacts) if artifacts else 0
        })
