import os
import json
import glob
import datetime
from typing import Dict, List, Optional, Any

class SessionLogger:
    """
    Session Archiving and Markdown Turn Logger transferred from pure-python localsleep-moe.
    Maintains:
      1. Human-readable Markdown log ({session_dir}/session_{profile}_{ts}.md)
      2. Structured JSONL event stream ({session_dir}/session_{profile}_{ts}.jsonl)
      3. Recall mechanism across historical markdown and jsonl files
    """
    def __init__(self, session_dir: str = "./sessions", profile: str = "default", model_path: str = ""):
        self.session_dir = os.path.abspath(session_dir)
        os.makedirs(self.session_dir, exist_ok=True)
        self.profile = profile
        self.model_path = model_path
        
        self.session_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{profile}_{self.session_ts}"
        
        self.md_file = os.path.join(self.session_dir, f"session_{self.session_id}.md")
        self.jsonl_file = os.path.join(self.session_dir, f"session_{self.session_id}.jsonl")
        
        self._init_logs()

    def _init_logs(self):
        with open(self.md_file, "w", encoding="utf-8") as f:
            f.write(f"# LocalSleep-MoE Agentic Session Log: {self.session_id}\n\n")
            f.write(f"- **Profile:** `{self.profile}`\n")
            f.write(f"- **Model Path:** `{self.model_path}`\n")
            f.write(f"- **Start Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
            
        self.record_event("session_start", {
            "session_id": self.session_id,
            "profile": self.profile,
            "model_path": self.model_path
        })

    def record_event(self, event_type: str, data: Any):
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_turn(
        self,
        user_prompt: str,
        assistant_reply: str,
        stats: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None
    ):
        cur_time = datetime.datetime.now().strftime("%H:%M:%S")
        with open(self.md_file, "a", encoding="utf-8") as f:
            f.write(f"### 💬 USER ({cur_time})\n\n{user_prompt}\n\n")
            f.write(f"### 🐈‍⬛ KURO / MODEL ({cur_time})\n\n{assistant_reply}\n\n")
            if stats:
                toks = stats.get("tokens", 0)
                tps = stats.get("tps", 0.0)
                elapsed = stats.get("elapsed", 0.0)
                f.write(f"> ⚡ **Tokens:** {toks} | **TPS:** {tps:.2f} tokens/s | **Duration:** {elapsed:.2f}s\n\n")
            if artifacts:
                f.write("📁 **Saved Artifacts:**\n")
                for art in artifacts:
                    f.write(f"- `{art.get('rel_path', art.get('path'))}` ({art.get('size', 0)} bytes)\n")
                f.write("\n")
            f.write("---\n\n")

        self.record_event("turn_complete", {
            "user_prompt": user_prompt,
            "reply_length": len(assistant_reply),
            "stats": stats or {},
            "artifacts_count": len(artifacts) if artifacts else 0
        })
        print(f"📝 \033[1;32m[Session Logged]\033[0m Markdown: \033[1;36m{os.path.basename(self.md_file)}\033[0m")

    def recall_recent_sessions(self, limit: int = 5) -> str:
        """
        列出最近的歷史 session 摘要列表。
        """
        md_files = sorted(glob.glob(os.path.join(self.session_dir, "session_*.md")), reverse=True)
        if not md_files:
            return "[No historical sessions found.]"
        results = [f"Found {len(md_files)} session(s). Recent {min(limit, len(md_files))}:"]
        for p in md_files[:limit]:
            fname = os.path.basename(p)
            size = os.path.getsize(p)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
            results.append(f"- `{fname}` ({size} bytes, modified: {mtime})")
        return "\n".join(results)

    def recall_search(self, query: str, max_matches: int = 3) -> str:
        """
        在所有歷史 session (Markdown) 中搜尋關鍵字並提取相關上下文片段。
        """
        md_files = sorted(glob.glob(os.path.join(self.session_dir, "session_*.md")), reverse=True)
        query_lower = query.lower()
        matches = []

        for p in md_files:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if query_lower in content.lower():
                    idx = content.lower().find(query_lower)
                    start = max(0, idx - 100)
                    end = min(len(content), idx + 200)
                    snippet = content[start:end].replace("\n", " ")
                    matches.append(f"• File: `{os.path.basename(p)}`\n  Snippet: ...{snippet}...")
                    if len(matches) >= max_matches:
                        break
            except Exception:
                continue

        if not matches:
            return f"[No historical sessions matched query: {query}]"
        return "\n\n".join(matches)
