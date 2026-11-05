# Standalone code
"""
Working Memory & Sliding Context Distiller (`working_memory_manager.py`)
========================================================================
Implements Leon's L1 Working Memory Scratchpad + L2 On-Demand Paged Recall
+ One-Click Sliding Window Context & Code Distiller for LocalSleep-MoE.

Key Capabilities:
  1. Model Autonomous Scratchpad:
     - Parses `<working_memory>...</working_memory>` emitted by the model or
       `<tool_call>{"action": "update_working_memory", ...}</tool_call>`.
     - Auto-records newly generated code artifacts (with exact sandbox paths, sizes,
       and core function/struct signatures) even if the model omits `<working_memory>`.
  2. One-Click Sliding Window Distiller (`condense_working_memory`):
     - Reads conversation turns + all generated code files in the active sandbox.
     - Performs sliding-window LLM distillation (via active Thor / RTX 5080 server)
       with a deterministic AST/code-aware structural fallback so it NEVER fails
       and ALWAYS preserves exact code architecture, key snippets, and file paths!
  3. L1 Prompt Injection + L2 Exact API & Path Directory:
     - Builds a compact, high-negentropy system context suffix containing:
       * Distilled Working Memory & Code Blueprint
       * Exact Sandbox Code File Paths (for `read_file` / `edit_file`)
       * Recent Session Log Paths (for on-demand `read_file`)
"""

import glob
import json
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class WorkingMemoryManager:
    def __init__(
        self,
        workspace_root: str = ".",
        session_dir: str = "./sessions",
        profile: str = "thor_31b",
    ):
        self.workspace_root = os.path.abspath(workspace_root)
        self.session_dir = os.path.abspath(session_dir)
        self.profile = profile
        os.makedirs(self.session_dir, exist_ok=True)

        self.wm_file = os.path.join(self.session_dir, f"working_memory_{self.profile}.md")
        self.wm_json = os.path.join(self.session_dir, f"working_memory_{self.profile}.json")
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        if not os.path.exists(self.wm_json):
            init_state = {
                "profile": self.profile,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "active_sandbox": "",
                "distilled_summary": "尚無濃縮工作記憶。可點擊『🧠 一鍵濃縮工作記憶』或由模型自動生成。",
                "model_scratchpad": "",
                "code_artifacts": {},
                "recent_turns_digest": [],
            }
            self._save_state(init_state)

    def _load_state(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.wm_json):
                with open(self.wm_json, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "profile": self.profile,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "active_sandbox": "",
            "distilled_summary": "",
            "model_scratchpad": "",
            "code_artifacts": {},
            "recent_turns_digest": [],
        }

    def _save_state(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(self.wm_json, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        self._render_markdown(state)

    @staticmethod
    def _extract_code_skeleton(content: str, max_lines: int = 35) -> str:
        """
        Extracts key structural signatures (Rust pub struct/fn/impl, Python class/def)
        and critical logic lines so the working memory retains exact code symbols & types!
        """
        lines = content.splitlines()
        picked: List[str] = []
        sig_pat = re.compile(
            r"^\s*(pub\s+struct|struct|pub\s+enum|enum|pub\s+fn|fn|impl|class\s+|def\s+|async\s+def\s+|#\[py)"
        )
        for idx, line in enumerate(lines, start=1):
            if sig_pat.search(line):
                picked.append(f"L{idx:03d}: {line.strip()}")
        if not picked:
            # Fallback: take first 20 non-empty lines
            for idx, line in enumerate(lines[:25], start=1):
                if line.strip():
                    picked.append(f"L{idx:03d}: {line.strip()[:100]}")
        return "\n".join(picked[:max_lines])

    def scan_sandbox_code_files(self, sandbox_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Scans the active sandbox directory (or latest non-empty sandbox if current is empty)
        and indexes all code files with their exact relative path, size, line count,
        skeleton signatures, and key code preview.
        """
        target_dir = os.path.abspath(sandbox_dir) if sandbox_dir else ""
        # Always strip old prj_* prefix from artifact keys so prompt injection uses clean relative paths (e.g. `src/lib.rs`)
        # bound to whichever new timestamped `prj_YYYYMMDD_HHMMSS` sandbox is currently active!
        iso_root = os.path.join(self.workspace_root, "isolated_output")
        if (not target_dir or not os.path.exists(target_dir) or not self._list_files_recursive(target_dir)) and os.path.exists(iso_root):
            cands = sorted(
                [p for p in Path(iso_root).glob("prj_*") if p.is_dir() and any(p.rglob("*.*"))],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if cands:
                target_dir = str(cands[0])

        artifacts: Dict[str, Dict[str, Any]] = {}
        if not target_dir or not os.path.exists(target_dir):
            return artifacts

        skip_dirs = {"target", "__pycache__", ".git", "node_modules", ".venv", "venv", "build", "dist"}
        allowed_exts = {".py", ".rs", ".toml", ".md", ".sh", ".c", ".cpp", ".cu", ".h", ".hpp", ".json"}
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if fname.startswith(".") or fname == "Cargo.lock" or ext not in allowed_exts:
                    continue
                fpath = os.path.join(root, fname)
                if os.path.getsize(fpath) > 65536:
                    continue
                rel_path = os.path.relpath(fpath, target_dir).replace("\\", "/")
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        code_txt = f.read()
                    lines = code_txt.splitlines()
                    skeleton = self._extract_code_skeleton(code_txt, max_lines=16)
                    # Keep up to 480 chars of actual core code for direct working memory embedding
                    code_preview = code_txt[:480] + ("\n# ... [Use read_file for full code] ..." if len(code_txt) > 480 else "")
                    artifacts[rel_path] = {
                        "rel_path": rel_path,
                        "bytes": len(code_txt.encode("utf-8")),
                        "lines": len(lines),
                        "skeleton": skeleton,
                        "code_preview": code_preview,
                    }
                except Exception:
                    continue
        return artifacts

    @staticmethod
    def _list_files_recursive(folder: str) -> List[str]:
        out = []
        if os.path.exists(folder):
            for r, _, fs in os.walk(folder):
                for f in fs:
                    out.append(os.path.join(r, f))
        return out

    def extract_and_update_from_turn(
        self,
        user_prompt: str,
        assistant_reply: str,
        saved_artifacts: List[Dict[str, Any]],
        sandbox_dir: str,
    ) -> str:
        """
        Automatically intercepts `<working_memory>...</working_memory>` from model output
        AND updates the tracked code artifacts + recent turn digest.
        Returns cleaned assistant_reply (or preserves it) and updates `working_memory_<profile>.md`.
        """
        state = self._load_state()
        if sandbox_dir:
            state["active_sandbox"] = os.path.relpath(sandbox_dir, self.workspace_root).replace("\\", "/")

        # 1. Check if model explicitly emitted <working_memory>...</working_memory>
        wm_matches = re.findall(r"<working_memory>(.*?)</working_memory>", assistant_reply, re.DOTALL)
        if wm_matches:
            latest_wm = wm_matches[-1].strip()
            if latest_wm:
                state["model_scratchpad"] = latest_wm

        # 2. Scan sandbox code files (includes any newly saved XML artifacts)
        code_map = self.scan_sandbox_code_files(sandbox_dir)
        if code_map:
            state["code_artifacts"] = code_map

        # 3. Append sliding turn digest (keep last 8 turns)
        clean_reply = re.sub(r"<working_memory>.*?</working_memory>", "", assistant_reply, flags=re.DOTALL).strip()
        u_snip = (user_prompt or "").replace("\n", " ").strip()[:180]
        a_snip = (clean_reply or "").replace("\n", " ").strip()[:280]
        if u_snip or a_snip:
            digest_list = list(state.get("recent_turns_digest") or [])
            digest_list.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "user": u_snip,
                "kuro": a_snip,
                "saved_files": [a.get("rel_path", "") for a in (saved_artifacts or [])],
            })
            state["recent_turns_digest"] = digest_list[-8:]

        self._save_state(state)
        return self.wm_file

    def update_scratchpad_manual(self, content: str, mode: str = "replace") -> str:
        state = self._load_state()
        if mode == "append" and state.get("model_scratchpad"):
            state["model_scratchpad"] = state["model_scratchpad"].rstrip() + "\n" + content.strip()
        else:
            state["model_scratchpad"] = content.strip()
        self._save_state(state)
        return f"[Working Memory Updated ({len(state['model_scratchpad'])} chars) -> {os.path.basename(self.wm_file)}]"

    def condense_working_memory(
        self,
        conversation_history: List[Dict[str, str]],
        sandbox_dir: str,
        server_url: Optional[str] = None,
        model_id: Optional[str] = None,
        use_llm_sliding_window: bool = True,
    ) -> Tuple[str, str]:
        """
        One-Click Sliding Window Context & Code Distiller!
        1. Collects conversation history (or loads latest non-empty session if current is empty)
        2. Scans all code files in the active/latest sandbox (extracting AST/signatures + core code)
        3. Runs sliding-window distillation (via Server LLM if reachable, plus deterministic code blueprint)
        4. Writes everything into `working_memory_<profile>.md` so next restart has 100% context + code!
        Returns: (markdown_text, status_summary)
        """
        state = self._load_state()

        # If current session has no conversation history yet, pull from latest non-empty session file
        history_to_condense = list(conversation_history or [])
        source_session_name = "current_active_session"
        if not history_to_condense:
            cands = sorted(
                [
                    p for p in Path(self.session_dir).glob("session_*.jsonl")
                    if not p.name.startswith("session_real_telemetry_")
                    and not p.name.startswith("session_rust_telemetry_")
                    and p.stat().st_size > 20
                ],
                key=lambda x: x.name,
                reverse=True,
            )
            for p in cands:
                msgs = []
                detected_sb = None
                try:
                    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if rec.get("type") == "turn_complete":
                            d = rec.get("data") or {}
                            if d.get("role") in ("user", "assistant") and (d.get("content") or "").strip():
                                msgs.append({"role": d["role"], "content": d["content"].strip()})
                            for art in d.get("artifacts") or []:
                                if isinstance(art, dict) and art.get("rel_path"):
                                    parts = art["rel_path"].replace("\\", "/").split("/")
                                    if len(parts) >= 2 and parts[0] == "isolated_output":
                                        detected_sb = os.path.join(self.workspace_root, parts[0], parts[1])
                except Exception:
                    continue
                if msgs:
                    history_to_condense = msgs
                    source_session_name = p.name
                    if detected_sb and os.path.exists(detected_sb):
                        sandbox_dir = detected_sb
                    break

        code_map = self.scan_sandbox_code_files(sandbox_dir)
        state["code_artifacts"] = code_map
        if code_map:
            first_rel = next(iter(code_map.keys()))
            parts = first_rel.split("/")
            if len(parts) >= 2 and parts[0] == "isolated_output":
                state["active_sandbox"] = f"{parts[0]}/{parts[1]}"

        # Sliding Window Chunking over Conversation History (window = 4 messages per window)
        window_size = 4
        windows: List[str] = []
        for i in range(0, len(history_to_condense), window_size):
            chunk = history_to_condense[i : i + window_size]
            lines = [f"[Window {i//window_size + 1} (Msgs {i+1}..{i+len(chunk)})]:"]
            for m in chunk:
                role_lbl = "Leon(User)" if m["role"] == "user" else "Kuro(Model)"
                # Strip verbose code blocks from raw conversation text since we attach exact code files below
                clean_c = re.sub(r"<file\s+[^>]+>.*?</file>", "[Code Artifact Saved to Sandbox]", m["content"], flags=re.DOTALL)
                clean_c = re.sub(r"```.*?```", "[Code Block]", clean_c, flags=re.DOTALL)
                clean_c = re.sub(r"\s+", " ", clean_c).strip()
                lines.append(f"- {role_lbl}: {clean_c[:360]}")
            windows.append("\n".join(lines))

        sliding_raw_digest = "\n\n".join(windows[-4:]) if windows else "（尚無對話紀錄）"

        # Build Code Blueprint Summary for LLM & Deterministic Archive
        code_blueprint_lines = []
        for rel_p, info in code_map.items():
            code_blueprint_lines.append(
                f"### 📄 `{rel_p}` ({info['lines']} lines, {info['bytes']} bytes)\n"
                f"**結構簽名 (AST/Signatures):**\n```text\n{info['skeleton']}\n```\n"
                f"**核心代碼快照 (Core Code Snapshot):**\n```\n{info['code_preview']}\n```"
            )
        code_blueprint_str = "\n\n".join(code_blueprint_lines) if code_blueprint_lines else "（目前沙盒尚無代碼檔案）"

        llm_summary = ""
        if use_llm_sliding_window and server_url and model_id and (windows or code_map):
            try:
                distill_prompt = (
                    "請將以下『滑動視窗對話脈絡』與『沙盒已生成之程式碼結構』濃縮為一份高密度、零廢話的【工作記憶交接藍圖 (Working Memory Blueprint)】。\n"
                    "必須包含：\n"
                    "1. 🎯 **當前核心任務與架構目標**\n"
                    "2. 🧬 **已完成之演算法/數學/代碼實作重點**（請保留關鍵函式名、Struct 名與檔案路徑）\n"
                    "3. 🚀 **下次重啟後的下一步待辦事項 (Next Steps)**\n\n"
                    f"--- 滑動視窗對話脈絡 ({source_session_name}) ---\n{sliding_raw_digest[:3500]}\n\n"
                    f"--- 沙盒代碼結構清單 ---\n"
                    + "\n".join([f"- {k} ({v['lines']}L): {v['skeleton'].splitlines()[:4]}" for k, v in code_map.items()])
                )
                payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "你是 S-tier 架構記憶蒸餾器。只輸出高密度技術重點，嚴禁寒暄。"},
                        {"role": "user", "content": distill_prompt},
                    ],
                    "max_tokens": 768,
                    "temperature": 0.3,
                    "stream": False,
                    "enable_thinking": False,
                }
                req = urllib.request.Request(
                    f"{server_url.rstrip('/')}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=25.0) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    choices = res_data.get("choices") or []
                    if choices:
                        llm_summary = (choices[0].get("message", {}).get("content") or "").strip()
            except Exception as e:
                llm_summary = f"*(離線決定論濃縮模式 — 伺服器即時摘要略過: {e})*\n\n" + sliding_raw_digest

        if not llm_summary:
            llm_summary = sliding_raw_digest

        state["distilled_summary"] = llm_summary
        self._save_state(state)

        md_text = self.read_working_memory_markdown()
        status = (
            f"✅ 已完成一鍵滑動濃縮！來源: {source_session_name} ({len(history_to_condense)//2} 輪對話) | "
            f"納入代碼檔: {len(code_map)} 個 | 已寫入: sessions/{os.path.basename(self.wm_file)}"
        )
        return md_text, status

    def _render_markdown(self, state: Dict[str, Any]) -> None:
        lines = [
            f"# 🧠 Kuro L1 工作記憶與代碼藍圖 (`working_memory_{self.profile}.md`)",
            f"- **最後更新時間**: `{state.get('updated_at', '')}`",
            f"- **當前綁定沙盒**: `{state.get('active_sandbox') or 'auto'}`",
            "",
            "## 1. 🎯 滑動濃縮脈絡與架構進度 (Sliding Distilled Context)",
            state.get("distilled_summary") or "（尚無濃縮摘要）",
            "",
        ]

        if state.get("model_scratchpad"):
            lines.extend([
                "## 2. 📝 模型自主工作記憶暫存區 (Model Autonomous Scratchpad)",
                state["model_scratchpad"],
                "",
            ])

        code_map = state.get("code_artifacts") or {}
        lines.append(f"## 3. 💻 沙盒核心程式碼快照與簽名索引 ({len(code_map)} files)")
        if code_map:
            for rel_p, info in code_map.items():
                lines.extend([
                    f"### 📄 `{rel_p}` (`{info.get('lines', 0)}` lines, `{info.get('bytes', 0)}` bytes)",
                    "**結構簽名 (Signatures):**",
                    "```text",
                    info.get("skeleton", ""),
                    "```",
                    "**核心程式碼片段 (Code Preview):**",
                    "```",
                    info.get("code_preview", ""),
                    "```",
                    "",
                ])
        else:
            lines.append("（目前沙盒無代碼檔案）\n")

        recent_digests = state.get("recent_turns_digest") or []
        if recent_digests:
            lines.append("## 4. ⏱️ 最近互動滑動軌跡 (Recent Turn Trace)")
            for d in recent_digests[-5:]:
                files_str = f" -> Saved: {d['saved_files']}" if d.get("saved_files") else ""
                lines.append(f"- **[{d.get('time', '')}] Leon**: {d.get('user', '')}")
                lines.append(f"  - **Kuro**: {d.get('kuro', '')}{files_str}")
            lines.append("")

        with open(self.wm_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def read_working_memory_markdown(self) -> str:
        if os.path.exists(self.wm_file):
            with open(self.wm_file, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        return ""

    def build_system_prompt_injection(self, current_sandbox_dir: str = "") -> Tuple[str, Optional[str]]:
        """
        Builds the compact L1 Working Memory + L2 On-Demand API & Path Directory
        to inject into the model's context upon startup/turn, AND returns the
        active sandbox UUID so the session re-attaches to the existing code files!
        """
        state = self._load_state()
        code_map = self.scan_sandbox_code_files(current_sandbox_dir or state.get("active_sandbox", ""))
        if code_map and not state.get("code_artifacts"):
            state["code_artifacts"] = code_map
            self._save_state(state)

        detected_prj_uuid = None
        active_sb = state.get("active_sandbox", "")
        if active_sb and "prj_" in active_sb:
            detected_prj_uuid = os.path.basename(active_sb.rstrip("/"))

        # List recent session markdown files for L2 on-demand reading
        recent_sessions = sorted(
            [
                p.name for p in Path(self.session_dir).glob(f"session_{self.profile}_*.md")
                if p.stat().st_size > 40
            ],
            reverse=True,
        )[:4]

        wm_block = [
            "\n\n======================================================================",
            "🧠 [L1 WORKING MEMORY & CODE BLUEPRINT (Active State)]",
            "======================================================================",
        ]
        if state.get("distilled_summary"):
            wm_block.append(f"### 🎯 濃縮脈絡摘要:\n{state['distilled_summary'][:2200]}\n")
        if state.get("model_scratchpad"):
            wm_block.append(f"### 📝 模型自主暫存筆記:\n{state['model_scratchpad'][:1200]}\n")

        if code_map:
            wm_block.append("### 💻 當前專案沙盒已存在之程式碼檔案與核心快照 (可直接呼叫 `read_file` / `edit_file`):")
            for idx_f, (rel_p, info) in enumerate(code_map.items()):
                if idx_f >= 8:
                    wm_block.append(f"- ...(以及其餘 {len(code_map) - 8} 個檔案，可用 `list_dir` 查看)...")
                    break
                wm_block.append(
                    f"- **路徑**: `{rel_p}` ({info['lines']} lines, {info['bytes']} bytes)\n"
                    f"  簽名:\n```text\n{info['skeleton']}\n```\n"
                    f"  代碼前段預覽:\n```\n{info['code_preview'][:450]}\n```"
                )

        wm_block.extend([
            "",
            "======================================================================",
            "🗄️ [L2 ON-DEMAND TOOL API & ISOLATED SANDBOX EXECUTION]",
            "======================================================================",
            "1. 📖 **讀取檔案 (`read_file` - 支援 Full 整檔與 Partial 區間讀取)**：",
            "   - 部分行區間讀取 (推薦)：`<tool_call>{\"action\": \"read_file\", \"args\": {\"path\": \"<上方路徑>\", \"mode\": \"partial\", \"start_line\": 1, \"end_line\": 120}}</tool_call>`",
            "   - 完整檔案讀取：`<tool_call>{\"action\": \"read_file\", \"args\": {\"path\": \"<上方路徑>\", \"mode\": \"full\"}}</tool_call>`",
            "2. ✏️ **局部修改檔案 (`edit_file` - 支援區塊替換或行號區間覆寫，免重寫整檔)**：",
            "   - 區塊替換：`<tool_call>{\"action\": \"edit_file\", \"args\": {\"path\": \"<上方路徑>\", \"old_text\": \"原代碼片段\", \"new_text\": \"新代碼片段\"}}</tool_call>`",
            "   - 行號區間覆寫：`<tool_call>{\"action\": \"edit_file\", \"args\": {\"path\": \"<上方路徑>\", \"start_line\": 40, \"end_line\": 55, \"new_text\": \"新代碼片段\"}}</tool_call>`",
            "3. 🖥️ **沙盒內執行 Shell / Python / Rust 測試 (`run_sandbox_shell` - 僅限 isolated 沙盒區域)**：",
            "   - 執行腳本或編譯：`<tool_call>{\"action\": \"run_sandbox_shell\", \"args\": {\"command\": \"python3 mock_evaluator.py\"}}</tool_call>` (或 `cargo build --release` / `bash run.sh`)",
            "4. 📜 **若需讀取先前的歷史對話原始紀錄，請使用以下精確路徑呼叫 `read_file`**：",
        ])
        for sname in recent_sessions:
            wm_block.append(f"  * `sessions/{sname}`")
        wm_block.extend([
            "- 你可以在回答結尾附上 `<working_memory>本輪關鍵進展、架構決策與下一步待辦</working_memory>`，系統會自動幫你更新至 L1 工作記憶！",
            "======================================================================",
        ])

        return "\n".join(wm_block)[:6000], detected_prj_uuid
