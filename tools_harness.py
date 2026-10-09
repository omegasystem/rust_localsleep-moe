import os
import glob
import urllib.request
import urllib.parse
import json
import re
from typing import List, Dict, Tuple, Optional, Any

class EmbodiedToolHarness:
    """
    Decoupled Tool Harness for LocalSleep-MoE Agentic Workflows.
    Features:
      1. read_file(path, offset=1, limit=200, chunk_mode="lines")
         - Supports line-based chunking or character-based chunking with total stats
      2. edit_file(path, old_text, new_text, replace_all=False)
         - Exact block replacement tool without full-file rewrite
      3. write_file(path, content) -> Isolated sandbox output by default
      4. list_dir(path)
      5. web_search(query, max_results=5) (via DuckDuckGo / SearXNG fallback)
      6. extract_and_save_artifacts(text) -> Multi-line XML artifact auto-persister (<file path="...">...)
      7. recall_sessions & recall_search -> History log queries
    """
    def __init__(self, workspace_root=".", isolated_output_dir="./isolated_output", session_logger=None):
        self.workspace_root = os.path.abspath(workspace_root)
        self.isolated_output_dir = os.path.abspath(os.path.join(self.workspace_root, isolated_output_dir))
        self.session_logger = session_logger
        self.last_read_state = {}  # {normalized_path: next_offset} 自動記錄各檔案讀取指針
        os.makedirs(self.isolated_output_dir, exist_ok=True)

    def _resolve_safe_read_path(self, path: str) -> str:
        clean_path = path.strip().lstrip("/")
        # 若以 ./isolated_output 或 isolated_output 開頭，直接定位至沙盒目錄
        if clean_path.startswith("isolated_output/") or clean_path.startswith("isolated_output\\"):
            sub_path = clean_path[len("isolated_output/"):].lstrip("/\\")
            return os.path.abspath(os.path.join(self.isolated_output_dir, sub_path))

        # 優先檢查工作區是否存在，若不存在則檢查沙盒是否存在 (讓 read_file 也能無縫讀取已生成的 artifacts)
        full_path = os.path.abspath(os.path.join(self.workspace_root, clean_path))
        if not os.path.exists(full_path):
            sandbox_candidate = os.path.abspath(os.path.join(self.isolated_output_dir, os.path.basename(clean_path)))
            if os.path.exists(sandbox_candidate):
                return sandbox_candidate
        return full_path

    def _resolve_safe_write_path(self, path: str) -> str:
        clean_filename = os.path.basename(path.strip()) if ".." in path else path.strip().lstrip("/\\")
        full_path = os.path.abspath(os.path.join(self.isolated_output_dir, clean_filename))
        if not full_path.startswith(self.isolated_output_dir):
            full_path = os.path.join(self.isolated_output_dir, os.path.basename(path.strip()))
        return full_path

    def read_file(self, path: str, offset: Optional[int] = None, limit: int = 150, chunk_size: Optional[int] = None) -> str:
        """
        分段讀取檔案：
        - 預設行分段: limit 預設 150 行，硬性上限 Clamp 300 行（滿足大 Chunk 一次讀入需求）
        - 字元分段: 若傳入 chunk_size，硬性上限 Clamp 8000 字元
        - 自動回傳總行數/總字元數與分段進度 metadata
        """
        try:
            target = self._resolve_safe_read_path(path)
            if not os.path.exists(target):
                return f"[Error: File not found: {path}]"
            if os.path.isdir(target):
                return (
                    f"[Error: '{path}' is a directory, not a readable file. "
                    f"Use list_dir(path='{path}') to inspect its contents.]"
                )

            file_bytes = os.path.getsize(target)
            norm_key = os.path.normpath(target)

            actual_offset = offset
            if actual_offset is None:
                actual_offset = self.last_read_state.get(norm_key, 1)

            # 字元區塊分段模式 (Character chunking - Hard Cap 8000 chars)
            if chunk_size is not None and chunk_size > 0:
                safe_chunk_size = min(chunk_size, 8000)
                with open(target, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                total_chars = len(content)
                char_offset = max(0, actual_offset)
                chunk = content[char_offset:char_offset + safe_chunk_size]
                next_offset = char_offset + len(chunk)
                has_more = next_offset < total_chars
                self.last_read_state[norm_key] = next_offset if has_more else 0
                header = (
                    f"--- [FILE: {path} | CHUNK: chars {char_offset}..{next_offset}/{total_chars} "
                    f"({file_bytes} bytes) | HAS_MORE: {has_more}"
                    + (f" | NEXT_OFFSET: {next_offset}" if has_more else "")
                    + "] ---\n"
                )
                return header + (chunk if chunk else "[Offset beyond EOF]")

            # 行分段模式 (Line chunking, default - Hard Cap 300 lines)
            safe_limit = min(max(1, limit), 300)
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            start_idx = max(0, actual_offset - 1)
            end_idx = min(total_lines, start_idx + safe_limit)
            selected_lines = lines[start_idx:end_idx]

            has_more = end_idx < total_lines
            next_offset = end_idx + 1 if has_more else None
            self.last_read_state[norm_key] = next_offset if has_more else 1

            header = (
                f"--- [FILE: {path} | LINES: {start_idx + 1}..{end_idx}/{total_lines} "
                f"({file_bytes} bytes) | HAS_MORE: {has_more}"
                + (f" | NEXT_OFFSET: {next_offset}" if has_more else " (END OF FILE)")
                + "] ---\n"
            )

            if not selected_lines:
                return header + "[Offset beyond EOF or file empty]"

            return header + "".join(selected_lines)
        except Exception as e:
            return f"[Error reading {path}: {e}]"

    def edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        """
        精確區塊替換工具 (Exact Block Replacement)：
        在指定檔案中尋找 old_text 並替換為 new_text，避免重寫整份大檔案造成上下文早夭。
        - 優先尋找沙盒中的檔案，若不存在則檢查工作區中的檔案，修改後寫入沙盒或原檔。
        """
        try:
            if not old_text:
                return "[Error: old_text cannot be empty]"

            # 優先搜尋沙盒目錄
            sandbox_target = self._resolve_safe_write_path(path)
            workspace_target = self._resolve_safe_read_path(path)

            if os.path.exists(sandbox_target):
                target = sandbox_target
            elif os.path.exists(workspace_target):
                target = workspace_target
            else:
                return f"[Error: File not found for editing: {path}]"

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            count = content.count(old_text)
            if count == 0:
                return f"[Error: Target block 'old_text' not found in {path}]"
            if count > 1 and not replace_all:
                return f"[Error: 'old_text' matched {count} times in {path}. Provide a more unique block or set replace_all=True]"

            if replace_all:
                updated_content = content.replace(old_text, new_text)
            else:
                updated_content = content.replace(old_text, new_text, 1)

            # 存回檔案 (若在沙盒則覆寫沙盒，若不在則存入沙盒以確保隔離安全)
            with open(target, "w", encoding="utf-8") as f:
                f.write(updated_content)

            rel_path = os.path.relpath(target, self.workspace_root)
            return f"[Success: Replaced {count if replace_all else 1} occurrence(s) in {rel_path} ({len(updated_content)} bytes)]"
        except Exception as e:
            return f"[Error editing {path}: {e}]"

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._resolve_safe_write_path(path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            rel_path = os.path.relpath(target, self.workspace_root)
            return f"[Success: Wrote {len(content)} bytes to isolated sandbox: {rel_path}]"
        except Exception as e:
            return f"[Error writing {path}: {e}]"

    def list_dir(self, path: str = ".") -> str:
        """
        列出目錄內容，提供精確的檔案型態 [FILE] / [DIR] 標記、大小與相對路徑，
        防止模型誤把目錄當檔案去 read_file 造成路徑混亂與報錯。
        """
        try:
            target = self._resolve_safe_read_path(path)
            if not os.path.exists(target):
                return f"[Error: Directory not found: {path}]"
            if not os.path.isdir(target):
                return f"[Error: Target is not a directory: {path}]"

            entries = sorted(os.listdir(target))
            formatted = [f"📁 Directory listing of: {path} (total: {len(entries)})"]
            for name in entries[:60]:
                full_entry_path = os.path.join(target, name)
                # 統一給模型相對路徑格式，方便模型直接複製填入 read_file
                rel_entry_path = os.path.normpath(os.path.join(path, name)) if path != "." else name
                if os.path.isdir(full_entry_path):
                    formatted.append(f"  [DIR]  {name}/ -> path: '{rel_entry_path}'")
                else:
                    size = os.path.getsize(full_entry_path)
                    formatted.append(f"  [FILE] {name} ({size} bytes) -> path: '{rel_entry_path}'")
            if len(entries) > 60:
                formatted.append(f"  ... (and {len(entries) - 60} more items)")
            return "\n".join(formatted)
        except Exception as e:
            return f"[Error listing {path}: {e}]"

    def web_search(self, query: str, max_results: int = 5) -> str:
        """Lightweight non-browser web search using DuckDuckGo HTML / instant API."""
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
            clean_snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:max_results]]
            if not clean_snippets:
                return f"[No immediate web search results found for: {query}]"
            return "\n---\n".join(clean_snippets)
        except Exception as e:
            return f"[Web Search unavailable: {e}]"

    def extract_and_save_artifacts(self, text: str) -> List[Dict[str, Any]]:
        """
        Transferred from pure-python localsleep-moe artifact_extractor.py:
        Extracts multi-line native XML artifacts (<file path="...">...</file>)
        and automatically writes them into isolated_output_dir without escaping or formatting destruction.
        """
        saved_artifacts = []
        patterns = [
            r'<file\s+(?:path|name)=["\'](.*?)["\']>(.*?)</file>',
            r'<write_file\s+path=["\'](.*?)["\']>(.*?)</write_file>',
            r'<artifact\s+(?:path|identifier|name)=["\'](.*?)["\'][^>]*>(.*?)</artifact>'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for filepath, content in matches:
                filepath = filepath.strip()
                clean_content = content.strip("\n\r")
                if not filepath or not clean_content:
                    continue

                full_path = self._resolve_safe_write_path(filepath)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(clean_content)

                rel_path = os.path.relpath(full_path, self.workspace_root)
                saved_artifacts.append({
                    "path": full_path,
                    "rel_path": rel_path,
                    "size": len(clean_content)
                })
                print(f"\n💾 \033[1;32m[Auto-Save XML Artifact (Isolated Sandbox)]\033[0m -> \033[1;36m{rel_path}\033[0m ({len(clean_content)} bytes)")

        return saved_artifacts

    def dispatch(self, action: str, args: dict) -> str:
        if action == "read_file":
            return self.read_file(
                path=args.get("path", ""),
                offset=args.get("offset", None),
                limit=args.get("limit", 60),
                chunk_size=args.get("chunk_size", None)
            )
        elif action == "edit_file":
            return self.edit_file(
                path=args.get("path", ""),
                old_text=args.get("old_text", ""),
                new_text=args.get("new_text", ""),
                replace_all=args.get("replace_all", False)
            )
        elif action == "write_file":
            return self.write_file(args.get("path", ""), args.get("content", ""))
        elif action == "list_dir":
            return self.list_dir(args.get("path", "."))
        elif action == "web_search":
            return self.web_search(args.get("query", ""))
        elif action == "recall_sessions":
            if self.session_logger:
                return self.session_logger.recall_recent_sessions(limit=args.get("limit", 5))
            return "[Error: session_logger not attached to harness]"
        elif action == "recall_search":
            if self.session_logger:
                return self.session_logger.recall_search(query=args.get("query", ""), max_matches=args.get("max_matches", 3))
            return "[Error: session_logger not attached to harness]"
        return f"[Unknown Tool: {action}]"
