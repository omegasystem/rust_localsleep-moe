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
      1. read_file(path, offset=0, limit=2000)
      2. write_file(path, content) -> Isolated sandbox output by default
      3. list_dir(path)
      4. web_search(query, max_results=5) (via DuckDuckGo / SearXNG fallback)
      5. extract_and_save_artifacts(text) -> Multi-line XML artifact auto-persister (<file path="...">...)
    """
    def __init__(self, workspace_root=".", isolated_output_dir="./isolated_output"):
        self.workspace_root = os.path.abspath(workspace_root)
        self.isolated_output_dir = os.path.abspath(os.path.join(self.workspace_root, isolated_output_dir))
        os.makedirs(self.isolated_output_dir, exist_ok=True)

    def _resolve_safe_read_path(self, path: str) -> str:
        clean_path = path.strip().lstrip("/")
        full_path = os.path.abspath(os.path.join(self.workspace_root, clean_path))
        return full_path

    def _resolve_safe_write_path(self, path: str) -> str:
        clean_filename = os.path.basename(path.strip()) if ".." in path else path.strip().lstrip("/\\")
        full_path = os.path.abspath(os.path.join(self.isolated_output_dir, clean_filename))
        if not full_path.startswith(self.isolated_output_dir):
            full_path = os.path.join(self.isolated_output_dir, os.path.basename(path.strip()))
        return full_path

    def read_file(self, path: str, offset: int = 1, limit: int = 200) -> str:
        try:
            target = self._resolve_safe_read_path(path)
            if not os.path.exists(target):
                return f"[Error: File not found: {path}]"
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            start = max(0, offset - 1)
            end = start + limit
            snippet = "".join(lines[start:end])
            return snippet if snippet else "[File empty or offset beyond EOF]"
        except Exception as e:
            return f"[Error reading {path}: {e}]"

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
        try:
            target = self._resolve_safe_read_path(path)
            if not os.path.exists(target):
                return f"[Error: Directory not found: {path}]"
            entries = os.listdir(target)
            return json.dumps(entries[:50], ensure_ascii=False)
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
            return self.read_file(args.get("path", ""), args.get("offset", 1), args.get("limit", 200))
        elif action == "write_file":
            return self.write_file(args.get("path", ""), args.get("content", ""))
        elif action == "list_dir":
            return self.list_dir(args.get("path", "."))
        elif action == "web_search":
            return self.web_search(args.get("query", ""))
        return f"[Unknown Tool: {action}]"
