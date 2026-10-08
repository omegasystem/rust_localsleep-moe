import os
import glob
import urllib.request
import urllib.parse
import json
import re

class EmbodiedToolHarness:
    """
    Decoupled Tool Harness for LocalSleep-MoE Agentic Workflows.
    Features:
      1. read_file(path, offset=0, limit=2000)
      2. write_file(path, content)
      3. list_dir(path)
      4. web_search(query, max_results=5) (via DuckDuckGo / SearXNG fallback)
    """
    def __init__(self, workspace_root="."):
        self.workspace_root = os.path.abspath(workspace_root)

    def _resolve_safe_path(self, path: str) -> str:
        full_path = os.path.abspath(os.path.join(self.workspace_root, path))
        return full_path

    def read_file(self, path: str, offset: int = 1, limit: int = 200) -> str:
        try:
            target = self._resolve_safe_path(path)
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
            target = self._resolve_safe_path(path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[Success: Wrote {len(content)} bytes to {path}]"
        except Exception as e:
            return f"[Error writing {path}: {e}]"

    def list_dir(self, path: str = ".") -> str:
        try:
            target = self._resolve_safe_path(path)
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
            # Simple regex extraction of snippets
            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
            clean_snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:max_results]]
            if not clean_snippets:
                return f"[No immediate web search results found for: {query}]"
            return "\n---\n".join(clean_snippets)
        except Exception as e:
            return f"[Web Search unavailable: {e}]"

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
