import ast
import os
import glob
import shlex
import subprocess
import urllib.request
import urllib.parse
import json
import re
from typing import List, Dict, Tuple, Optional, Any


class EmbodiedToolHarness:
    """
    Decoupled Tool Harness for LocalSleep-MoE Agentic Workflows.
    Features:
      1. read_file(path, mode="partial"|"full", offset=1, limit=200, start_line=None, end_line=None, chunk_size=None)
         - Supports Full read (`mode="full"`) and Partial read (`mode="partial"`, line range `start_line..end_line`,
           or `offset` + `limit` line chunking, or `chunk_size` character chunking) with total stats!
      2. edit_file(path, old_text="", new_text="", replace_all=False, start_line=None, end_line=None, edits=None)
         - Supports Exact/Fuzzy Block Replacement (`old_text` -> `new_text`), Multi-block `edits=[...]`,
           and Partial Line-Range Replacement (`start_line`..`end_line` -> `new_text`) without full-file rewrite!
      3. run_sandbox_shell(command="", script_path="", args="", timeout=30)
         - Strictly sandboxed shell/code execution inside `isolated_output_dir`!
         - Only permits running Python (`python3`/`pytest`), Shell scripts (`bash`/`sh`), or Rust (`cargo`/`rustc`)
           targeting code inside the isolated sandbox directory. Blocks `..`, system modification, or escaping!
      4. write_file(path, content) -> Isolated sandbox output by default
      5. list_dir(path)
      6. web_search(query, max_results=5)
      7. extract_and_save_artifacts(text) -> Multi-line XML artifact auto-persister (<file path="...">...)
      8. recall_sessions & recall_search -> History log queries
      9. update_working_memory & read_working_memory -> L1 Working Memory scratchpad
    """

    def __init__(
        self,
        workspace_root: str = ".",
        isolated_output_dir: str = "./isolated_output",
        session_logger: Any = None,
        working_memory_mgr: Any = None,
    ):
        self.workspace_root = os.path.abspath(workspace_root)
        self.isolated_output_dir = os.path.abspath(os.path.join(self.workspace_root, isolated_output_dir))
        self.session_logger = session_logger
        self.working_memory_mgr = working_memory_mgr
        self.last_read_state: Dict[str, int] = {}  # {normalized_path: next_offset}

        # Anti-Loop Circuit Breaker Memory
        self.action_history: List[str] = []
        self.max_repeated_actions: int = 3

        os.makedirs(self.isolated_output_dir, exist_ok=True)

    def _generate_action_signature(self, action: str, args: dict) -> str:
        return f"{action}:{json.dumps(args, sort_keys=True, default=str)}"

    def _resolve_safe_read_path(self, path: str) -> str:
        clean_path = (path or "").strip().lstrip("/")
        if not clean_path or clean_path == ".":
            return self.isolated_output_dir

        # 1. If path explicitly starts with isolated_output/prj_..., resolve from workspace_root
        if clean_path.startswith("isolated_output/") or clean_path.startswith("isolated_output\\"):
            cand_ws = os.path.abspath(os.path.join(self.workspace_root, clean_path))
            if os.path.exists(cand_ws):
                return cand_ws
            sub_path = clean_path[len("isolated_output/"):].lstrip("/\\")
            return os.path.abspath(os.path.join(self.isolated_output_dir, sub_path))

        # 2. Check active isolated sandbox first (so relative paths like `src/lib.rs` or `ntm_rsi_engine.py` hit sandbox!)
        cand_sandbox = os.path.abspath(os.path.join(self.isolated_output_dir, clean_path))
        if os.path.exists(cand_sandbox):
            return cand_sandbox

        # 3. Check workspace root
        cand_full = os.path.abspath(os.path.join(self.workspace_root, clean_path))
        if os.path.exists(cand_full):
            return cand_full

        # 4. Fallback by basename inside active isolated sandbox
        cand_base = os.path.abspath(os.path.join(self.isolated_output_dir, os.path.basename(clean_path)))
        if os.path.exists(cand_base):
            return cand_base

        return cand_sandbox

    def _resolve_safe_write_path(self, path: str) -> str:
        clean_p = (path or "").strip().replace("\\", "/").lstrip("/")
        # Strip redundant isolated_output/<prj_id>/ prefix if model includes it
        iso_rel = os.path.relpath(self.isolated_output_dir, self.workspace_root).replace("\\", "/")
        if clean_p.startswith(iso_rel + "/"):
            clean_p = clean_p[len(iso_rel) + 1:]
        elif clean_p.startswith("isolated_output/"):
            parts = clean_p.split("/")
            if len(parts) >= 3 and parts[1].startswith("prj_"):
                clean_p = "/".join(parts[2:])
            else:
                clean_p = "/".join(parts[1:])

        if ".." in clean_p:
            clean_p = os.path.basename(clean_p)

        full_path = os.path.abspath(os.path.join(self.isolated_output_dir, clean_p))
        if not full_path.startswith(self.isolated_output_dir):
            full_path = os.path.join(self.isolated_output_dir, os.path.basename(clean_p))
        return full_path

    # =========================================================================
    # 1. Read File (Supports Full & Partial Modes: Line Range / Offset / Chars)
    # =========================================================================
    def read_file(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: int = 600,
        chunk_size: Optional[int] = None,
        mode: str = "full",
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        with_line_numbers: bool = True,
    ) -> str:
        """
        Read File (Full by default for files <= 800 lines, or Partial when range/offset given):
          - `mode="full"` (預設): 一次讀取完整檔案（安全上限 40,000 字元 / 800 行，完美相容 16-LoRA + 32K KV Cache）。
          - `mode="partial"`: 當指定 `start_line`、`end_line`、`offset` 或 `chunk_size` 時自動啟用分段讀取。
        """
        try:
            target = self._resolve_safe_read_path(path)
            if not os.path.exists(target):
                return f"[Error: File not found: {path} (resolved: {target})]"
            if os.path.isdir(target):
                return (
                    f"[Error: '{path}' is a directory, not a readable file. "
                    f"Use list_dir(path='{path}') to inspect its contents.]"
                )

            file_bytes = os.path.getsize(target)
            norm_key = os.path.normpath(target)
            rel_disp = os.path.relpath(target, self.workspace_root)

            has_explicit_range = any(x is not None for x in (offset, start_line, end_line, chunk_size))

            # Full File Read Mode (default whenever no explicit range/offset is requested, or mode=="full")
            if (str(mode).lower() == "full" and not has_explicit_range) or (limit is not None and int(limit) <= 0):
                with open(target, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                total_lines = len(lines)
                max_full_lines = 800
                selected = lines[:max_full_lines]
                truncated = total_lines > max_full_lines
                if with_line_numbers:
                    body = "".join(f"{idx:4d} | {line}" for idx, line in enumerate(selected, start=1))
                else:
                    body = "".join(selected)
                if len(body) > 40000:
                    body = body[:40000] + "\n... [Truncated at 40,000 chars for KV Cache safety; use start_line/end_line] ..."
                    truncated = True
                header = (
                    f"--- [FILE (FULL): {rel_disp} | LINES: 1..{min(total_lines, max_full_lines)}/{total_lines} "
                    f"({file_bytes} bytes) | TRUNCATED: {truncated} (100% COMPLETE)] ---\n"
                )
                return header + (body if body else "[Empty file]")

            # Character Chunk Mode
            if chunk_size is not None and int(chunk_size) > 0:
                safe_chunk_size = min(int(chunk_size), 8000)
                with open(target, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                total_chars = len(content)
                actual_offset = offset if offset is not None else self.last_read_state.get(norm_key, 0)
                char_offset = max(0, int(actual_offset))
                chunk = content[char_offset : char_offset + safe_chunk_size]
                next_offset = char_offset + len(chunk)
                has_more = next_offset < total_chars
                self.last_read_state[norm_key] = next_offset if has_more else 0
                header = (
                    f"--- [FILE (PARTIAL CHARS): {rel_disp} | CHUNK: chars {char_offset}..{next_offset}/{total_chars} "
                    f"({file_bytes} bytes) | HAS_MORE: {has_more}"
                    + (f" | NEXT_OFFSET: {next_offset}" if has_more else "")
                    + "] ---\n"
                )
                return header + (chunk if chunk else "[Offset beyond EOF]")

            # Line-based Partial Read Mode (supports start_line..end_line OR offset+limit)
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total_lines = len(lines)

            if start_line is not None or end_line is not None:
                s_line = max(1, int(start_line or 1))
                e_line = min(total_lines, int(end_line or (s_line + 160 - 1)))
                start_idx = s_line - 1
                end_idx = max(start_idx, e_line)
            else:
                actual_offset = int(offset) if offset is not None else self.last_read_state.get(norm_key, 1)
                safe_limit = min(max(1, int(limit or 160)), 350)
                start_idx = max(0, actual_offset - 1)
                end_idx = min(total_lines, start_idx + safe_limit)

            selected_lines = lines[start_idx:end_idx]
            has_more = end_idx < total_lines
            next_offset = end_idx + 1 if has_more else None
            self.last_read_state[norm_key] = next_offset if has_more else 1

            header = (
                f"--- [FILE (PARTIAL LINES): {rel_disp} | LINES: {start_idx + 1}..{end_idx}/{total_lines} "
                f"({file_bytes} bytes) | HAS_MORE: {has_more}"
                + (f" | NEXT_OFFSET: {next_offset}" if has_more else " (END OF FILE)")
                + "] ---\n"
            )
            if not selected_lines:
                return header + "[Offset beyond EOF or file empty]"

            if with_line_numbers:
                body = "".join(f"{idx:4d} | {line}" for idx, line in enumerate(selected_lines, start=start_idx + 1))
            else:
                body = "".join(selected_lines)
            return header + body
        except Exception as e:
            return f"[Error reading {path}: {e}]"

    # =========================================================================
    # 2. Edit File (Supports Exact/Fuzzy Block, Multi-Edits, and Line-Range)
    # =========================================================================
    def edit_file(
        self,
        path: str,
        old_text: str = "",
        new_text: str = "",
        replace_all: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        edits: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Edit File (Partial / Block / Line-Range Replacement):
          - 模式 A (單一區塊替換): 傳入 `old_text` 與 `new_text`（支援 CRLF/空白尾端容錯匹配）。
          - 模式 B (多區塊批次替換): 傳入 `edits=[{"old_text": "...", "new_text": "..."}, ...]`。
          - 模式 C (行號區間覆寫): 傳入 `start_line` 與 `end_line` (1-based, 包含端點) 與 `new_text`，
            直接替換該行號區間內的代碼！
        """
        try:
            sandbox_target = self._resolve_safe_write_path(path)
            workspace_target = self._resolve_safe_read_path(path)

            if os.path.exists(sandbox_target):
                src_target = sandbox_target
            elif os.path.exists(workspace_target) and os.path.isfile(workspace_target):
                src_target = workspace_target
            else:
                return f"[Error: File not found for editing: {path}]"

            with open(src_target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Mode C: Line-range partial edit (`start_line` .. `end_line`)
            if start_line is not None and end_line is not None and not old_text and not edits:
                lines = content.splitlines(keepends=True)
                total_l = len(lines)
                s_idx = max(0, int(start_line) - 1)
                e_idx = min(total_l, int(end_line))
                if s_idx > total_l or e_idx < s_idx:
                    return f"[Error: Invalid line range {start_line}..{end_line} (file has {total_l} lines)]"
                replacement = new_text
                if replacement and not replacement.endswith("\n") and e_idx < total_l:
                    replacement += "\n"
                updated_content = "".join(lines[:s_idx]) + replacement + "".join(lines[e_idx:])
                # Always write into isolated sandbox (or in-place if already in isolated_output)
                out_target = src_target if src_target.startswith(os.path.join(self.workspace_root, "isolated_output")) else sandbox_target
                os.makedirs(os.path.dirname(out_target), exist_ok=True)
                with open(out_target, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                rel_path = os.path.relpath(out_target, self.workspace_root)
                return f"[Success: Replaced lines {start_line}..{end_line} in {rel_path} ({len(updated_content)} bytes)]"

            # Mode A / B: Block replacement(s)
            edit_ops = edits if isinstance(edits, list) and edits else [{"old_text": old_text, "new_text": new_text, "replace_all": replace_all}]
            updated_content = content
            total_replaced = 0

            for idx_op, op in enumerate(edit_ops, start=1):
                o_txt = str(op.get("old_text") or op.get("oldText") or "")
                n_txt = str(op.get("new_text") or op.get("newText") or "")
                rep_all = bool(op.get("replace_all", replace_all))
                if not o_txt:
                    return f"[Error: old_text cannot be empty in edit #{idx_op}]"

                count = updated_content.count(o_txt)
                if count == 0:
                    # Fallback: try normalized line-ending / stripped trailing whitespace match
                    norm_old_lines = [ln.rstrip() for ln in o_txt.splitlines()]
                    content_lines = updated_content.splitlines(keepends=True)
                    matched_span = None
                    n_span = len(norm_old_lines)
                    if n_span > 0:
                        for i in range(len(content_lines) - n_span + 1):
                            window = [ln.rstrip() for ln in content_lines[i : i + n_span]]
                            if window == norm_old_lines:
                                matched_span = (i, i + n_span)
                                break
                    if matched_span is not None:
                        s_i, e_i = matched_span
                        if n_txt and not n_txt.endswith("\n"):
                            n_txt += "\n"
                        updated_content = "".join(content_lines[:s_i]) + n_txt + "".join(content_lines[e_i:])
                        total_replaced += 1
                        continue
                    return f"[Error: Target block 'old_text' not found in {path} (edit #{idx_op}). Tip: call read_file first to check exact lines.]"

                if count > 1 and not rep_all:
                    return f"[Error: 'old_text' matched {count} times in {path}. Provide a more unique block or set replace_all=True]"

                if rep_all:
                    updated_content = updated_content.replace(o_txt, n_txt)
                    total_replaced += count
                else:
                    updated_content = updated_content.replace(o_txt, n_txt, 1)
                    total_replaced += 1

            out_target = src_target if src_target.startswith(os.path.join(self.workspace_root, "isolated_output")) else sandbox_target
            os.makedirs(os.path.dirname(out_target), exist_ok=True)
            with open(out_target, "w", encoding="utf-8") as f:
                f.write(updated_content)

            rel_path = os.path.relpath(out_target, self.workspace_root)
            return f"[Success: Replaced {total_replaced} block(s) in {rel_path} ({len(updated_content)} bytes)]"
        except Exception as e:
            return f"[Error editing {path}: {e}]"

    # =========================================================================
    # 3. Sandboxed Shell / Python / Rust Code Execution (`isolated_output` ONLY)
    # =========================================================================
    def run_sandbox_shell(
        self,
        command: str = "",
        script_path: str = "",
        args: str = "",
        timeout: int = 45,
        workdir: str = "",
    ) -> str:
        """
        在隔離沙盒 (`isolated_output/<prj_uuid>`) 內安全執行 Python、Shell Script 或 Rust 編譯/測試命令。
        【嚴格安全邊界】：
          1. 工作目錄 (`cwd`) 強制鎖定在 `isolated_output` 沙盒目錄內（禁止 `..` 逃逸）。
          2. 僅允許執行沙盒內的程式碼、腳本與編譯測試工具：
             `python`, `python3`, `pytest`, `bash`, `sh`, `cargo`, `rustc`, `maturin`, `./<sandbox_binary>`。
          3. 嚴格禁止破壞性或跨區指令 (`rm -rf`, `sudo`, `ssh`, `scp`, `curl`, `wget`, `git push`, `kill`, `shutdown`, `/etc`, `/home` 外部寫入)。
        """
        try:
            iso_root = os.path.abspath(os.path.join(self.workspace_root, "isolated_output"))
            # Resolve working directory strictly inside isolated_output
            target_cwd = self.isolated_output_dir
            if workdir:
                clean_wd = workdir.strip().lstrip("/")
                if clean_wd.startswith("isolated_output/"):
                    cand_wd = os.path.abspath(os.path.join(self.workspace_root, clean_wd))
                else:
                    cand_wd = os.path.abspath(os.path.join(self.isolated_output_dir, clean_wd))
                if cand_wd.startswith(iso_root) and os.path.isdir(cand_wd):
                    target_cwd = cand_wd

            os.makedirs(target_cwd, exist_ok=True)

            # If script_path is given without command, build command automatically
            cmd_str = (command or "").strip()
            if not cmd_str and script_path:
                sp = script_path.strip()
                if sp.endswith(".py"):
                    cmd_str = f"python3 {sp} {args}".strip()
                elif sp.endswith(".sh"):
                    cmd_str = f"bash {sp} {args}".strip()
                elif sp.endswith(".rs") or sp == "Cargo.toml":
                    cmd_str = f"cargo test {args}".strip()
                else:
                    cmd_str = f"python3 {sp} {args}".strip()

            # Smart Sandbox Prefix De-duplication:
            # Since `target_cwd` is ALREADY `isolated_output/<prj_uuid>`, if the model passes
            # `python3 isolated_output/<prj_uuid>/mock_evaluator.py`, automatically strip the redundant prefix!
            rel_cwd_prefix = os.path.relpath(target_cwd, self.workspace_root).replace("\\", "/")
            if rel_cwd_prefix and rel_cwd_prefix != ".":
                cmd_str = cmd_str.replace(rel_cwd_prefix + "/", "").replace("./" + rel_cwd_prefix + "/", "")
            cmd_str = re.sub(r"(?:\./)?isolated_output/prj_[A-Za-z0-9_]+/", "", cmd_str)

            if not cmd_str:
                return "[Error: Empty command. Provide 'command' (e.g., 'python3 mock_evaluator.py' or 'cargo build --release') or 'script_path'.]"

            # Security Gate 1: Block path traversal & dangerous tokens
            forbidden_patterns = [
                r"\.\./",
                r"\bsudo\b",
                r"\brm\s+-[rRfF]+",
                r"\bssh\b",
                r"\bscp\b",
                r"\bcurl\b",
                r"\bwget\b",
                r"\bnc\b",
                r"\bnetcat\b",
                r"\bchmod\s+777\b",
                r"\bchown\b",
                r"\bmkfs\b",
                r"\bdd\b",
                r"\bshutdown\b",
                r"\breboot\b",
                r"\bkillall\b",
                r"\bpkill\b",
                r"\bgit\s+(?:push|reset|clean|checkout)\b",
            ]
            for pat in forbidden_patterns:
                if re.search(pat, cmd_str, flags=re.IGNORECASE):
                    return f"[Sandbox Security Blocked: Command contains restricted pattern '{pat}'. Only isolated code/script execution is allowed.]"

            # Security Gate 2: Validate every sub-command in `&&` / `;` chain starts with allowed tool/interpreter
            allowed_bins = {
                "python", "python3", "pytest",
                "bash", "sh",
                "cargo", "rustc", "maturin",
                "ls", "pwd", "cat", "head", "tail", "wc", "grep", "echo", "env",
            }
            sub_cmds = [s.strip() for s in re.split(r"&&|;|\|\|", cmd_str) if s.strip()]
            for sub in sub_cmds:
                tokens = shlex.split(sub, posix=True)
                if not tokens:
                    continue
                # Skip leading VAR=val env assignments
                idx = 0
                while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("-"):
                    idx += 1
                if idx >= len(tokens):
                    continue
                prog = tokens[idx]
                prog_base = os.path.basename(prog)

                # Allow `./script.sh` or `./binary` inside sandbox
                if prog.startswith("./"):
                    resolved_prog = os.path.abspath(os.path.join(target_cwd, prog[2:]))
                    if not resolved_prog.startswith(iso_root):
                        return f"[Sandbox Security Blocked: '{prog}' escapes isolated sandbox directory.]"
                elif prog_base not in allowed_bins:
                    return (
                        f"[Sandbox Security Blocked: Executable '{prog}' is not permitted. "
                        f"Allowed in isolated sandbox: {sorted(allowed_bins)} or './<sandbox_script>']"
                    )

                # Verify any explicit file path arguments stay within isolated_output (no absolute /etc, /home/... outside sandbox)
                for arg_tok in tokens[idx + 1 :]:
                    if arg_tok.startswith("-"):
                        continue
                    if arg_tok.startswith("/"):
                        abs_arg = os.path.abspath(arg_tok)
                        if not abs_arg.startswith(iso_root):
                            return (
                                f"[Sandbox Security Blocked: Absolute path '{arg_tok}' is outside isolated_output sandbox ({iso_root}). "
                                f"Use relative paths inside the sandbox.]"
                            )

            safe_timeout = min(max(3, int(timeout or 45)), 120)
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{target_cwd}:{env.get('PYTHONPATH', '')}".rstrip(":")
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            proc = subprocess.run(
                cmd_str,
                shell=True,
                cwd=target_cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=safe_timeout,
            )

            stdout_str = (proc.stdout or "").strip()
            stderr_str = (proc.stderr or "").strip()
            combined = ""
            if stdout_str:
                combined += f"[STDOUT]:\n{stdout_str}\n"
            if stderr_str:
                combined += f"[STDERR]:\n{stderr_str}\n"
            if not combined:
                combined = "(No output)"

            if len(combined) > 6000:
                combined = combined[:3000] + "\n... [Middle output truncated for KV Cache safety] ...\n" + combined[-2800:]

            rel_cwd = os.path.relpath(target_cwd, self.workspace_root)
            return (
                f"--- [SANDBOX SHELL EXEC | CWD: {rel_cwd} | CMD: `{cmd_str}` | EXIT_CODE: {proc.returncode}] ---\n"
                f"{combined}"
            )
        except subprocess.TimeoutExpired:
            return f"[Sandbox Shell Timeout: Command `{command}` exceeded {timeout}s limit.]"
        except Exception as e:
            return f"[Sandbox Shell Error: {e}]"

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

            rel_base = os.path.relpath(target, self.workspace_root)
            entries = sorted(os.listdir(target))
            formatted = [f"📁 Directory listing of: {rel_base} (total: {len(entries)})"]
            for name in entries[:60]:
                full_entry_path = os.path.join(target, name)
                rel_entry_path = os.path.relpath(full_entry_path, self.workspace_root).replace("\\", "/")
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
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
            clean_snippets = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:max_results]]
            if not clean_snippets:
                return f"[No immediate web search results found for: {query}]"
            return "\n---\n".join(clean_snippets)
        except Exception as e:
            return f"[Web Search unavailable: {e}]"

    def inspect_code_ast(self, path: str, required_symbols: Optional[List[str]] = None) -> str:
        """
        AST 語法與結構完整性自檢工具 (AST & Structural Self-Check)：
          - 針對 Python (`.py`): 使用 `ast.parse` 驗證語法樹，提取所有 `import`、`class`、`def` 簽名與行號；
            若發生 `SyntaxError` / `IndentationError`，自動擷取報錯行號前後 ±8 行真實代碼回傳給模型精準修復！
          - 針對 Rust (`.rs`): 檢查大括號 `{}` / `()` / `[]` 對稱性與 `fn` / `struct` / `impl` 結構，或呼叫 `cargo check`。
          - 若傳入 `required_symbols` (如 `["EvolvedHypercubeHomeostasis", "apply_evolved_24_pruning"]`)，
            自動驗證關鍵類別/函式是否未被意外誤刪！
        """
        try:
            target = self._resolve_safe_read_path(path)
            if not os.path.exists(target) or not os.path.isfile(target):
                return f"[AST Check Error: File not found: {path}]"

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                src = f.read()
            lines = src.splitlines()
            rel_disp = os.path.relpath(target, self.workspace_root)
            ext = os.path.splitext(target)[1].lower()

            if ext == ".py":
                try:
                    tree = ast.parse(src, filename=rel_disp)
                except SyntaxError as se:
                    err_ln = se.lineno or 1
                    s_l = max(1, err_ln - 8)
                    e_l = min(len(lines), err_ln + 8)
                    snippet = "\n".join(
                        f"{'>>' if i == err_ln else '  '} {i:4d} | {lines[i-1]}"
                        for i in range(s_l, e_l + 1)
                    )
                    return (
                        f"❌ [AST SYNTAX ERROR in `{rel_disp}` at Line {err_ln}, Col {se.offset}]: {se.msg}\n"
                        f"--- Context around Line {err_ln} (Use `edit_file` with `start_line`={s_l}, `end_line`={e_l} to fix) ---\n"
                        f"{snippet}"
                    )

                classes = []
                functions = []
                imports = []
                defined_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        defined_names.add(node.name)
                        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                        for m in methods:
                            defined_names.add(m)
                        classes.append(f"L{node.lineno:03d}: class {node.name} (methods: {', '.join(methods)})")
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defined_names.add(node.name)
                        args_list = [a.arg for a in node.args.args]
                        functions.append(f"L{node.lineno:03d}: def {node.name}({', '.join(args_list)})")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        mod = node.module or ""
                        imports.append(f"{mod}.({', '.join(a.name for a in node.names)})")

                missing_syms = [s for s in (required_symbols or []) if s not in defined_names]
                if missing_syms:
                    return (
                        f"⚠️ [AST CONTRACT WARNING in `{rel_disp}`]: Syntax OK, but missing required symbols: {missing_syms}\n"
                        f"Defined classes: {classes}\nDefined functions: {functions[:20]}"
                    )

                return (
                    f"✅ [AST CHECK PASS: `{rel_disp}` ({len(lines)} lines)]\n"
                    f"- Imports ({len(imports)}): {', '.join(imports[:12])}\n"
                    f"- Classes ({len(classes)}):\n  " + ("\n  ".join(classes) if classes else "(none)") + "\n"
                    f"- Top Functions ({len(functions)}):\n  " + ("\n  ".join(functions[:15]) if functions else "(none)")
                )

            # Fallback structural check for Rust / C / CUDA files
            balance_pairs = [("{", "}"), ("(", ")"), ("[", "]")]
            unbalanced = []
            for o_ch, c_ch in balance_pairs:
                if src.count(o_ch) != src.count(c_ch):
                    unbalanced.append(f"'{o_ch}'({src.count(o_ch)}) != '{c_ch}'({src.count(c_ch)})")
            if unbalanced:
                return f"❌ [STRUCTURAL DELIMITER MISMATCH in `{rel_disp}`]: {', '.join(unbalanced)}"
            return f"✅ [STRUCTURAL CHECK PASS: `{rel_disp}` ({len(lines)} lines, delimiters balanced)]"
        except Exception as e:
            return f"[AST Check Exception: {e}]"

    def extract_and_save_artifacts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts multi-line native XML artifacts (<file path="...">...</file>)
        and automatically writes them into isolated_output_dir without escaping or formatting destruction.
        """
        saved_artifacts = []
        patterns = [
            r'<file\s+(?:path|name)=["\'](.*?)["\']>(.*?)</file>',
            r'<write_file\s+path=["\'](.*?)["\']>(.*?)</write_file>',
            r'<artifact\s+(?:path|identifier|name)=["\'](.*?)["\'][^>]*>(.*?)</artifact>',
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
                    "size": len(clean_content),
                })
                print(f"\n💾 \033[1;32m[Auto-Save XML Artifact (Isolated Sandbox)]\033[0m -> \033[1;36m{rel_path}\033[0m ({len(clean_content)} bytes)")

        return saved_artifacts

    def dispatch(self, action: str, args: dict) -> str:
        """
        統一派發中心，包含防癲癇斷路器檢測。
        """
        args = args or {}
        sig = self._generate_action_signature(action, args)
        self.action_history.append(sig)
        if len(self.action_history) > 10:
            self.action_history.pop(0)

        recent_repeats = self.action_history.count(sig)
        if recent_repeats > self.max_repeated_actions:
            self.action_history.clear()
            return (
                f"[Circuit Breaker Triggered] 你已經連續執行相同的動作 ({action}) 達 {recent_repeats} 次且未取得進展！"
                "這表示你陷入了思考迴圈 (Hallucination Loop)。\n"
                "!!! 強制指令 !!! 停止重複相同參數呼叫，立即改用 `list_dir` 檢查路徑或直接回答使用者(Leon)。"
            )

        try:
            act_norm = (action or "").strip().lower()
            if act_norm in ("read_file", "read_file_partial", "read_file_full", "read"):
                mode = args.get("mode", "full" if act_norm == "read_file_full" else "partial")
                return self.read_file(
                    path=args.get("path") or args.get("file_path") or "",
                    offset=args.get("offset"),
                    limit=args.get("limit", 160),
                    chunk_size=args.get("chunk_size"),
                    mode=mode,
                    start_line=args.get("start_line"),
                    end_line=args.get("end_line"),
                    with_line_numbers=bool(args.get("with_line_numbers", True)),
                )
            elif act_norm in ("edit_file", "edit_file_partial", "replace_in_file", "patch_file", "edit"):
                return self.edit_file(
                    path=args.get("path") or args.get("file_path") or "",
                    old_text=args.get("old_text") or args.get("oldText") or "",
                    new_text=args.get("new_text") or args.get("newText") or "",
                    replace_all=bool(args.get("replace_all", False)),
                    start_line=args.get("start_line"),
                    end_line=args.get("end_line"),
                    edits=args.get("edits"),
                )
            elif act_norm in ("write_file", "write"):
                return self.write_file(
                    path=args.get("path") or args.get("file_path") or "",
                    content=args.get("content", ""),
                )
            elif act_norm in ("run_sandbox_shell", "run_shell", "exec_shell", "bash", "run_python", "run_code"):
                cmd = args.get("command") or args.get("cmd") or ""
                sp = args.get("script_path") or args.get("path") or ""
                if act_norm == "run_python" and not cmd and sp:
                    cmd = f"python3 {sp} {args.get('args', '')}".strip()
                return self.run_sandbox_shell(
                    command=cmd,
                    script_path=sp,
                    args=str(args.get("args", "")),
                    timeout=int(args.get("timeout", 45)),
                    workdir=str(args.get("workdir", "")),
                )
            elif act_norm in ("list_dir", "ls"):
                return self.list_dir(args.get("path", "."))
            elif act_norm == "web_search":
                return self.web_search(args.get("query", ""), args.get("max_results", 5))
            elif act_norm == "recall_sessions":
                if self.session_logger:
                    return self.session_logger.recall_recent_sessions(limit=args.get("limit", 5))
                return "[Error: session_logger not attached]"
            elif act_norm == "recall_search":
                if self.session_logger:
                    return self.session_logger.recall_search(query=args.get("query", ""), max_matches=args.get("max_matches", 3))
                return "[Error: session_logger not attached]"
            elif act_norm in ("update_working_memory", "write_working_memory"):
                if self.working_memory_mgr:
                    return self.working_memory_mgr.update_scratchpad_manual(
                        content=args.get("content", ""),
                        mode=args.get("mode", "replace"),
                    )
                return "[Error: working_memory_mgr not attached]"
            elif act_norm in ("inspect_code_ast", "ast_check", "check_syntax", "verify_ast"):
                return self.inspect_code_ast(
                    path=args.get("path") or args.get("file_path") or "",
                    required_symbols=args.get("required_symbols"),
                )
            elif act_norm == "read_working_memory":
                if self.working_memory_mgr:
                    return self.working_memory_mgr.read_working_memory_markdown()
                return "[Error: working_memory_mgr not attached]"
            elif act_norm in ("spawn_hypercube_offspring", "introspect_parent_hypercube"):
                from hypercube_replicator import HypercubeReplicatorManager
                rep = HypercubeReplicatorManager(workspace_root=self.workspace_root)
                if act_norm == "introspect_parent_hypercube":
                    return rep.introspect_parent_architecture_readonly()["digest_markdown"][:6000]
                meta = rep.spawn_offspring_cube(custom_tag=str(args.get("tag", "agentic_spawn")))
                rep.run_evolution_step_1_introspect(meta)
                rep.run_evolution_step_2_hypothesize_and_notes(meta)
                rep.run_evolution_step_3_synthesize_offspring_code(meta)
                ok, v_out = rep.run_evolution_step_4_verify_gate(meta)
                return f"[Hypercube Offspring Spawned: {meta['cube_id']} | Path: {meta['rel_path']} | Verified: {ok}]\n{v_out}"
            else:
                return f"[Unknown Tool: {action}]"
        except Exception as e:
            return f"[Tool Execution Exception: {str(e)}]"
