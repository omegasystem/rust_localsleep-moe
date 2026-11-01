import re
import json
from typing import Tuple, List, Optional
from tools_harness import EmbodiedToolHarness
import logging


class ToolCallParser:
    """
    通用型 Tool Call 解析器 (Generic XML/JSON)
    支援：
      1. `<tool_call>{"action": "...", "args": {...}}</tool_call>`
         - `read_file` (Full / Partial: `mode="full"|"partial"`, `start_line`, `end_line`, `offset`, `limit`, `chunk_size`)
         - `edit_file` (Block / Multi-Edits / Line-Range: `old_text`, `new_text`, `start_line`, `end_line`, `edits`)
         - `run_sandbox_shell` / `run_shell` / `run_python` (Restricted strictly to `isolated_output` sandbox!)
         - `write_file`, `list_dir`, `update_working_memory`, `read_working_memory`
      2. 快捷 XML 標籤相容：
         - `<call:fs:read path="..." mode="full|partial" start_line="1" end_line="100" />`
         - `<call:fs:edit path="..." start_line="10" end_line="20">...new code...</call:fs:edit>`
         - `<call:shell cmd="python3 mock_evaluator.py" />` or `<sandbox_shell>python3 mock_evaluator.py</sandbox_shell>`
    """

    @staticmethod
    def detect_and_execute(text: str, harness: EmbodiedToolHarness) -> tuple[Optional[str], Optional[str], str]:
        """
        解析文字中的 tool call。
        回傳: (action, tool_output, clean_text)
        """
        # 1. 快捷 XML 相容: <call:fs:read ... />
        fs_read_match = re.search(
            r'<call:fs:read\s+path=["\'](.*?)["\']'
            r'\s*(?:mode=["\'](.*?)["\'])?'
            r'\s*(?:offset=["\']?(\d+)["\']?)?'
            r'\s*(?:limit=["\']?(\d+)["\']?)?'
            r'\s*(?:start_line=["\']?(\d+)["\']?)?'
            r'\s*(?:end_line=["\']?(\d+)["\']?)?\s*/>',
            text,
        )
        if fs_read_match:
            req_path = fs_read_match.group(1)
            mode = fs_read_match.group(2) or "partial"
            off = int(fs_read_match.group(3)) if fs_read_match.group(3) else None
            lim = int(fs_read_match.group(4)) if fs_read_match.group(4) else 160
            s_ln = int(fs_read_match.group(5)) if fs_read_match.group(5) else None
            e_ln = int(fs_read_match.group(6)) if fs_read_match.group(6) else None
            out = harness.dispatch(
                "read_file",
                {"path": req_path, "mode": mode, "offset": off, "limit": lim, "start_line": s_ln, "end_line": e_ln},
            )
            clean = text[: fs_read_match.start()].rstrip()
            return "read_file", out, clean

        # 2. 快捷 XML 相容: <call:fs:list />
        fs_list_match = re.search(r'<call:fs:(?:list|list_dir)\s*(?:path=["\'](.*?)["\'])?\s*/>', text)
        if fs_list_match:
            req_path = fs_list_match.group(1) or "."
            out = harness.dispatch("list_dir", {"path": req_path})
            clean = text[: fs_list_match.start()].rstrip()
            return "list_dir", out, clean

        # 3. 快捷 XML 相容: <call:shell cmd="..." /> 或 <sandbox_shell>...</sandbox_shell>
        sh_tag_match = re.search(r'<call:shell\s+(?:cmd|command)=["\'](.*?)["\']\s*/>', text)
        if sh_tag_match:
            cmd_str = sh_tag_match.group(1)
            out = harness.dispatch("run_sandbox_shell", {"command": cmd_str})
            clean = text[: sh_tag_match.start()].rstrip()
            return "run_sandbox_shell", out, clean

        sb_sh_match = re.search(r'<sandbox_shell>(.*?)</sandbox_shell>', text, re.DOTALL)
        if sb_sh_match:
            cmd_str = sb_sh_match.group(1).strip()
            out = harness.dispatch("run_sandbox_shell", {"command": cmd_str})
            clean = text[: sb_sh_match.start()].rstrip()
            return "run_sandbox_shell", out, clean

        # 4. 泛用型 XML Tool Call: <tool_call>{...}</tool_call>
        pattern = r"<tool_call>(.*?)(?:</tool_call>|$)"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            raw_payload = match.group(1).strip()
            clean_text = text[: match.start()].rstrip()

            j_start = raw_payload.find("{")
            j_end = raw_payload.rfind("}")

            if j_start != -1 and j_end != -1 and j_end > j_start:
                json_str = raw_payload[j_start : j_end + 1]
                try:
                    call_obj = json.loads(json_str)
                    act = call_obj.get("action") or call_obj.get("name") or call_obj.get("tool")
                    args = call_obj.get("args") or call_obj.get("arguments") or {
                        k: v for k, v in call_obj.items() if k not in ("action", "name", "tool")
                    }
                    if act:
                        out = harness.dispatch(act, args)
                        return act, out, clean_text
                except json.JSONDecodeError as e:
                    return None, f"[Tool Call JSON Syntax Error: {e}. Please output valid JSON format.]", clean_text
            else:
                try:
                    repaired_json = raw_payload[j_start:] + "}"
                    call_obj = json.loads(repaired_json)
                    act = call_obj.get("action") or call_obj.get("name") or call_obj.get("tool")
                    args = call_obj.get("args") or call_obj.get("arguments") or {}
                    if act:
                        out = harness.dispatch(act, args)
                        return act, out, clean_text
                except Exception:
                    return None, "[Tool Call Error: Malformed JSON. You MUST use valid JSON format inside <tool_call>.]", clean_text

        return None, None, text

    @staticmethod
    def has_tool_call_start(text: str) -> bool:
        """快速檢測字串流是否包含工具呼叫的起點，用於 Stream 早期攔截"""
        return (
            "<tool_call>" in text
            or "<call:fs:" in text
            or "<call:shell" in text
            or "<sandbox_shell>" in text
        )

    @staticmethod
    def has_complete_tool_call(text: str) -> bool:
        """檢測字串流是否已完整閉合一個 Tool Call"""
        return (
            "</tool_call>" in text
            or ("<tool_call>" in text and ("</tool_call" in text or text.endswith("}\n")))
            or ("<call:fs:" in text and "/>" in text)
            or ("<call:shell" in text and "/>" in text)
            or "</sandbox_shell>" in text
        )

    @staticmethod
    def repair_missing_closing_angle(text: str) -> tuple[bool, str]:
        """
        若串流剛吐出 `</tool_call` (因為 Tokenizer 將 `</tool_call` 與 `>` 拆成兩個 token)，
        自動補上結尾的 `>`，避免畫面停在 `</tool_call` 少一個 `>`！
        回傳: (did_repair, repaired_text)
        """
        if text.endswith("</tool_call"):
            return True, text + ">"
        return False, text
