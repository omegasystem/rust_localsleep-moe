import re
import json
from typing import Tuple, List, Optional
from tools_harness import EmbodiedToolHarness

class ToolCallParser:
    """
    容錯 Tool Call 解析器（借鑒 Python 原版 agentic_real_chat 與強韌容錯正規表達式）。
    支援：
      1. 標準格式：<tool_call>{"action": "...", "args": {...}}</tool_call>
      2. 殘缺未閉合標籤：</tool_call 或 </tool_call<end_of_turn>
      3. Python 原版仿生格式：<call:fs:read path="..." /> 或 <call:fs:list />
    """

    @staticmethod
    def detect_and_execute(text: str, harness: EmbodiedToolHarness) -> Tuple[Optional[str], Optional[str], str]:
        """
        解析文字中的 tool call。
        若偵測到 tool call，回傳 (action, tool_output, clean_text)。
        若未偵測到，回傳 (None, None, text)。
        """
        # 1. 檢查 Python 原版仿生短標籤：<call:fs:read ... /> 或 <call:fs:list ... />
        fs_list = re.search(r'<call:fs:(?:list|list_dir)\s*(?:path=["\'](.*?)["\'])?\s*/>', text)
        if fs_list:
            req_path = fs_list.group(1) or "."
            out = harness.dispatch("list_dir", {"path": req_path})
            clean = text[:fs_list.start()].rstrip()
            return "list_dir", out, clean

        fs_read = re.search(r'<call:fs:read\s+path=["\'](.*?)["\']\s*(?:offset=["\']?(\d+)["\']?)?\s*(?:limit=["\']?(\d+)["\']?)?\s*/>', text)
        if fs_read:
            req_path = fs_read.group(1)
            off = int(fs_read.group(2)) if fs_read.group(2) else None
            lim = int(fs_read.group(3)) if fs_read.group(3) else 60
            out = harness.dispatch("read_file", {"path": req_path, "offset": off, "limit": lim})
            clean = text[:fs_read.start()].rstrip()
            return "read_file", out, clean

        # 2. 檢查標準及容錯 JSON tool_call：<tool_call> ... (</tool_call> 或 </tool_call 或結尾)
        # 寬鬆匹配：就算模型漏掉右引號或角括號也能容錯抓取
        pattern = r'<tool_call>(.*?)(?:</tool_call>|(?=</tool_call)|(?=<end_of_turn>)|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            raw_payload = match.group(1).strip()
            # 必須包含起碼的 JSON 特徵
            if "{" in raw_payload and "}" in raw_payload:
                try:
                    # 擷取最外層的完整 JSON 區塊
                    j_start = raw_payload.find("{")
                    j_end = raw_payload.rfind("}") + 1
                    json_str = raw_payload[j_start:j_end]
                    call_obj = json.loads(json_str)
                    act = call_obj.get("action")
                    args = call_obj.get("args", {})

                    out = harness.dispatch(act, args)

                    clean = text[:match.start()].rstrip()
                    return act, out, clean
                except Exception as e:
                    clean = text[:match.start()].rstrip()
                    return "error", f"[Tool Call JSON Syntax Error: {e}]", clean

        return None, None, text

    @staticmethod
    def has_tool_call_start(text: str) -> bool:
        """快速檢測串流是否開始輸出工具呼叫"""
        return "<tool_call>" in text or "<call:fs:" in text
