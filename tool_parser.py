import re
import json
from typing import Tuple, List, Optional
from tools_harness import EmbodiedToolHarness
import logging

class ToolCallParser:
    """
    通用型 Tool Call 解析器 (Generic XML/JSON)
    不再綁定特定模型的 end_of_turn 標籤，具備更強的殘缺閉合修復能力。
    """
    
    @staticmethod
    def detect_and_execute(text: str, harness) -> tuple[str, str, str]:
        """
        解析文字中的 tool call。
        回傳: (action, tool_output, clean_text)
        """
        # 1. 舊版相容: <call:fs:read ... />
        fs_read_match = re.search(r'<call:fs:read\s+path=["\'](.*?)["\']\s*(?:offset=["\']?(\d+)["\']?)?\s*(?:limit=["\']?(\d+)["\']?)?\s*/>', text)
        if fs_read_match:
            req_path = fs_read_match.group(1)
            off = int(fs_read_match.group(2)) if fs_read_match.group(2) else None
            lim = int(fs_read_match.group(3)) if fs_read_match.group(3) else 150
            out = harness.dispatch("read_file", {"path": req_path, "offset": off, "limit": lim})
            clean = text[:fs_read_match.start()].rstrip()
            return "read_file", out, clean

        # 2. 舊版相容: <call:fs:list />
        fs_list_match = re.search(r'<call:fs:(?:list|list_dir)\s*(?:path=["\'](.*?)["\'])?\s*/>', text)
        if fs_list_match:
            req_path = fs_list_match.group(1) or "."
            out = harness.dispatch("list_dir", {"path": req_path})
            clean = text[:fs_list_match.start()].rstrip()
            return "list_dir", out, clean

        # 3. 泛用型 XML Tool Call: <tool_call>{...}</tool_call>
        # 移除 (?=<end_of_turn>)，改為泛用式的貪婪匹配與提早截斷
        pattern = r'<tool_call>(.*?)(?:</tool_call>|$)'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            raw_payload = match.group(1).strip()
            clean_text = text[:match.start()].rstrip()
            
            # 尋找有效的 JSON 區塊 (從第一個 { 到最後一個 })
            j_start = raw_payload.find('{')
            j_end = raw_payload.rfind('}')
            
            if j_start != -1 and j_end != -1 and j_end > j_start:
                json_str = raw_payload[j_start:j_end+1]
                try:
                    call_obj = json.loads(json_str)
                    act = call_obj.get("action")
                    args = call_obj.get("args", {})
                    if act:
                        out = harness.dispatch(act, args)
                        return act, out, clean_text
                except json.JSONDecodeError as e:
                    return None, f"[Tool Call JSON Syntax Error: {e}. Please output valid JSON format.]", clean_text
            else:
                # 針對模型忘記輸出 } 的癲癇狀況進行搶救
                try:
                    # 嘗試自動補齊結尾括號
                    repaired_json = raw_payload[j_start:] + "}"
                    call_obj = json.loads(repaired_json)
                    act = call_obj.get("action")
                    args = call_obj.get("args", {})
                    if act:
                        out = harness.dispatch(act, args)
                        return act, out, clean_text
                except Exception:
                    return None, "[Tool Call Error: Malformed JSON. You MUST use valid JSON format inside <tool_call>.]", clean_text

        return None, None, text

    @staticmethod
    def has_tool_call_start(text: str) -> bool:
        """快速檢測字串流是否包含工具呼叫的起點，用於 Stream 早期攔截"""
        return "<tool_call>" in text or "<call:fs:" in text