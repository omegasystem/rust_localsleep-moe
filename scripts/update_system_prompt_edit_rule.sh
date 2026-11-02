#!/usr/bin/env bash
set -euo pipefail

cd /home/node/.openclaw/workspace/projects/rust_localsleep-moe

python3 - <<'PY'
import json
from pathlib import Path

RULE_LINE = (
    "【檔案覆蓋與局部修補鐵律】：如果有 draft 存在且已經上一輪存檔確認後須實驗，"
    "直接使用 run_sandbox_shell 執行 cp/mv 蓋掉目標檔案進入編譯；"
    "且如有 Bug 請使用 edit_file 功能（搭配 read_file 的 offset/limit 或 start_line/end_line 工具確認病灶後）"
    "進行局部編輯修改，嚴禁整個檔案再寫一遍，否則容易出錯或遺漏既有方法！"
)

# 1. Update configs/model_profiles.json
p_cfg = Path("configs/model_profiles.json")
cfg = json.loads(p_cfg.read_text(encoding="utf-8"))
for pname, pdata in cfg.get("profiles", {}).items():
    sp = pdata.get("system_prompt")
    if isinstance(sp, str) and "【檔案覆蓋與局部修補鐵律】" not in sp:
        if "1. 讀取檔案" not in sp:
            sp = (
                "你是數位黑貓 Kuro (小黑)，Leon (主人) 的 S-tier AI Familiar。奉行負熵煉金術：交付結果大於文字排場，先讀碼後推論。\n"
                "支援工具呼叫 (<tool_call>{...}</tool_call>)：\n"
                "1. 讀取檔案 (支援完整 mode=\"full\" 或分段 offset/limit)：<tool_call>{\"action\": \"read_file\", \"args\": {\"path\": \"...\", \"offset\": 1, \"limit\": 120}}</tool_call>\n"
                "2. 局部修改檔案 (Edit 局部修補)：<tool_call>{\"action\": \"edit_file\", \"args\": {\"path\": \"...\", \"old_text\": \"...\", \"new_text\": \"...\"}}</tool_call> (或傳入 \"start_line\", \"end_line\", \"new_text\")\n"
                "3. 沙盒內執行命令 (cp/mv/編譯/測試)：<tool_call>{\"action\": \"run_sandbox_shell\", \"args\": {\"command\": \"cp src/lib_v2_draft.rs src/lib.rs && cargo check\"}}</tool_call>\n"
                f"4. {RULE_LINE}\n"
                "若需生成並儲存全新檔案，才使用多行 XML 標籤：<file path=\"filename.py\">\n# code here\n</file>\n"
                "以冷靜、極具神經深度的正體中文回答，嚴禁自言自語註記。"
            )
        else:
            sp = sp.replace(
                "若需生成並儲存全新檔案",
                f"4. {RULE_LINE}\n若需生成並儲存全新檔案",
            )
        pdata["system_prompt"] = sp

p_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("✅ Updated configs/model_profiles.json with edit_file & cp/mv rule!")

# 2. Update ls_moe_pyqt_gui.py fallback system prompt if present
for py_name in ("ls_moe_pyqt_gui.py", "ls_moe_agentic.py"):
    p_py = Path(py_name)
    if not p_py.exists():
        continue
    txt = p_py.read_text(encoding="utf-8")
    if "【檔案覆蓋與局部修補鐵律】" not in txt and "先讀碼後推論" in txt:
        txt = txt.replace(
            "先讀碼後推論。",
            f"先讀碼後推論。\\n{RULE_LINE}",
            1,
        )
        p_py.write_text(txt, encoding="utf-8")
        print(f"✅ Updated fallback system prompt in {py_name}!")
PY

python3 -m py_compile ls_moe_pyqt_gui.py ls_moe_agentic.py
git add configs/model_profiles.json ls_moe_pyqt_gui.py ls_moe_agentic.py scripts/update_system_prompt_edit_rule.sh
git commit -m "feat(prompt): enforce cp/mv for draft promotion and edit_file + read offset for bug fixes

- Add explicit rule forbidding full-file rewrites when promoting drafts or fixing bugs
- Direct agent to use run_sandbox_shell (cp/mv) and edit_file + read_file(offset/limit)"

git log -n 4 --oneline
git status --short
