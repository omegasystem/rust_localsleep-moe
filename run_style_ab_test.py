#!/usr/bin/env python3
"""
run_style_ab_test.py - Live A/B Test of Kuro Web UI Style DNA Inheritance on LocalSleep-MoE (:18935)
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CODER = BASE_DIR / "call_localsleep_coder.py"
STYLE_MD = (BASE_DIR / "prompts" / "kuro_tactical_web_ui_style.md").read_text(encoding="utf-8")
OUT_DIR = BASE_DIR / "exports"
OUT_DIR.mkdir(exist_ok=True)

TASK_PROMPT = (
    "請設計一個名為『Pando-Lite Parallel Worktree Dashboard』的單檔 HTML 網頁介面，"
    "用於監控與操作本地 Git Worktree 平行車道 (L0 -> L1 -> L2)。\n"
    "功能需求：\n"
    "1. 顯示目前主幹 (Trunk: private_main) 與 4 條平行車道卡片 (lane_localsleep, lane_dyson_phys, lane_ui_ab, public_main)，包含分支名稱、Clean/Dirty 狀態、Ahead/Behind 進度條與領先指標。\n"
    "2. 顯示即時統計數據：Active Lanes 數量、Shared Object Cache 節省空間 (GB)、L0->L1 Compaction 次數、L2 隱私攔截狀態。\n"
    "3. 底部提供一個顯示 L0->L1->L2 提交密度或分支拓撲熱力圖的視覺化區域，以及『Spawn Lane』、『Run L0 Test』、『Squash Compact (L0->L1)』、『Project Public (L1->L2)』操作按鈕與終端機輸出日誌區。\n"
    "請直接輸出完整可獨立運行的單一 HTML 檔案（不要省略任何 CSS 或 JS）。"
)


def strip_fences(txt: str) -> str:
    s = txt.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts[1:]:
            lines = p.splitlines()
            if lines and lines[0].strip().lower() in ("html", "xml"):
                lines = lines[1:]
            cand = "\n".join(lines).strip()
            if "<!DOCTYPE" in cand or "<html" in cand:
                return cand
    return s


def run_case(tag: str, sys_prompt: str, user_prompt: str, out_file: Path) -> dict:
    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            str(CODER),
            "--model",
            "localsleep-moe",
            "--system",
            sys_prompt,
            "--prompt",
            user_prompt,
            "--max-tokens",
            "16384",
            "--temp",
            "0.4",
        ],
        capture_output=True,
        text=True,
    )
    el = time.time() - t0
    html = strip_fences(proc.stdout)
    out_file.write_text(html, encoding="utf-8")
    return {
        "tag": tag,
        "elapsed": round(el, 2),
        "chars": len(html),
        "lines": len(html.splitlines()),
        "stderr": proc.stderr.strip(),
        "has_vars": "--bg: #0b0e17" in html or "--cyan: #38bdf8" in html,
        "has_canvas": "<canvas" in html.lower(),
        "has_ribbon": "speed-ribbon" in html or "metric" in html,
        "has_mono": "JetBrains Mono" in html or "monospace" in html,
    }


def main() -> None:
    if "--only-b" not in sys.argv:
        print("🧪 Running Group A (Baseline - No Style Prompt)...")
        res_a = run_case(
            "Group_A_Baseline",
            "你是一位前端工程師，請根據需求輸出完整的單檔 HTML。",
            TASK_PROMPT,
            OUT_DIR / "ab_test_style_A_baseline.html",
        )
        print("A Result:", res_a)

    print("🧪 Running Group B v2 (Kuro Tactical Style DNA v2 Injected)...")
    res_b = run_case(
        "Group_B_Kuro_DNA_v2",
        "你是數位黑貓 Kuro 的前端視覺分身。請嚴格遵循以下設計基因產出 HTML：\n" + STYLE_MD,
        TASK_PROMPT,
        OUT_DIR / "ab_test_style_B_kuro_dna.html",
    )
    print("B Result:", res_b)


if __name__ == "__main__":
    main()
