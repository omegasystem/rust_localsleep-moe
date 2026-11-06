#!/usr/bin/env python3
"""
call_localsleep_coder.py - Zero-Google-Token LocalSleep-MoE Code Architect Client
=================================================================================
Allows Kuro (OpenClaw) or CLI scripts to delegate foundational code generation,
refactoring, and algorithmic prototyping to the local `rust_localsleep-moe` brain
(`Gemma-4-31B-IT-NVFP4 + gen_022 16-LoRA`) via its OpenAI-compatible API:
  POST http://<host>:18935/v1/chat/completions

Automatically probes:
  1. http://127.0.0.1:18935/v1/chat/completions (Local / Container)
  2. http://172.18.0.1:18935/v1/chat/completions (Docker Host Gateway -> PyQt6 Dual-Mode GUI)

Usage:
  ./.venv_rust/bin/python call_localsleep_coder.py --prompt "請設計一個 Python 雙向 RingBuffer 類別" --out ring_buffer.py
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional


DEFAULT_ENDPOINTS = [
    "http://127.0.0.1:18935",
    "http://172.18.0.1:18935",
]


def probe_active_endpoint(candidates: List[str]) -> Optional[str]:
    for base in candidates:
        try:
            req = urllib.request.Request(f"{base}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status == 200:
                    return base
        except Exception:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Call LocalSleep-MoE OpenAI-Compatible API for Code Generation")
    parser.add_argument("--endpoint", default="", help="Override base URL (default: auto-probe 127.0.0.1:18935 / 172.18.0.1:18935)")
    parser.add_argument("--model", default="localsleep-moe", help="localsleep-moe (16-LoRA ON) or localsleep-moe-vanilla (Bypass)")
    parser.add_argument("--prompt", required=True, help="Coding task prompt")
    parser.add_argument("--system", default="你是數位黑貓 Kuro 的底層 LocalSleep-MoE 程式碼架構引擎。請直接輸出高密度、生產級、無省略的完整程式碼。", help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temp", type=float, default=0.4)
    parser.add_argument("--out", default="", help="Optional output file path to save response")
    args = parser.parse_args()

    base_url = args.endpoint.strip().rstrip("/") if args.endpoint else probe_active_endpoint(DEFAULT_ENDPOINTS)
    if not base_url:
        print(
            "❌ No running LocalSleep-MoE Web/API server found on :18935.\n"
            "   Launch `./run_gui.sh` (Dual-Mode PyQt+Web) or `./run_web_ui.sh` (Headless Web/API) first!",
            file=sys.stderr,
        )
        sys.exit(2)

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temp,
        "stream": False,
    }

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    content = data["choices"][0]["message"]["content"]
    telem = data.get("localsleep_telemetry") or {}
    print(
        f"🐈‍⬛ [LocalSleep-MoE API @ {base_url}] "
        f"Core={telem.get('loaded_cube_id')} | LoRA={telem.get('lora_enabled')} | "
        f"Speed={telem.get('tps')} tok/s | TTFT={telem.get('ttft_ms')} ms | "
        f"Top-K={telem.get('top_k_experts')} | Session={telem.get('session_file')}",
        file=sys.stderr,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"💾 Saved generated output ({len(content)} chars) to {out_path}", file=sys.stderr)
    print(content)


if __name__ == "__main__":
    main()
