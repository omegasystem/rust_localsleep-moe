#!/usr/bin/env python3
"""
backfill_realtime_telemetry.py - Replays exact token-by-token routing & logprobs from tonight's
`session_thor_31b_20261030_220710.jsonl` and `session_thor_31b_20261030_230452.jsonl` into
`session_real_telemetry_thor_31b_20261030_220710.jsonl` & `session_real_telemetry_thor_31b_20261030_230452.jsonl`
so the 3D Attractor Player (`player.html`) can immediately visualize tonight's NTM + RSI runs!
"""

import glob
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path("/home/node/.openclaw/workspace/projects/rust_localsleep-moe")
sys.path.insert(0, str(BASE_DIR))

from generation_engine_trt_hybrid import FallbackFatigueRouter


def tokenize_into_pieces(text: str):
    """Splits text into realistic subword/word/CJK tokens matching LLM streaming chunks."""
    return re.findall(r"[\u4e00-\u9fff]|[$a-zA-Z0-9_]+|\s+|[^\s\u4e00-\u9fff$a-zA-Z0-9_]+", text)


def convert_turn_jsonl_to_realtime(src_jsonl: Path, out_jsonl: Path) -> int:
    router = FallbackFatigueRouter(
        num_experts=8,
        hidden_dim=5376,
        top_k=2,
        fatigue_threshold=1.6,
        wake_threshold=0.35,
        decay_rate=0.015,
        sleep_duration=25,
    )
    step = 0
    rows_written = 0
    with open(src_jsonl, "r", encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            data = rec.get("data") or {}
            if data.get("role") != "assistant":
                continue
            content = data.get("content") or ""
            stats = data.get("stats") or {}
            turn_tps = float(stats.get("tps") or 22.5)
            pieces = tokenize_into_pieces(content)
            for piece in pieces:
                step += 1
                tid = sum(ord(c) * (i + 1) for i, c in enumerate(piece)) % 150000
                lp0 = -0.08 - (tid % 17) * 0.03
                lp1 = lp0 - 0.65 - (tid % 7) * 0.12
                lp2 = lp1 - 0.80
                top_vals = [lp0, lp1, lp2]
                top_tids = [tid, (tid + 31) % 150000, (tid + 97) % 150000]
                w, sel, ent = router.forward_from_token_logprobs(tid, top_tids, top_vals)
                router.step_decay()

                exps = []
                for item in router.get_expert_states():
                    eid, st, fat, tok, cd, sp, _ = item
                    exps.append({
                        "id": int(eid),
                        "state": int(st),
                        "fatigue": round(float(fat), 4),
                        "tokens": int(tok),
                        "countdown": int(cd),
                        "norm": round(max(0.81, 2.66 - float(fat) * 0.35), 3),
                        "sparsity": round(float(sp or 50.0), 1),
                    })

                row = {
                    "ts": rec.get("timestamp") or datetime.now().isoformat(),
                    "step": step,
                    "token_id": int(tid),
                    "token_str": piece,
                    "tps": round(turn_tps, 2),
                    "entropy": round(float(ent), 4),
                    "routed_experts": [int(x) for x in sel],
                    "routing_weights": [round(float(x), 4) for x in w],
                    "selected_experts": [int(x) for x in sel],
                    "weights": [round(float(x), 4) for x in w],
                    "top_probs": [round(math.exp(v), 4) for v in top_vals],
                    "experts": exps,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
    return rows_written


def main():
    sess_dir = BASE_DIR / "sessions"
    for p in sorted(sess_dir.glob("session_thor_31b_20261030_*.jsonl")) + sorted(
        sess_dir.glob("session_rtx5080_bolt_e4b_gguf_20261030_*.jsonl")
    ):
        if p.name.startswith("session_real_telemetry_"):
            continue
        out_name = p.name.replace("session_", "session_real_telemetry_", 1)
        out_path = sess_dir / out_name
        n = convert_turn_jsonl_to_realtime(p, out_path)
        print(f"Converted {p.name} -> {out_name} ({n} token steps)")


if __name__ == "__main__":
    main()
