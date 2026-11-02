#!/usr/bin/env bash
set -euo pipefail

cd /home/node/.openclaw/workspace/projects/rust_localsleep-moe

python3 - <<'PY'
from pathlib import Path

p = Path("generation_engine_trt_hybrid.py")
txt = p.read_text(encoding="utf-8")

old_md = 'f"NREM 2:4 Prunes: `{stats.get(\'gpu_24_prunes\', 0)}`\\n\\n"'
new_md = 'f"NREM 2:4 Prunes: `{stats.get(\'gpu_24_prunes\', 0)}` | Bifurcation Flips: `{stats.get(\'bifurcation_flips\', 0)}`\\n\\n"'
if old_md in txt:
    txt = txt.replace(old_md, new_md, 1)
    p.write_text(txt, encoding="utf-8")
    print("✅ Added Bifurcation Flips to SessionLogger .md output!")
PY

python3 -m py_compile generation_engine_trt_hybrid.py ls_moe_pyqt_gui.py

git add generation_engine_trt_hybrid.py scripts/commit_bifurcation_fix.sh
git commit -m "fix(telemetry): reconstruct pre-bias raw logits (z_raw = z_tilde - g) in Bifurcation Flip Counter

- Subtract active live_gate bias g_t from SSE top_logprobs to recover true pre-bias raw logits
- Also display Bifurcation Flips directly in SessionLogger markdown (.md) metrics line"

git log -n 4 --oneline
git status --short
