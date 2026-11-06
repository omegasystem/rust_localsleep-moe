#!/usr/bin/env bash
set -euo pipefail
cd /home/node/.openclaw/workspace/projects/rust_localsleep-moe

git status --short

git add \
  configs/model_profiles.json \
  generation_engine_trt_hybrid.py \
  ls_moe_pyqt_gui.py \
  ls_moe_agentic.py \
  tools_harness.py \
  .cargo/config.toml \
  scripts/sync_rust_env_and_crates.sh

git commit -m "feat(agentic-sandbox): unlock MAX_TOOL_CALLS=100, cross-project copy bypass, PyO3 3.14 ABI & batteries-included Cargo crates"

git log -n 4 --oneline
