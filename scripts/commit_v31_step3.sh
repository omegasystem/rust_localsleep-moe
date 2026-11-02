#!/usr/bin/env bash
set -euo pipefail

cd /home/node/.openclaw/workspace/projects/rust_localsleep-moe

python3 -m py_compile expert_pool_lora.py generation_engine_trt_hybrid.py ls_moe_pyqt_gui.py

git add expert_pool_lora.py generation_engine_trt_hybrid.py ls_moe_pyqt_gui.py scripts/commit_v31_step3.sh
git commit -m "feat(v3.1-step3): add Hybrid Dual-Scale Decay, FWHT 4x4 Hadamard Dispersion, and Bifurcation Flip Counter

- Implement Hybrid Dual-Scale Eligibility Trace Decay (gamma_token=0.9985 * gamma_syntax=0.935 on syntax boundaries) to eliminate token-length verbosity bias
- Add 4x4 Orthonormal Fast Walsh-Hadamard Dispersion (apply_incoherent_hadamard_dispersion) before Triton 2:4 pruning and quad-dispersed strided LTP (quad_offset::16) + crest-factor guard to prevent dark preview slices and 4-hot-channel 2:4 collisions
- Add Saddle-Node Bifurcation Flip Counter (argmax(z+g) != argmax(z)) to quantify deterministic trajectory switching under Eagle3 Greedy decoding"

git log -n 4 --oneline
git status --short
