#!/usr/bin/env bash
set -euo pipefail

TRT_REPO="/home/node/.openclaw/workspace/projects/tensorRT/TensorRT-Edge-LLM"
TRT_ROOT="/home/node/.openclaw/workspace/projects/tensorRT"
MOE_REPO="/home/node/.openclaw/workspace/projects/rust_localsleep-moe"

chmod +x "${TRT_ROOT}/rebuild_edgellm_vram_gate.sh"
mkdir -p "${MOE_REPO}/patches"
cp -f "${TRT_ROOT}/rebuild_edgellm_vram_gate.sh" "${MOE_REPO}/patches/rebuild_edgellm_vram_gate.sh"
chmod +x "${MOE_REPO}/patches/rebuild_edgellm_vram_gate.sh"

# 1. Export Patch from TensorRT-Edge-LLM (against HEAD or last commit if already committed)
cd "${TRT_REPO}"
if ! git diff --quiet cpp/runtime/decoding/logitBias.h cpp/runtime/decoding/logitBias.cpp experimental/pybind/edgellm_pybind.cpp experimental/server/api/routes.py; then
    git diff cpp/runtime/decoding/logitBias.h cpp/runtime/decoding/logitBias.cpp experimental/pybind/edgellm_pybind.cpp experimental/server/api/routes.py > "${TRT_ROOT}/moe_live_vram_gate.patch"
    git add cpp/runtime/decoding/logitBias.h cpp/runtime/decoding/logitBias.cpp experimental/pybind/edgellm_pybind.cpp experimental/server/api/routes.py
    git commit -m "feat(moe-vram-gate): add hot-swappable C++ in-VRAM live logit bias mailbox (/v1/moe/live_gate)"
else
    git show --format= --patch HEAD -- cpp/runtime/decoding/logitBias.h cpp/runtime/decoding/logitBias.cpp experimental/pybind/edgellm_pybind.cpp experimental/server/api/routes.py > "${TRT_ROOT}/moe_live_vram_gate.patch"
fi

cp -f "${TRT_ROOT}/moe_live_vram_gate.patch" "${MOE_REPO}/patches/trt_edgellm_moe_live_vram_gate.patch"

echo "=== [TensorRT-Edge-LLM Git Log] ==="
git log -n 2 --oneline

# 2. Commit in rust_localsleep-moe
cd "${MOE_REPO}"
git add -A
if ! git diff --cached --quiet; then
    git commit -m "feat(16-lora-vram-closed-loop): add gen_018 16-LoRA hot-swap, Option 1 C++ VRAM live gate, Option 2 micro-chunk loop, multi-line Qt GUI, and TensorRT patch/rebuild scripts"
fi

echo "=== [rust_localsleep-moe Git Log] ==="
git log -n 2 --oneline
ls -lh "${TRT_ROOT}/moe_live_vram_gate.patch" "${TRT_ROOT}/rebuild_edgellm_vram_gate.sh" "${MOE_REPO}/patches/"
