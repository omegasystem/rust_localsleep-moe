#!/usr/bin/env bash
# ==============================================================================
# rebuild_edgellm_vram_gate.sh
# ------------------------------------------------------------------------------
# TensorRT-Edge-LLM Hot-Swappable C++ VRAM Live MoE Mailbox Rebuild & Sync Script
#
# Modified Files in TensorRT-Edge-LLM:
#   1. cpp/runtime/decoding/logitBias.h
#   2. cpp/runtime/decoding/logitBias.cpp
#   3. experimental/pybind/edgellm_pybind.cpp
#   4. experimental/server/api/routes.py
#
# Usage:
#   - On AGX Thor (Local Rebuild):
#       ./rebuild_edgellm_vram_gate.sh
#   - From 5080 Host (Sync 4 files to Thor + Remote Incremental Rebuild):
#       ./rebuild_edgellm_vram_gate.sh --sync-to-thor
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRT_REPO="${SCRIPT_DIR}/TensorRT-Edge-LLM"

THOR_HOST="${THOR_HOST:-omegacd@192.168.50.67}"
THOR_TRT_REPO="${THOR_TRT_REPO:-/home/omegacd/local_projects/tensorRT/TensorRT-Edge-LLM}"

MODIFIED_FILES=(
    "cpp/runtime/decoding/logitBias.h"
    "cpp/runtime/decoding/logitBias.cpp"
    "experimental/pybind/edgellm_pybind.cpp"
    "experimental/server/api/routes.py"
)

if [[ "${1:-}" == "--sync-to-thor" ]]; then
    echo "🚀 [1/2] Syncing 4 modified C++/Pybind/Server files to AGX Thor (${THOR_HOST}:${THOR_TRT_REPO})..."
    for rel_path in "${MODIFIED_FILES[@]}"; do
        echo "   -> scp ${rel_path}"
        scp "${TRT_REPO}/${rel_path}" "${THOR_HOST}:${THOR_TRT_REPO}/${rel_path}"
    done
    scp "${SCRIPT_DIR}/moe_live_vram_gate.patch" "${THOR_HOST}:${THOR_TRT_REPO}/moe_live_vram_gate.patch" || true

    echo "🔨 [2/2] Triggering incremental C++ pybind build (_edgellm_runtime) on AGX Thor..."
    ssh "${THOR_HOST}" "cd ${THOR_TRT_REPO} && cmake --build build --target _edgellm_runtime -j\$(nproc) && cp -v build/pybind/_edgellm_runtime*.so ./ && ls -lh _edgellm_runtime*.so"
    echo "✅ Done! Restart ./ignite_31B_Assis.sh on Thor to activate."
    exit 0
fi

echo "🔨 [Local Build on Thor] Building _edgellm_runtime in ${TRT_REPO}..."
cd "${TRT_REPO}"

if [[ -d "build" ]]; then
    cmake --build build --target _edgellm_runtime -j"$(nproc)"
    cp -v build/pybind/_edgellm_runtime*.so ./
else
    echo "⚠️ No existing build/ directory found; running setup_pybind.py build_ext --inplace..."
    AARCH64_BUILD=1 python3 experimental/server/setup_pybind.py build_ext --inplace
fi

echo "✅ Rebuild complete! Verify endpoint after launching ./ignite_31B_Assis.sh:"
echo "   curl -s http://127.0.0.1:8000/v1/moe/live_gate"
