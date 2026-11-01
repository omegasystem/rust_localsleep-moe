#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/node/.openclaw/workspace/projects/rust_localsleep-moe"
EXP_DIR="${REPO_DIR}/experiments/gpt1900_time_capsule_20261101"

mkdir -p "${EXP_DIR}"

cd "${REPO_DIR}"

# 1. Back up the GPT-1900 experimental versions both in root (*_gpt1900_exp.*) and in experiments/gpt1900_time_capsule_20261101/
cp -v ls_moe_pyqt_gui.py ls_moe_pyqt_gui_gpt1900_exp.py
cp -v generation_engine_trt_hybrid.py generation_engine_trt_hybrid_gpt1900_exp.py
cp -v configs/model_profiles.json configs/model_profiles_gpt1900_exp.json
cp -v snapshot_studio.html snapshot_studio_gpt1900_exp.html

# Wire ls_moe_pyqt_gui_gpt1900_exp.py to use generation_engine_trt_hybrid_gpt1900_exp and configs/model_profiles_gpt1900_exp.json
sed -i 's/from generation_engine_trt_hybrid import/from generation_engine_trt_hybrid_gpt1900_exp import/g' ls_moe_pyqt_gui_gpt1900_exp.py
sed -i 's|configs/model_profiles.json|configs/model_profiles_gpt1900_exp.json|g' ls_moe_pyqt_gui_gpt1900_exp.py

# Copy all GPT-1900 artifacts into the dedicated experiment archive folder
cp -v ls_moe_pyqt_gui_gpt1900_exp.py "${EXP_DIR}/"
cp -v generation_engine_trt_hybrid_gpt1900_exp.py "${EXP_DIR}/"
cp -v configs/model_profiles_gpt1900_exp.json "${EXP_DIR}/"
cp -v snapshot_studio_gpt1900_exp.html "${EXP_DIR}/"
mv -v ignite_gpt1900_server.py "${EXP_DIR}/" 2>/dev/null || true
mv -v ignite_gpt1900_server.sh "${EXP_DIR}/" 2>/dev/null || true

# Move GPT-1900 checkpoints into the experiment folder so main checkpoints/ stays 100% clean for Gemma-4 31B
mkdir -p "${EXP_DIR}/checkpoints"
for ck in checkpoints/*gpt1900*.safetensors; do
    if [[ -f "${ck}" ]]; then
        mv -v "${ck}" "${EXP_DIR}/checkpoints/"
    fi
done

# Copy the two GPT-1900 experimental session logs into the experiment archive as well
mkdir -p "${EXP_DIR}/sessions"
for slog in sessions/session_thor_31b_20261101_223332.md sessions/session_thor_31b_20261101_224612.md; do
    if [[ -f "${slog}" ]]; then
        cp -v "${slog}" "${EXP_DIR}/sessions/"
    fi
done

# 2. Restore the main production files back to the clean Gemma-4 31B flagship state (HEAD)
git checkout HEAD -- ls_moe_pyqt_gui.py generation_engine_trt_hybrid.py configs/model_profiles.json snapshot_studio.html

echo "✅ Successfully archived GPT-1900 experiment and restored clean Gemma-4 31B main files!"
git status --short
