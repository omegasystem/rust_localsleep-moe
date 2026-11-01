#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="/home/node/.openclaw/workspace/projects/rust_localsleep-moe/checkpoints"
BACKUP_DIR="${CKPT_DIR}/backup_31b/polluted_by_gpt1900_20261101"
GOLDEN_CKPT="${CKPT_DIR}/checkpoint_thor_31b_16lora_20261101_155418_406511.safetensors"

mkdir -p "${BACKUP_DIR}"

for f in \
    "checkpoint_thor_31b_16lora_20261101_223641_9787.safetensors" \
    "checkpoint_thor_31b_16lora_20261101_225010_15009.safetensors" \
    "checkpoint_thor_31b_16lora_20261101_225222_15009.safetensors" \
    "checkpoint_thor_31b_16lora_20261101_230137_20480.safetensors"
do
    if [[ -f "${CKPT_DIR}/${f}" ]]; then
        mv -v "${CKPT_DIR}/${f}" "${BACKUP_DIR}/${f}"
    fi
done

cp -v "${GOLDEN_CKPT}" "${CKPT_DIR}/latest_thor_31b_16lora.safetensors"
echo "✅ Restored golden 15:54:18 checkpoint (406,511 tokens) to latest_thor_31b_16lora.safetensors!"
ls -lh "${CKPT_DIR}/latest_thor_31b_16lora.safetensors" "${GOLDEN_CKPT}"
