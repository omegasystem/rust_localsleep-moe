#!/usr/bin/env bash
# ==============================================================================
# ignite_gpt1900_server.sh
# Launch GPT-1900 (3.29B) on Local RTX 5080 with Live VRAM Mailbox (/v1/moe/live_gate)
# Default Port: 8002 (so it never conflicts with Thor 8000 or llama.cpp 8001)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-/home/omegacd/.openclaw/workspace/models/gpt1900-instruct-v3-sft}"
PORT="${PORT:-8002}"

echo "🕰️ Igniting GPT-1900 (Machina Mirabilis 3.29B) Server on Port ${PORT}..."
echo "   Model Dir : ${MODEL_DIR}"
echo "   Endpoints : http://127.0.0.1:${PORT}/v1/chat/completions"
echo "               http://127.0.0.1:${PORT}/v1/moe/live_gate"

exec python3 "${SCRIPT_DIR}/ignite_gpt1900_server.py" --model-dir "${MODEL_DIR}" --port "${PORT}"
