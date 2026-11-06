#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PORT="${PORT:-18935}"
HOST="${HOST:-0.0.0.0}"
if [ -x "./.venv_rust/bin/python" ]; then
    PY_BIN="./.venv_rust/bin/python"
else
    PY_BIN="python3"
fi
echo "🐈‍⬛ Starting LocalSleep-MoE Remote Web UI & OpenAI API on http://${HOST}:${PORT} ..."
exec "$PY_BIN" ls_moe_web_ui.py --host "$HOST" --port "$PORT" "$@"
