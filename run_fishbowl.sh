#!/usr/bin/env bash
# SOMA-Fishbowl Launcher (Independent PyQt6 GUI + Network Control API on :18940)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.cargo/bin:$PATH"

if [ -x "$SCRIPT_DIR/.venv_rust/bin/python" ]; then
    PY_BIN="$SCRIPT_DIR/.venv_rust/bin/python"
else
    PY_BIN="python3"
fi

echo "🐟 Starting SOMA-Fishbowl (PyQt6 Tactical Aquarium GUI + Network API :18940)..."
exec "$PY_BIN" "$SCRIPT_DIR/fishbowl_pyqt_gui.py" --host 0.0.0.0 --port 18940 "$@"
