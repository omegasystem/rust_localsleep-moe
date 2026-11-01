#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PORT="${PORT:-18930}"
echo "🔬 Starting LocalSleep-MoE SafeTensors Synaptic Studio & 3D Attractor Web UI on http://0.0.0.0:${PORT}"
exec python3 snapshot_studio_web.py --port "$PORT"
