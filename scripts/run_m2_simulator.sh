#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --quiet mujoco numpy
else
    source .venv/bin/activate
fi

cd "Milestone 02/code"
exec mjpython sim_bridge.py
