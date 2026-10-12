#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    echo "No .venv found - setting it up (first run only)..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --quiet mujoco numpy
else
    source .venv/bin/activate
fi

exec mjpython scripts/test_mujoco_viewer.py
