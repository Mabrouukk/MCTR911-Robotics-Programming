#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../docker"

docker rm -f mctr911_m2 >/dev/null 2>&1 || true

exec docker compose run --rm --name mctr911_m2 robotics bash -lc '
    source /opt/ros/humble/setup.bash
    cd "/workspace/Milestone 02/code"
    python3 -u mujoco_bridge_node.py
'
