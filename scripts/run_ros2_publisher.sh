#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../docker"

docker rm -f mctr911_demo >/dev/null 2>&1 || true

exec docker compose run --rm --name mctr911_demo robotics bash -lc "
    source /opt/ros/humble/setup.bash
    python3 /workspace/ros2_nodes/pick_place_publisher.py
"
