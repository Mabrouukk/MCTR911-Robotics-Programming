#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../docker"

if ! docker ps --format '{{.Names}}' | grep -q '^mctr911_demo$'; then
    echo "Error: no running 'mctr911_demo' container found."
    echo "Run scripts/run_ros2_publisher.sh first, in another terminal, and leave it running."
    exit 1
fi

exec docker compose exec robotics bash -lc "
    source /opt/ros/humble/setup.bash
    python3 /workspace/ros2_nodes/pick_place_subscriber.py
"
