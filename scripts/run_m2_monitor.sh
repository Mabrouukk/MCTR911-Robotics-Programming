#!/usr/bin/env bash
# Usage: scripts/run_m2_monitor.sh [topic]   (default: /ur5e/ee_position, or /ur5e/joint_states)
set -euo pipefail

if ! docker ps --format '{{.Names}}' | grep -q '^mctr911_m2$'; then
    echo "Error: the ROS2 bridge isn't running. Start scripts/run_m2_bridge.sh first."
    exit 1
fi

TTY_FLAGS=-i
[ -t 0 ] && TTY_FLAGS=-it

exec docker exec $TTY_FLAGS mctr911_m2 bash -lc '
    source /opt/ros/humble/setup.bash
    exec ros2 topic echo "$1"
' _ "${1:-/ur5e/ee_position}"
