#!/usr/bin/env bash
# Usage:
#   scripts/run_m2_commands.sh constant [--deg q1 q2 q3 q4 q5 q6] [--gripper 0-255]
#   scripts/run_m2_commands.sh sine [--amp-deg 20] [--freq 0.2]
#   scripts/run_m2_commands.sh sequence [--period 2.5]
#   scripts/run_m2_commands.sh ik --xyz X Y Z [--gripper 0-255]
set -euo pipefail

if ! docker ps --format '{{.Names}}' | grep -q '^mctr911_m2$'; then
    echo "Error: the ROS2 bridge isn't running. Start scripts/run_m2_bridge.sh first (and scripts/run_m2_simulator.sh before that)."
    exit 1
fi

TTY_FLAGS=-i
[ -t 0 ] && TTY_FLAGS=-it

exec docker exec $TTY_FLAGS mctr911_m2 bash -lc '
    source /opt/ros/humble/setup.bash
    cd "/workspace/Milestone 02/code"
    exec python3 -u joint_command_publisher.py "$@"
' _ "$@"
