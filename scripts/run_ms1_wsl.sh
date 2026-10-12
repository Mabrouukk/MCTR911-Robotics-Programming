#!/usr/bin/env bash
# Milestone 1 demo natively on WSL/Linux (no Docker): MuJoCo viewer + ROS2 pick-and-place
# publisher/subscriber. Requires ROS2 (default /opt/ros/jazzy) and `pip install mujoco numpy`.
# Press Ctrl+C to stop everything.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
VIEWER_PID=""
PUB_PID=""

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Milestone 1..."
    [ -n "$VIEWER_PID" ] && kill "$VIEWER_PID" 2>/dev/null || true
    [ -n "$PUB_PID" ] && kill "$PUB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

[ -f "$ROS_SETUP" ] || { echo "ROS2 not found at $ROS_SETUP"; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"
python3 -c "import mujoco, numpy" 2>/dev/null || { echo "Missing python deps: pip install mujoco numpy"; exit 1; }

cd "$ROOT"

echo "[1/3] Opening the MuJoCo viewer (UR5e + Robotiq gripper)"
python3 scripts/test_mujoco_viewer.py >/dev/null 2>&1 &
VIEWER_PID=$!

echo "[2/3] Starting the ROS2 publisher"
python3 -u ros2_nodes/pick_place_publisher.py &
PUB_PID=$!

echo "[3/3] Starting the ROS2 subscriber (Ctrl+C to stop everything)"
python3 -u ros2_nodes/pick_place_subscriber.py
