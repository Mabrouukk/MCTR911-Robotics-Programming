#!/usr/bin/env bash
# Milestone 2 natively on WSL/Linux (no Docker): MuJoCo simulator + ROS2 bridge + one joint input.
#
#   scripts/run_ms2_wsl.sh                      colour sorting (default)
#   scripts/run_ms2_wsl.sh sine                 sine wave on every joint
#   scripts/run_ms2_wsl.sh constant             hold home pose (or: constant --deg q1 q2 q3 q4 q5 q6)
#   scripts/run_ms2_wsl.sh ik --xyz X Y Z       move the gripper flange to a world point
#   scripts/run_ms2_wsl.sh sliders              no ROS2 input: use the viewer's Control panel
#
# Requires: ROS2 (default /opt/ros/jazzy; override with ROS_DISTRO) and `pip install mujoco numpy`.
# Press Ctrl+C to stop everything.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="$ROOT/Milestone 02/code"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
SIM_PID=""
BRIDGE_PID=""
[ $# -eq 0 ] && set -- sort

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Milestone 2..."
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
    [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

[ -f "$ROS_SETUP" ] || { echo "ROS2 not found at $ROS_SETUP"; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"
python3 -c "import mujoco, numpy" 2>/dev/null || { echo "Missing python deps: pip install mujoco numpy"; exit 1; }

export SIM_HOST=127.0.0.1
cd "$CODE"

echo "[1/3] Opening the MuJoCo simulator (UR5e + gripper + colour sorting cell)"
PYTHONUNBUFFERED=1 python3 sim_bridge.py > >(sed 's/^/[simulator] /') 2>&1 &
SIM_PID=$!

echo "[2/3] Starting the ROS2 bridge"
python3 -u mujoco_bridge_node.py &
BRIDGE_PID=$!
for _ in $(seq 60); do sleep 1; kill -0 "$BRIDGE_PID" 2>/dev/null || { echo "Bridge exited"; exit 1; }; [ "$_" -ge 3 ] && break; done

if [ "$1" = "sliders" ]; then
    echo "[3/3] No ROS2 input running: drag the joint sliders in the viewer's Control panel (Ctrl+C to stop)"
    while kill -0 "$SIM_PID" 2>/dev/null; do sleep 1; done
    exit 0
fi

echo "[3/3] Input: $* (Ctrl+C to stop everything)"
python3 -u joint_command_publisher.py "$@"
