#!/usr/bin/env bash
# Challenge 1 (Franka Panda FK + IK) natively on WSL/Linux (no Docker).
#
#   scripts/run_ch1_wsl.sh            MuJoCo simulator + ROS2 bridge + panda_kinematics node (default)
#   scripts/run_ch1_wsl.sh verify     only check FK and IK against MuJoCo, no simulator window, no ROS2
#
# Requires: ROS2 + colcon (default /opt/ros/jazzy; override with ROS_DISTRO) and
# `pip install mujoco numpy matplotlib`. Press Ctrl+C to stop everything.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CH="$ROOT/Challenge 01"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
MODE="${1:-run}"
SIM_PID=""
BRIDGE_PID=""
WS="$(mktemp -d /tmp/ch1_ws.XXXXXX)"

python3 -c "import mujoco, numpy, matplotlib" 2>/dev/null || { echo "Missing python deps: pip install mujoco numpy matplotlib"; exit 1; }

if [ "$MODE" = "verify" ]; then
    rm -rf "$WS"
    exec python3 "$CH/verify/verify_kinematics.py"
fi

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Challenge 1..."
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
    [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" 2>/dev/null || true
    rm -rf "$WS"
}
trap cleanup EXIT INT TERM

[ -f "$ROS_SETUP" ] || { echo "ROS2 not found at $ROS_SETUP"; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"
command -v colcon >/dev/null || { echo "colcon not found: sudo apt install python3-colcon-common-extensions"; exit 1; }

export SIM_HOST=127.0.0.1

echo "[1/4] Opening the MuJoCo simulator (Panda at its home pose, RGB frame = target)"
PYTHONUNBUFFERED=1 python3 "$CH/sim/panda_sim.py" > >(sed 's/^/[simulator] /') 2>&1 &
SIM_PID=$!

echo "[2/4] Starting the ROS2 bridge (/joint_commands -> simulator -> /joint_state)"
(cd "$CH/sim" && exec python3 -u panda_bridge_node.py) 2>&1 | sed -u 's/^/[bridge] /' &
BRIDGE_PID=$!
sleep 3

echo "[3/4] Building the panda_kinematics package (colcon, in $WS)"
# Built in a temp dir so no build/ install/ log/ folders land in the repo.
mkdir -p "$WS/src"
cp -r "$CH/panda_kinematics" "$WS/src/"
(cd "$WS" && colcon build --packages-select panda_kinematics 2>&1 | tail -1)

echo "[4/4] Running our kinematics node: IK to TARGET_POSE (Ctrl+C to stop everything)"
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
ros2 run panda_kinematics kinematics
