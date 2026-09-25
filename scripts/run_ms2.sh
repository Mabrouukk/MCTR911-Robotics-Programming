#!/usr/bin/env bash
# Milestone 2 in one command: MuJoCo simulator + ROS2 bridge + one joint input.
#
#   scripts/run_ms2.sh                      pick-and-place step sequence (default)
#   scripts/run_ms2.sh sine                 sine wave on every joint
#   scripts/run_ms2.sh constant             hold the home pose (or: constant --deg q1 q2 q3 q4 q5 q6)
#   scripts/run_ms2.sh ik --xyz X Y Z       move the gripper flange to a world point
#   scripts/run_ms2.sh sliders              no ROS2 input: drag the sliders in the viewer's Control panel
#
# Press Ctrl+C to stop everything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="Milestone 02/code"
CONTAINER=mctr911_ms2
SIM_PID=""
LOGS_PID=""
[ $# -eq 0 ] && set -- sequence

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Milestone 2..."
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
    [ -n "$LOGS_PID" ] && kill "$LOGS_PID" 2>/dev/null || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd "$ROOT"

if ! docker info >/dev/null 2>&1; then
    echo "Starting Docker Desktop..."
    open -a Docker
    until docker info >/dev/null 2>&1; do sleep 2; done
fi
if ! docker image inspect mctr911-ros2-mujoco:latest >/dev/null 2>&1; then
    echo "Building the ROS2 + MuJoCo Docker image (first run only, takes a while)..."
    (cd docker && docker compose build)
fi
if [ ! -d .venv ]; then
    echo "Setting up the Python environment (first run only)..."
    python3 -m venv .venv
    .venv/bin/pip install --quiet mujoco numpy
fi

echo "[1/3] Opening the MuJoCo simulator (UR5e + gripper + work cell)"
PYTHONUNBUFFERED=1 .venv/bin/mjpython "$CODE/sim_bridge.py" > >(sed -l 's/^/[simulator] /') 2>&1 &
SIM_PID=$!
disown

echo "[2/3] Starting the ROS2 bridge in Docker"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
(cd docker && docker compose run -d --rm --name "$CONTAINER" robotics bash -lc \
    'source /opt/ros/humble/setup.bash && cd "/workspace/Milestone 02/code" && exec python3 -u mujoco_bridge_node.py') >/dev/null
for _ in $(seq 60); do
    docker logs "$CONTAINER" 2>&1 | grep -q "Connected to MuJoCo" && break
    sleep 1
done
docker logs -f "$CONTAINER" 2>&1 &
LOGS_PID=$!
disown

if [ "$1" = "sliders" ]; then
    echo "[3/3] No ROS2 input running: drag the joint sliders in the viewer's Control panel (Ctrl+C to stop)"
    while kill -0 "$SIM_PID" 2>/dev/null; do sleep 1; done
    exit 0
fi

echo "[3/3] Input: $* (Ctrl+C to stop everything)"
docker exec "$CONTAINER" bash -lc \
    'source /opt/ros/humble/setup.bash && cd "/workspace/Milestone 02/code" && exec python3 -u joint_command_publisher.py "$@"' _ "$@"
