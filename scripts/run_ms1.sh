#!/usr/bin/env bash
# Milestone 1 demo in one command: MuJoCo viewer (UR5e + gripper) and the ROS2 pick-and-place
# publisher/subscriber pair. Press Ctrl+C to stop everything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER=mctr911_ms1
VIEWER_PID=""
LOGS_PID=""

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Milestone 1..."
    [ -n "$VIEWER_PID" ] && kill "$VIEWER_PID" 2>/dev/null || true
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

echo "[1/3] Opening the MuJoCo viewer (UR5e + Robotiq gripper)"
.venv/bin/mjpython scripts/test_mujoco_viewer.py >/dev/null 2>&1 &
VIEWER_PID=$!
disown

echo "[2/3] Starting the ROS2 publisher in Docker"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
(cd docker && docker compose run -d --rm --name "$CONTAINER" robotics bash -lc \
    'source /opt/ros/humble/setup.bash && exec python3 -u /workspace/ros2_nodes/pick_place_publisher.py') >/dev/null
docker logs -f "$CONTAINER" 2>&1 &
LOGS_PID=$!
disown

echo "[3/3] Starting the ROS2 subscriber (Ctrl+C to stop everything)"
docker exec "$CONTAINER" bash -lc \
    'source /opt/ros/humble/setup.bash && exec python3 -u /workspace/ros2_nodes/pick_place_subscriber.py'
