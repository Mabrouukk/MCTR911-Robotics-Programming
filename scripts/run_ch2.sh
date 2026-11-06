#!/usr/bin/env bash
# Challenge 2 (Panda Jacobian, Jacobian derivative, trajectory planning) in one command.
#
#   scripts/run_ch2.sh position   simulator in POSITION mode, node publishes q_traj (default)
#   scripts/run_ch2.sh velocity   simulator in VELOCITY mode, node publishes dq_traj
#   scripts/run_ch2.sh verify     only the offline checks + plots, no simulator window, no ROS 2
#
# Options (environment variables):
#   NO_GRAVCOMP=1   turn off gravity compensation in the simulator
#   HEADLESS=1      no viewer window; stop after the tracking summary (for automated tests)
#
# Press Ctrl+C to stop everything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CH="Challenge 02"
CONTAINER=mctr911_ch2
SIM_PID=""
LOGS_PID=""
MODE="${1:-position}"
MODE_UPPER="$(echo "$MODE" | tr a-z A-Z)"   # macOS bash 3.2 has no ${var^^} uppercase

case "$MODE" in
    position|velocity|verify) ;;
    *) echo "Usage: $0 [position|velocity|verify]"; exit 1 ;;
esac

cd "$ROOT"

# A venv stops working if the project folder is moved or copied, so check it actually runs.
if ! .venv/bin/mjpython -c "import mujoco, matplotlib" >/dev/null 2>&1; then
    echo "Setting up the Python environment (first run, or the project folder moved)..."
    rm -rf .venv
    python3 -m venv .venv
    .venv/bin/pip install --quiet mujoco numpy matplotlib
fi

if [ "$MODE" = "verify" ]; then
    exec .venv/bin/python "$CH/verify/verify_trajectory.py"
fi

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Challenge 2..."
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
    [ -n "$LOGS_PID" ] && kill "$LOGS_PID" 2>/dev/null || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if ! docker info >/dev/null 2>&1; then
    echo "Starting Docker Desktop..."
    open -a Docker
    until docker info >/dev/null 2>&1; do sleep 2; done
fi
if ! docker image inspect mctr911-ros2-mujoco:latest >/dev/null 2>&1; then
    echo "Building the ROS2 + MuJoCo Docker image (first run only, takes a while)..."
    (cd docker && docker compose build)
fi

SIM_ARGS=(--mode "$MODE")
[ "${NO_GRAVCOMP:-0}" = "1" ] && SIM_ARGS+=(--no-gravcomp)
if [ "${HEADLESS:-0}" = "1" ]; then
    SIM_ARGS+=(--headless)
    PY=.venv/bin/python
else
    PY=.venv/bin/mjpython
fi

echo "[1/4] Opening the MuJoCo simulator in $MODE_UPPER mode (arm at Q_START, yellow dots = planned hand path)"
PYTHONUNBUFFERED=1 "$PY" "$CH/sim/panda_sim.py" "${SIM_ARGS[@]}" > >(sed -l 's/^/[simulator] /') 2>&1 &
SIM_PID=$!

echo "[2/4] Starting the ROS2 bridge in Docker (/joint_commands -> simulator -> /joint_state)"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
(cd docker && docker compose run -d --rm --name "$CONTAINER" robotics bash -lc \
    'source /opt/ros/humble/setup.bash && cd "/workspace/Challenge 02/sim" && exec python3 -u panda_bridge_node.py') >/dev/null
for _ in $(seq 60); do
    docker logs "$CONTAINER" 2>&1 | grep -q "Connected to MuJoCo" && break
    sleep 1
done
docker logs -f "$CONTAINER" 2>&1 | sed -l 's/^/[bridge] /' &
LOGS_PID=$!
disown

echo "[3/4] Building panda_kinematics (colcon, inside Docker), node set to publish $MODE_UPPER commands"
# Built in /tmp inside the container so no build/ install/ log/ folders land in the repo.
# set_node_mode.py swaps the commented lines in tick() on that copy only.
docker exec "$CONTAINER" bash -lc \
    "source /opt/ros/humble/setup.bash && mkdir -p /tmp/ws/src && rm -rf /tmp/ws/src/panda_kinematics &&
     cp -r '/workspace/Challenge 02/panda_kinematics' /tmp/ws/src/ &&
     python3 '/workspace/Challenge 02/sim/set_node_mode.py' $MODE /tmp/ws/src/panda_kinematics/panda_kinematics/trajectory_node.py &&
     cd /tmp/ws && colcon build --packages-select panda_kinematics 2>&1 | tail -1"

echo "[4/4] Running trajectory_node (Ctrl+C to stop everything)"
NODE_CMD='source /opt/ros/humble/setup.bash && source /tmp/ws/install/setup.bash && exec ros2 run panda_kinematics trajectory_node'
if [ "${HEADLESS:-0}" = "1" ]; then
    docker exec "$CONTAINER" bash -lc "$NODE_CMD" 2>&1 | sed -l 's/^/[node] /' &
    wait "$SIM_PID"
else
    docker exec "$CONTAINER" bash -lc "$NODE_CMD"
fi
