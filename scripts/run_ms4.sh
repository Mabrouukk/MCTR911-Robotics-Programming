#!/usr/bin/env bash
# Milestone 4 in one command: torque-mode MuJoCo simulator + ROS 2 bridge + the controller node.
#
#   scripts/run_ms4.sh                 colour sorting, computed-torque control (default)
#   scripts/run_ms4.sh sort  [ctrl]    colour sorting with controller ctrl
#   scripts/run_ms4.sh joint [ctrl]    joint-space test trajectory (quintic + LSPB), tracking summary at the end
#   scripts/run_ms4.sh task  [ctrl]    task-space test trajectory (lines + circle), tracking summary at the end
#   scripts/run_ms4.sh validate        offline: dynamics vs MuJoCo, trajectories, controllers, figures (no ROS 2)
#
#   ctrl = ctc (computed torque, default) | ctci (computed torque + integral) | pdg (PD + gravity comp.) | pd (PD only)
#
# HEADLESS=1 runs the simulator without a window and stops by itself (automated tests).
# PAYLOAD=2  hides a 2 kg mass in the gripper that the controller does not know about.
# Press Ctrl+C to stop everything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="Milestone 04/code"
CONTAINER=mctr911_ms4
SIM_PID=""
LOGS_PID=""
MODE="${1:-sort}"
CTRL="${2:-ctc}"

case "$MODE" in
    sort|joint|task|validate) ;;
    *) echo "Usage: $0 [sort|joint|task|validate] [ctc|pdg|pd]"; exit 1 ;;
esac
case "$CTRL" in
    ctc|ctci|pdg|pd) ;;
    *) echo "Controller must be ctc, ctci, pdg or pd"; exit 1 ;;
esac

cd "$ROOT"

# A venv stops working if the project folder is moved or copied, so check it actually runs.
if ! .venv/bin/mjpython -c "import mujoco, matplotlib, sympy" >/dev/null 2>&1; then
    echo "Setting up the Python environment (first run, or the project folder moved)..."
    rm -rf .venv
    python3 -m venv .venv
    .venv/bin/pip install --quiet mujoco numpy matplotlib sympy pillow
fi

if [ "$MODE" = "validate" ]; then
    exec .venv/bin/python "$CODE/validate_control.py"
fi

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Milestone 4..."
    # Stop the controller first (Ctrl+C makes it print its summary and save its log).
    docker exec "$CONTAINER" pkill -INT -f ur5e_controller_node >/dev/null 2>&1 || true
    sleep 3
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
    [ -n "$SIM_PID" ] && wait "$SIM_PID" 2>/dev/null || true      # frees the simulator's port
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

SIM_ARGS=(--payload "${PAYLOAD:-0}")
PY=.venv/bin/mjpython
if [ "${HEADLESS:-0}" = "1" ]; then
    case "$MODE" in
        joint) DURATION=30 ;;
        task)  DURATION=32 ;;
        *)     DURATION=110 ;;
    esac
    SIM_ARGS+=(--headless --duration "$DURATION")
    PY=.venv/bin/python
fi

echo "[1/3] Opening the MuJoCo simulator in TORQUE mode (arm on its brakes until the controller starts)"
PYTHONUNBUFFERED=1 "$PY" "$CODE/sim_bridge.py" "${SIM_ARGS[@]}" > >(sed -l 's/^/[simulator] /') 2>&1 &
SIM_PID=$!

echo "[2/3] Starting the ROS 2 bridge in Docker"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
(cd docker && docker compose run -d --rm --name "$CONTAINER" robotics bash -lc \
    'source /opt/ros/humble/setup.bash && cd "/workspace/Milestone 04/code" && exec python3 -u mujoco_bridge_node.py') >/dev/null
for _ in $(seq 60); do
    docker logs "$CONTAINER" 2>&1 | grep -q "Connected to MuJoCo" && break
    sleep 1
done
docker logs -f "$CONTAINER" 2>&1 | grep --line-buffered -v "Simulator not reachable" | sed -l 's/^/[bridge] /' &
LOGS_PID=$!
disown

echo "[3/3] Running the controller node: $MODE, controller $CTRL (Ctrl+C to stop everything)"
NODE_CMD='source /opt/ros/humble/setup.bash && cd "/workspace/Milestone 04/code" && exec python3 -u ur5e_controller_node.py "$0" "$1"'
if [ "${HEADLESS:-0}" = "1" ]; then
    docker exec "$CONTAINER" bash -lc "$NODE_CMD" "$MODE" "$CTRL" 2>&1 | sed -l 's/^/[controller] /' &
    wait "$SIM_PID"
else
    docker exec "$CONTAINER" bash -lc "$NODE_CMD" "$MODE" "$CTRL"
fi
