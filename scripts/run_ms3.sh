#!/usr/bin/env bash
# Milestone 3 in one command: MuJoCo simulator (with the live kinematics panel) + ROS2 bridge + a test.
#
#   scripts/run_ms3.sh forward    joint-space input (sine waves); checks FK, J q_dot, J q_ddot + J_dot q_dot
#                                 and IK, J^-1, J^-1(...) against the simulator's sensors, live (default)
#   scripts/run_ms3.sh inverse    task-space input: a flange path -> IK / J^-1 / J^-1(...) -> joint commands;
#                                 same checks, plus how closely the flange follows the path
#   scripts/run_ms3.sh sort       Milestone 2's colour sorting in the full environment, checks running
#   scripts/run_ms3.sh validate   offline checks + report figures, no simulator window, no ROS2 (about 6 s)
#   scripts/run_ms3.sh derive     symbolic derivation of J, det J and J_dot with sympy (about 5 s)
#
# HEADLESS=1 runs the simulator without a window (automated tests).
# Press Ctrl+C to stop everything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="Milestone 03/code"
CONTAINER=mctr911_ms3
SIM_PID=""
LOGS_PID=""
MODE="${1:-forward}"

case "$MODE" in
    forward|inverse|sort|validate|derive) ;;
    *) echo "Usage: $0 [forward|inverse|sort|validate|derive]"; exit 1 ;;
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
    exec .venv/bin/python "$CODE/validate_kinematics.py"
fi
if [ "$MODE" = "derive" ]; then
    exec .venv/bin/python "$CODE/derive_symbolic.py"
fi

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Milestone 3..."
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

SIM_ARGS=()
PY=.venv/bin/mjpython
if [ "${HEADLESS:-0}" = "1" ]; then
    SIM_ARGS+=(--headless --duration 40)
    PY=.venv/bin/python
fi

echo "[1/3] Opening the MuJoCo simulator (UR5e cell + live kinematics panel, top right)"
PYTHONUNBUFFERED=1 "$PY" "$CODE/sim_bridge.py" ${SIM_ARGS[@]+"${SIM_ARGS[@]}"} > >(sed -l 's/^/[simulator] /') 2>&1 &
SIM_PID=$!

echo "[2/3] Starting the ROS2 bridge in Docker"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
(cd docker && docker compose run -d --rm --name "$CONTAINER" robotics bash -lc \
    'source /opt/ros/humble/setup.bash && cd "/workspace/Milestone 03/code" && exec python3 -u mujoco_bridge_node.py') >/dev/null
for _ in $(seq 60); do
    docker logs "$CONTAINER" 2>&1 | grep -q "Connected to MuJoCo" && break
    sleep 1
done
docker logs -f "$CONTAINER" 2>&1 | grep --line-buffered -v "Simulator not reachable" | sed -l 's/^/[bridge] /' &
LOGS_PID=$!
disown

echo "[3/3] Running the kinematics validator: $MODE (Ctrl+C to stop everything)"
NODE_CMD='source /opt/ros/humble/setup.bash && cd "/workspace/Milestone 03/code" && exec python3 -u kinematics_validator.py "$0"'
if [ "${HEADLESS:-0}" = "1" ]; then
    docker exec "$CONTAINER" bash -lc "$NODE_CMD" "$MODE" 2>&1 | sed -l 's/^/[validator] /' &
    wait "$SIM_PID"
else
    docker exec "$CONTAINER" bash -lc "$NODE_CMD" "$MODE"
fi
