#!/usr/bin/env bash
# Challenge 3 (Panda Lagrangian dynamics + fuzzy logic torque control) in one command.
#
#   scripts/run_ch3.sh                 model + FLC (the main design, default)
#   scripts/run_ch3.sh model_flc       tau = M q_ddot_des + C q_dot + g + FLC
#   scripts/run_ch3.sh model_only      tau = M q_ddot_des + C q_dot + g   (the FLC removed)
#   scripts/run_ch3.sh gravity_flc     tau = g(q) + FLC                   (Team 5 variant)
#   scripts/run_ch3.sh flc_only        tau = FLC                          (no model, for the debate)
#   scripts/run_ch3.sh verify          dynamics vs MuJoCo + the FLC study and plots, no ROS 2
#
# Options (environment variables):
#   MASS20=1    every link of the simulated arm 20 % heavier than the model thinks (Team 5 debate)
#   HEADLESS=1  no viewer window; stop after the tracking summary (automated tests)
#
# Press Ctrl+C to stop everything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CH="Challenge 03"
CONTAINER=mctr911_ch3
SIM_PID=""
LOGS_PID=""
MODE="${1:-model_flc}"
MODE_UPPER="$(echo "$MODE" | tr a-z A-Z)"   # macOS bash 3.2 has no ${var^^}

case "$MODE" in
    model_flc|model_only|gravity_flc|flc_only|verify) ;;
    *) echo "Usage: $0 [model_flc|model_only|gravity_flc|flc_only|verify]"; exit 1 ;;
esac

cd "$ROOT"

if ! .venv/bin/mjpython -c "import mujoco, matplotlib" >/dev/null 2>&1; then
    echo "Setting up the Python environment (first run, or the project folder moved)..."
    rm -rf .venv
    python3 -m venv .venv
    .venv/bin/pip install --quiet mujoco numpy matplotlib sympy pillow
fi

if [ "$MODE" = "verify" ]; then
    .venv/bin/python "$CH/verify/verify_dynamics.py"
    exec .venv/bin/python "$CH/verify/flc_study.py"
fi

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Challenge 3..."
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
    [ -n "$SIM_PID" ] && wait "$SIM_PID" 2>/dev/null || true
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

TAG="ros_$MODE"
SIM_ARGS=()
if [ "${MASS20:-0}" = "1" ]; then
    for i in 1 2 3 4 5 6 7; do SIM_ARGS+=(--mass-scale "link$i=1.2"); done
    TAG="${TAG}_mass20"
fi
SIM_ARGS+=(--tag "$TAG")
if [ "${HEADLESS:-0}" = "1" ]; then
    SIM_ARGS+=(--headless)
    PY=.venv/bin/python
else
    PY=.venv/bin/mjpython
fi

echo "[1/4] Opening the MuJoCo simulator in TORQUE mode (arm on its brakes at Q_START, yellow dots = planned hand path)"
PYTHONUNBUFFERED=1 "$PY" "$CH/sim/panda_sim.py" "${SIM_ARGS[@]}" > >(sed -l 's/^/[simulator] /') 2>&1 &
SIM_PID=$!

echo "[2/4] Starting the ROS 2 bridge in Docker (/joint_commands torques -> simulator -> /joint_state)"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
(cd docker && docker compose run -d --rm --name "$CONTAINER" robotics bash -lc \
    'source /opt/ros/humble/setup.bash && cd "/workspace/Challenge 03/sim" && exec python3 -u panda_bridge_node.py') >/dev/null
for _ in $(seq 60); do
    docker logs "$CONTAINER" 2>&1 | grep -q "Connected to MuJoCo" && break
    sleep 1
done
docker logs -f "$CONTAINER" 2>&1 | grep --line-buffered -v "Simulator not reachable" | sed -l 's/^/[bridge] /' &
LOGS_PID=$!
disown

echo "[3/4] Building panda_kinematics (colcon, inside Docker), CONTROL_MODE = $MODE_UPPER"
# Built in /tmp inside the container so no build/ install/ log/ folders land in the repo.
# The mode is set on that copy only; the file in the repo stays at the default MODEL_FLC.
docker exec "$CONTAINER" bash -lc \
    "source /opt/ros/humble/setup.bash && mkdir -p /tmp/ws/src && rm -rf /tmp/ws/src/panda_kinematics &&
     cp -r '/workspace/$CH/panda_kinematics' /tmp/ws/src/ &&
     sed -i 's/^CONTROL_MODE = .*/CONTROL_MODE = \"$MODE_UPPER\"/' /tmp/ws/src/panda_kinematics/panda_kinematics/controller_node.py &&
     cd /tmp/ws && colcon build --packages-select panda_kinematics 2>&1 | tail -1"

echo "[4/4] Running controller_node (Ctrl+C to stop everything)"
NODE_CMD='source /opt/ros/humble/setup.bash && source /tmp/ws/install/setup.bash && exec ros2 run panda_kinematics controller_node'
if [ "${HEADLESS:-0}" = "1" ]; then
    docker exec "$CONTAINER" bash -lc "$NODE_CMD" 2>&1 | sed -l 's/^/[node] /' &
    wait "$SIM_PID"
else
    docker exec "$CONTAINER" bash -lc "$NODE_CMD"
fi
