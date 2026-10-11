#!/usr/bin/env bash
# Challenge 1 (Franka Panda FK + IK) in one command.
#
#   scripts/run_ch1.sh            MuJoCo simulator + ROS2 bridge + our panda_kinematics node (default)
#   scripts/run_ch1.sh verify     only check FK and IK against MuJoCo, no simulator window, no ROS2
#
# Press Ctrl+C to stop everything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CH="Challenge 01"
CONTAINER=mctr911_ch1
SIM_PID=""
LOGS_PID=""
MODE="${1:-run}"

cd "$ROOT"

# A venv stops working if the project folder is moved or copied, so check it actually runs.
if ! .venv/bin/mjpython -c "import mujoco" >/dev/null 2>&1; then
    echo "Setting up the Python environment (first run, or the project folder moved)..."
    rm -rf .venv
    python3 -m venv .venv
    .venv/bin/pip install --quiet mujoco numpy matplotlib
fi

if [ "$MODE" = "verify" ]; then
    exec .venv/bin/python "$CH/verify/verify_kinematics.py"
fi

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Stopping Challenge 1..."
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

echo "[1/4] Opening the MuJoCo simulator (Panda at its home pose, RGB frame = target)"
PYTHONUNBUFFERED=1 .venv/bin/mjpython "$CH/sim/panda_sim.py" > >(sed -l 's/^/[simulator] /') 2>&1 &
SIM_PID=$!
disown

echo "[2/4] Starting the ROS2 bridge in Docker (/joint_commands -> simulator -> /joint_state)"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
(cd docker && docker compose run -d --rm --name "$CONTAINER" robotics bash -lc \
    'source /opt/ros/humble/setup.bash && cd "/workspace/Challenge 01/sim" && exec python3 -u panda_bridge_node.py') >/dev/null
for _ in $(seq 60); do
    docker logs "$CONTAINER" 2>&1 | grep -q "Connected to MuJoCo" && break
    sleep 1
done
docker logs -f "$CONTAINER" 2>&1 | sed -l 's/^/[bridge] /' &
LOGS_PID=$!
disown

echo "[3/4] Building the panda_kinematics package (colcon, inside Docker)"
# Built in /tmp inside the container so no build/ install/ log/ folders land in the repo.
docker exec "$CONTAINER" bash -lc \
    'source /opt/ros/humble/setup.bash && mkdir -p /tmp/ws/src && rm -rf /tmp/ws/src/panda_kinematics &&
     cp -r "/workspace/Challenge 01/panda_kinematics" /tmp/ws/src/ && cd /tmp/ws &&
     colcon build --packages-select panda_kinematics 2>&1 | tail -1'

echo "[4/4] Running our kinematics node: IK to TARGET_POSE (Ctrl+C to stop everything)"
docker exec "$CONTAINER" bash -lc \
    'source /opt/ros/humble/setup.bash && source /tmp/ws/install/setup.bash && exec ros2 run panda_kinematics kinematics'
