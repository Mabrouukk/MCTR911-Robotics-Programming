# ROS2 (Docker) + MuJoCo (native) environment for macOS

ROS2 is only officially supported on Ubuntu; macOS support is "best effort" and means compiling ROS2 from source, which is slow and fragile, especially on Apple Silicon. Docker is the standard way to run ROS2 on a Mac, so ROS2 runs in a container while MuJoCo, which has proper native macOS support, runs directly on the Mac. In Milestone 2 the two are linked over a local network socket (`host.docker.internal:9999`).

First time only, build the Docker image:
```bash
cd docker && docker compose build
```

All commands also exist as VS Code tasks: ⌘⇧P → "Tasks: Run Task".

## Milestone 1

| What | Command |
|---|---|
| MuJoCo UR5e + gripper demo | `scripts/run_mujoco_demo.sh` |
| ROS2 publisher (run first) | `scripts/run_ros2_publisher.sh` |
| ROS2 subscriber (second terminal) | `scripts/run_ros2_subscriber.sh` |

## Milestone 2 (ROS2 driving MuJoCo)

Start in this order, each in its own terminal:

| Step | Command |
|---|---|
| 1. Simulator (Mac) | `scripts/run_m2_simulator.sh` |
| 2. ROS2 bridge (Docker) | `scripts/run_m2_bridge.sh` |
| 3. Pick an input | `scripts/run_m2_commands.sh constant` / `sine` / `sequence` / `ik --xyz X Y Z` |
| 4. Watch state coming back (optional) | `scripts/run_m2_monitor.sh /ur5e/ee_position` or `/ur5e/joint_states` |

Sliders: when no input is running, drag the joint sliders in the MuJoCo viewer's **Control** panel (right side).
