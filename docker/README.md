# ROS2 (Docker) + MuJoCo (native) environment for macOS

ROS2 is only officially supported on Ubuntu; macOS support is "best effort" and means compiling ROS2 from source, which is slow and fragile, especially on Apple Silicon. Docker is the standard way to run ROS2 on a Mac, so ROS2 runs in a container while MuJoCo, which has proper native macOS support, runs directly on the Mac.

## Run everything

| What | Command | Or via VS Code |
|---|---|---|
| MuJoCo UR5e + gripper demo | `scripts/run_mujoco_demo.sh` | ⌘⇧P → "Tasks: Run Task" → **Run: MuJoCo demo** |
| ROS2 publisher (run first) | `scripts/run_ros2_publisher.sh` | ⌘⇧P → "Tasks: Run Task" → **Run: ROS2 publisher** |
| ROS2 subscriber (run second, separate terminal) | `scripts/run_ros2_subscriber.sh` | ⌘⇧P → "Tasks: Run Task" → **Run: ROS2 subscriber** |

First time only, build the Docker image:
```bash
cd docker && docker compose build
```
