# ROS2 (Docker) + MuJoCo (native) environment for macOS

ROS2 is only officially supported on Ubuntu; macOS support is "best effort" and means compiling ROS2 from source, which is slow and fragile, especially on Apple Silicon. Docker is the standard way to run ROS2 on a Mac, so ROS2 runs in a container while MuJoCo, which has proper native macOS support, runs directly on the Mac.

## Run Milestone 1

```bash
scripts/run_ms1.sh
```

Or in VS Code: ⌘⇧P → "Tasks: Run Task" → **Run Milestone 1**.

It starts Docker if needed, builds the image on the first run, opens the MuJoCo viewer (UR5e + gripper), and runs the ROS2 pick-and-place publisher and subscriber with both outputs in the same terminal. Press Ctrl+C to stop everything.
