# ROS2 (Docker) + MuJoCo (native) environment for macOS

ROS2 is only officially supported on Ubuntu; macOS support is "best effort" and means compiling ROS2 from source, which is slow and fragile, especially on Apple Silicon. Docker is the standard way to run ROS2 on a Mac, so ROS2 runs in a container while MuJoCo, which has proper native macOS support, runs directly on the Mac.

## Run Milestone 1

```bash
scripts/run_ms1.sh
```

Or in VS Code: ⌘⇧P → "Tasks: Run Task" → **Run Milestone 1**.

It starts Docker if needed, builds the image on the first run, opens the MuJoCo viewer (UR5e + gripper), and runs the ROS2 pick-and-place publisher and subscriber with both outputs in the same terminal. Press Ctrl+C to stop everything.

## Run Milestone 2

```bash
scripts/run_ms2.sh                  # pick-and-place step sequence (default)
scripts/run_ms2.sh sine             # sine wave on every joint
scripts/run_ms2.sh constant         # hold the home pose
scripts/run_ms2.sh ik --xyz X Y Z   # move the gripper flange to a world point
scripts/run_ms2.sh sliders          # no ROS2 input, use the sliders in the viewer's Control panel
```

Or in VS Code: ⌘⇧P → "Tasks: Run Task" → **Run Milestone 2** (pick the input from the list) or **Run Milestone 2 (IK to a point)**.

It opens the MuJoCo simulator with the work cell, starts the ROS2 bridge in Docker, waits until they're connected (over `host.docker.internal:9999`), then runs the chosen input. To watch the state coming back from the simulator, in another terminal run:

```bash
docker exec -it mctr911_ms2 bash -lc "source /opt/ros/humble/setup.bash && ros2 topic echo /ur5e/ee_position"
```
