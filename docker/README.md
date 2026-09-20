# ROS2 + MuJoCo environment (Docker, for macOS)

ROS2 has no reliable native support on macOS, especially on Apple Silicon, so we run it inside a Linux container. MuJoCo is installed in the same container (via `pip install mujoco`) so both tools live in one consistent environment — this also sets us up for Milestone 2, where ROS2 and MuJoCo need to be linked together.

To see GUI windows from the container (the MuJoCo viewer, rviz2, etc.) on your Mac, we forward X11 through **XQuartz**.

## One-time host setup

1. Install XQuartz:
   ```bash
   brew install --cask xquartz
   ```
2. Log out and back in (or reboot) so XQuartz's X server starts.
3. Open XQuartz → Settings → Security → check **"Allow connections from network clients"**.
4. Restart XQuartz, then in a terminal run:
   ```bash
   xhost + 127.0.0.1
   ```
   (Run this once per XQuartz session/reboot before launching the container.)

## Build the image

```bash
cd docker
docker compose build
```

## Run the container

```bash
cd docker
docker compose run --rm robotics
```

This mounts the whole project at `/workspace` inside the container, so any files you create/edit are visible on your Mac and vice versa.

## Verify ROS2 works

Inside the container:
```bash
ros2 --version
ros2 topic list
```
In one terminal (`docker compose run --rm robotics`) run:
```bash
ros2 run demo_nodes_cpp talker
```
In a second terminal (`docker compose exec mctr911 bash` while the first is running) run:
```bash
ros2 run demo_nodes_py listener
```
You should see messages being published/received — record this for the Milestone 1 video.

## Verify MuJoCo works (and view the UR5e model)

Inside the container:
```bash
python3 /workspace/docker/test_mujoco.py
```
This opens the MuJoCo passive viewer loaded with the UR5e model (forwarded to your Mac screen via XQuartz). Record this window for the Milestone 1 video.

## Notes

- The UR5e MJCF model lives at `/opt/mujoco_menagerie/universal_robots_ur5e` inside the image (cloned from [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)). A copy is also kept in `../Milestone 01/cad_models/` for submission.
- If the viewer window doesn't appear, double check `xhost + 127.0.0.1` was run on the host and that XQuartz's "Allow connections from network clients" setting is enabled, then re-run.
