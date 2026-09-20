# ROS2 (Docker) + MuJoCo (native) environment for macOS

ROS2 has no reliable native support on macOS, especially on Apple Silicon, so we run it inside a Linux container (Docker). MuJoCo, on the other hand, has first-class native macOS support, so we run **its viewer directly on the Mac** instead of forwarding it out of the container.

> We initially tried forwarding MuJoCo's GUI out of the Docker container via XQuartz/X11, but XQuartz's indirect GLX only supports old OpenGL profiles and fails to create the modern context MuJoCo's viewer needs (`GLX: Failed to create context: BadValue`). Running MuJoCo natively avoids this entirely and is the simpler, more reliable path for now. ROS2 and MuJoCo don't need to be linked together yet anyway — that's Milestone 2's job.

## Part A — ROS2 in Docker

### Build the image
```bash
cd docker
docker compose build
```

### Run the container
```bash
cd docker
docker compose run --rm robotics
```
This mounts the whole project at `/workspace` inside the container.

### Verify ROS2 works
Inside the container:
```bash
source /opt/ros/humble/setup.bash
ros2 topic list
```
In one terminal (`docker compose run --rm robotics`) run:
```bash
ros2 run demo_nodes_cpp talker
```
In a second terminal (`docker compose exec <container_name> bash`, while the first is running — check the running container's name with `docker ps`) run:
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```
You should see messages being published/received — **record this for the Milestone 1 video.**

## Part B — MuJoCo natively on macOS

### One-time setup
```bash
cd /path/to/MCTR
python3 -m venv .venv
source .venv/bin/activate
pip install mujoco numpy
```

### Run the UR5e viewer
MuJoCo's interactive viewer requires the special `mjpython` launcher on macOS (not plain `python3`):
```bash
source .venv/bin/activate
mjpython scripts/test_mujoco_viewer.py
```
This opens a native window with the UR5e arm loaded from `Milestone 01/cad_models/ur5e/scene.xml`. **Record this window for the Milestone 1 video.**

## Notes

- The UR5e MJCF model used by both the Docker image (`/opt/mujoco_menagerie/universal_robots_ur5e`) and the native script (`Milestone 01/cad_models/ur5e/`) comes from [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie).
- The Docker image still has `mujoco` installed and the model available, so it's ready for Milestone 2 when ROS2 and MuJoCo need to run together in the same process (at which point we'll revisit rendering — likely via offscreen/software rendering rather than a forwarded GUI window).
