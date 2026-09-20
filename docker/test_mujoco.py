"""Headless sanity check (run inside the ROS2 Docker container): load the UR5e
MJCF model and step the physics a few times. No GUI window is opened here —
use `mjpython scripts/test_mujoco_viewer.py` on the host macOS for the
interactive viewer (see docker/README.md for why)."""
import mujoco

MODEL_PATH = "/opt/mujoco_menagerie/universal_robots_ur5e/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"Loaded model with {model.nq} DoF")

for _ in range(100):
    mujoco.mj_step(model, data)

print("Stepped physics 100 times successfully. qpos:", data.qpos)
