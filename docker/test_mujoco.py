"""Milestone 1 sanity check: load the UR5e MJCF model and open the MuJoCo viewer."""
import mujoco
import mujoco.viewer

MODEL_PATH = "/opt/mujoco_menagerie/universal_robots_ur5e/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"Loaded model with {model.nq} DoF")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
