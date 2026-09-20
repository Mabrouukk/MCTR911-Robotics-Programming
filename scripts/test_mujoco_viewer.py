"""Local (native macOS) sanity check: load the UR5e MJCF model and open the MuJoCo viewer."""
import os
import mujoco
import mujoco.viewer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "Milestone 01", "cad_models", "ur5e", "scene.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"Loaded UR5e model with {model.nq} DoF")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
