"""Local (native macOS) sanity check: load the UR5e MJCF model and open the
MuJoCo viewer, showing it visibly react to real physics (gravity) rather
than just sitting rigid under its built-in position-holding motors."""
import os
import time

import mujoco
import mujoco.viewer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "Milestone 01", "cad_models", "ur5e", "scene.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"Loaded UR5e model with {model.nq} DoF")

# The UR5e's actuators default to holding every joint at its start angle
# (like a real robot with its motors engaged), so the arm looks frozen.
# Zeroing the gain/bias here lets gravity actually move it, which is a much
# clearer visual proof the simulator is running real physics.
model.actuator_gainprm[:] = 0
model.actuator_biasprm[:] = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        # Pace the loop to real time so motion plays at a natural speed
        # instead of resolving instantly.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
