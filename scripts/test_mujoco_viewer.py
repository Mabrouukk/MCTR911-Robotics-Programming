
import os
import time

import numpy as np

import mujoco
import mujoco.viewer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM_PATH = os.path.join(PROJECT_ROOT, "Milestone 01", "cad_models", "ur5e", "scene.xml")
GRIPPER_PATH = os.path.join(
    PROJECT_ROOT, "Milestone 01", "cad_models", "robotiq_2f85", "2f85.xml"
)

arm_spec = mujoco.MjSpec.from_file(ARM_PATH) // eft7 el files
gripper_spec = mujoco.MjSpec.from_file(GRIPPER_PATH)


arm_spec.option.cone = gripper_spec.option.cone
arm_spec.option.impratio = gripper_spec.option.impratio

attachment_site = arm_spec.site("attachment_site")
arm_spec.attach(gripper_spec, prefix="gripper_", site=attachment_site) // elza2 el gripper fel arm

model = arm_spec.compile() // compile el model
data = mujoco.MjData(model) // create el data structure for simulation

print(f"Loaded UR5e + Robotiq 2F-85 gripper: {model.nq} DoF, {model.nu} actuators")


home_arm_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]) // el arm fel home position
data.qpos[:6] = home_arm_qpos //set el arm  joint positions to home position
mujoco.mj_forward(model, data) // bet2ool le mujoko e7seb kol haga based pos da


amplitude = np.array([0.5, 0.3, 0.3, 0.4, 0.4, 0.6]) //how far each joint swings from its home position
frequency_hz = np.array([0.15, 0.15, 0.18, 0.2, 0.2, 0.25]) //how fast each joint oscillates back and forth per second
phase = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])


GRIPPER_OPEN = 0
GRIPPER_CLOSED = 255
GRIPPER_CYCLE_SECONDS = 4.0 //heyft7 w ye2fl el cycle kolaha 4 seconds

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()// el time ha2e2e
    while viewer.is_running():
        step_start = time.time()
        t = step_start - t0

        data.ctrl[:6] = home_arm_qpos + amplitude * np.sin(
            2 * np.pi * frequency_hz * t + phase
        )


        gripper_phase = (t % GRIPPER_CYCLE_SECONDS) / GRIPPER_CYCLE_SECONDS
        data.ctrl[6] = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * (
            0.5 - 0.5 * np.cos(2 * np.pi * gripper_phase)
        )

        mujoco.mj_step(model, data)
        viewer.sync()


        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
