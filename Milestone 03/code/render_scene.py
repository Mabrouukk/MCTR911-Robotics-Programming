"""
Screenshots of the Milestone 3 environment for the report.

Builds the same model as sim_bridge.py (UR5e + Robotiq gripper + environment.xml),
runs the colour sorting for a few seconds so the arm is caught mid-task, and saves:

  report/figures/environment_overview.png   the whole factory corner around the cell
  report/figures/environment_operator.png   operator desk, HMI, control cabinet and stack light
  report/figures/environment_closeup.png    the maintenance workbench with its tools

    .venv/bin/python "Milestone 03/code/render_scene.py"
"""
import os

import matplotlib
matplotlib.use("Agg")  # draw in memory, no window
import matplotlib.pyplot as plt
import mujoco

from kinematics import HOME_Q
from sim_bridge import ARM_ACTUATORS, ARM_JOINTS, GRIPPER_ACTUATOR, Conveyor, build_model
from sorting_task import SortingController

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures")
WIDTH, HEIGHT = 1600, 1000

# name: (look-at point, distance, azimuth, elevation)
# azimuth is the direction the camera looks in, measured from +x towards +y.
VIEWS = {
    "environment_overview.png": ((-0.2, 0.1, 0.5), 5.4, -135, -30),
    "environment_operator.png": ((0.1, -0.9, 0.85), 3.4, 105, -20),
    "environment_closeup.png": ((2.0, 0.2, 0.95), 1.9, -60, -32),
}


def render(model, data, filename, lookat, distance, azimuth, elevation):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera)
        image = renderer.render()
    path = os.path.join(FIGURES, filename)
    plt.imsave(path, image)
    print(f"Saved {os.path.relpath(path)}")


def main():
    model = build_model()
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    arm_act = [model.actuator(a).id for a in ARM_ACTUATORS]
    grip_act = model.actuator(GRIPPER_ACTUATOR).id
    data.qpos[[model.joint(j).qposadr[0] for j in ARM_JOINTS]] = HOME_Q
    data.ctrl[arm_act] = HOME_Q
    mujoco.mj_forward(model, data)
    conveyor = Conveyor(model)
    sorter = SortingController(HOME_Q)

    # Run the cell like sim_bridge.py does: commands at 50 Hz, conveyor moving the parts.
    step = 0
    while data.time < 13.0:  # the arm is carrying the first (red) part towards its bin
        if step % 10 == 0:
            data.ctrl[arm_act], data.ctrl[grip_act], _ = sorter.update(data.time, conveyor.station_color(data))
        conveyor.step(data)
        mujoco.mj_step(model, data)
        step += 1

    os.makedirs(FIGURES, exist_ok=True)
    for filename, (lookat, distance, azimuth, elevation) in VIEWS.items():
        render(model, data, filename, lookat, distance, azimuth, elevation)


if __name__ == "__main__":
    main()
