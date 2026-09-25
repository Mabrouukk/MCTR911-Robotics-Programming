import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from kinematics import HOME_Q
from sim_bridge import ARM_ACTUATORS, ARM_JOINTS, CAD, build_model

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures")
WIDTH, HEIGHT = 1600, 1000


def render(model, data, filename, lookat, distance, azimuth, elevation, show_frames=False):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    options = mujoco.MjvOption()
    if show_frames:
        options.frame = mujoco.mjtFrame.mjFRAME_BODY
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera, options)
        image = renderer.render()
    path = os.path.join(FIGURES, filename)
    plt.imsave(path, image)
    print(f"Saved {os.path.relpath(path)}")


def main():
    model = build_model()
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    data.qpos[[model.joint(j).qposadr[0] for j in ARM_JOINTS]] = HOME_Q
    data.ctrl[[model.actuator(a).id for a in ARM_ACTUATORS]] = HOME_Q
    for _ in range(500):
        mujoco.mj_step(model, data)

    os.makedirs(FIGURES, exist_ok=True)
    render(model, data, "environment.png", lookat=(0.0, 0.2, 0.35), distance=2.6, azimuth=-130, elevation=-25)

    arm = mujoco.MjModel.from_xml_path(os.path.join(CAD, "ur5e", "scene.xml"))
    arm.vis.global_.offwidth = WIDTH
    arm.vis.global_.offheight = HEIGHT
    arm.vis.scale.framelength = 0.12 / arm.stat.meansize
    arm.vis.scale.framewidth = 0.005 / arm.stat.meansize
    floor_material = arm.geom("floor").matid
    for m in range(arm.nmat):
        if m != floor_material:
            arm.mat_rgba[m, 3] = 0.25
    arm_data = mujoco.MjData(arm)
    arm_data.qpos[:] = HOME_Q
    mujoco.mj_forward(arm, arm_data)
    render(arm, arm_data, "simulator_frames.png", lookat=(-0.05, 0.25, 0.3), distance=1.4, azimuth=-135, elevation=-20,
           show_frames=True)


if __name__ == "__main__":
    main()
