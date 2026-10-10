import os

import matplotlib
matplotlib.use("Agg") #draw fel memory agg w matfta7sh ay window
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from kinematics import HOME_Q # el zawaya el hanbd2 beha 
from sim_bridge import ARM_ACTUATORS, ARM_JOINTS, CAD, GRIPPER_ACTUATOR, Conveyor, build_model # benakhod meno el model el 3amaly w el CAD files 
from sorting_task import SortingController # el robot by-sort el parts 3ala 7asab el loon

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures")
WIDTH, HEIGHT = 1600, 1000

# el fanction deh bt render el scene w t save el image fe el path elly enta 3ayzo w bta5od el model w data w filename w lookat w distance w azimuth w elevation w show_frames
def render(model, data, filename, lookat, distance, azimuth, elevation, show_frames=False):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    options = mujoco.MjvOption()
    if show_frames:
        options.frame = mujoco.mjtFrame.mjFRAME_BODY # mujoco frames ahmar akhdar azra2
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
    arm_act = [model.actuator(a).id for a in ARM_ACTUATORS]
    grip_act = model.actuator(GRIPPER_ACTUATOR).id
    data.qpos[[model.joint(j).qposadr[0] for j in ARM_JOINTS]] = HOME_Q
    data.ctrl[arm_act] = HOME_Q
    conveyor = Conveyor(model)
    sorter = SortingController(HOME_Q)

    # nshaghal el cell: el conveyor by7arak el parts w el robot by-sort, w nsawar marteen
    def run_until(t_end):
        while data.time < t_end:
            if round(data.time / model.opt.timestep) % 10 == 0:  # commands at 50 Hz, like the ROS2 publisher
                data.ctrl[arm_act], data.ctrl[grip_act], _ = sorter.update(data.time, conveyor.station_color(data))
            conveyor.step(data)
            mujoco.mj_step(model, data)

    os.makedirs(FIGURES, exist_ok=True)
    run_until(5.5)  # first part reached the station, the rest still on their way
    render(model, data, "environment.png", lookat=(-0.1, 0.2, 0.3), distance=2.7, azimuth=-125, elevation=-28)
    run_until(13.0)  # carrying the red part over to the red bin
    render(model, data, "sorting_in_action.png", lookat=(0.3, 0.25, 0.35), distance=1.9, azimuth=-150, elevation=-25)

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
