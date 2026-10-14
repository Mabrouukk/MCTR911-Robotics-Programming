import os
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kinematics import HOME_Q, ROBOT_BASE_POS, forward_kinematics
from sim_bridge import ARM_ACTUATORS, ARM_JOINTS, GRIPPER_ACTUATOR, Conveyor, build_model
from sorting_task import GRIPPER_CLOSED, GRIPPER_OPEN, SortingController, ik_tool_down

VIDEOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "videos")
WIDTH, HEIGHT, FPS = 1280, 720, 30
RATE_HZ = 50.0  # same command rate as joint_command_publisher.py


def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT, TEXT_FONT = _font(30), _font(22)


class Video:
    """Renders the scene from a fixed camera and pipes the frames into ffmpeg."""

    def __init__(self, model, filename, lookat, distance, azimuth, elevation):
        os.makedirs(VIDEOS, exist_ok=True)
        self.path = os.path.join(VIDEOS, filename)
        self.renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = lookat
        self.camera.distance = distance
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", self.path],
            stdin=subprocess.PIPE)

    def frame(self, data, title, lines=()):
        self.renderer.update_scene(data, self.camera)
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, WIDTH, 50 + 32 * len(lines)), fill=(0, 0, 0, 150))
        draw.text((20, 10), title, font=TITLE_FONT, fill="white")
        for i, line in enumerate(lines):
            draw.text((20, 52 + 32 * i), line, font=TEXT_FONT, fill=(230, 230, 230))
        draw.text((WIDTH - 150, HEIGHT - 36), f"t = {data.time:5.1f} s", font=TEXT_FONT, fill="white")
        self.ffmpeg.stdin.write(image.tobytes())

    def close(self):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        self.renderer.close()
        print(f"Saved {os.path.relpath(self.path)}")


class Cell:
    """The sorting cell with the arm at home, stepped the same way sim_bridge.py does it."""

    def __init__(self):
        self.model = build_model()
        self.model.vis.global_.offwidth = WIDTH
        self.model.vis.global_.offheight = HEIGHT
        self.data = mujoco.MjData(self.model)
        self.arm_act = [self.model.actuator(a).id for a in ARM_ACTUATORS]
        self.grip_act = self.model.actuator(GRIPPER_ACTUATOR).id
        self.qpos_adr = [self.model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.ee_site = self.model.site("attachment_site").id
        self.data.qpos[self.qpos_adr] = HOME_Q
        self.data.ctrl[self.arm_act] = HOME_Q
        mujoco.mj_forward(self.model, self.data)
        self.conveyor = Conveyor(self.model)
        self.steps_per_command = round(1 / (RATE_HZ * self.model.opt.timestep))
        self.steps_per_frame = round(1 / (FPS * self.model.opt.timestep))
        self._step = 0

    def run(self, video, duration, command, caption):
        """Steps for `duration` seconds. command(t) -> (q, gripper); caption(t) -> (title, lines)."""
        t0 = self.data.time
        while self.data.time < t0 + duration:
            if self._step % self.steps_per_command == 0:
                q, gripper = command(self.data.time - t0)
                self.data.ctrl[self.arm_act] = q
                self.data.ctrl[self.grip_act] = gripper
            if self._step % self.steps_per_frame == 0:
                video.frame(self.data, *caption(self.data.time - t0))
            self.conveyor.step(self.data)
            mujoco.mj_step(self.model, self.data)
            self._step += 1


def sorting_video():
    cell = Cell()
    sorter = SortingController(HOME_Q)
    video = Video(cell.model, "ms2_colour_sorting.mp4", lookat=(0.0, 0.25, 0.35), distance=2.3,
                  azimuth=-130, elevation=-25)
    state = {"step": "waiting for a part at the pick station"}

    def command(t):
        q, gripper, event = sorter.update(cell.data.time, cell.conveyor.station_color(cell.data))
        if event:
            state["step"] = event.split(" (")[0]
        return q, gripper

    def caption(t):
        seen = cell.conveyor.station_color(cell.data) or "-"
        counts = ", ".join(f"{c}: {n}" for c, n in sorter.sorted.items())
        return "UR5e colour sorting (Milestone 2)", [
            f"Colour sensor at the pick station: {seen}",
            f"Step: {state['step']}",
            f"Sorted so far -> {counts}"]

    # Run until all six parts are in their bins (about 83 s), plus a short pause at the end.
    while sum(sorter.sorted.values()) < 6 and cell.data.time < 120:
        cell.run(video, 1.0, command, caption)
    cell.run(video, 2.0, command, caption)
    video.close()


def joint_inputs_video():
    cell = Cell()
    video = Video(cell.model, "ms2_joint_inputs.mp4", lookat=(0.0, 0.3, 0.45), distance=2.0,
                  azimuth=-140, elevation=-20)

    def ee_line():
        return f"Flange position (sensed): {np.round(cell.data.site_xpos[cell.ee_site], 3)} m"

    # 1. Constant input: hold the home pose.
    cell.run(video, 4.0, lambda t: (HOME_Q, GRIPPER_OPEN),
             lambda t: ("Input 1: constant (home pose)",
                        [f"q = {np.round(np.degrees(HOME_Q)).astype(int)} deg", ee_line()]))

    # 2. Signal builder: sine wave on every joint, phase-shifted down the arm.
    amp, freq, phase = np.radians(20.0), 0.2, np.arange(6) * 0.5
    w = 2 * np.pi * freq

    def sine(t):
        return HOME_Q + amp * np.sin(w * t + phase), GRIPPER_CLOSED * 0.5 * (1 - np.cos(w * t))

    cell.run(video, 10.0, sine,
             lambda t: ("Input 2: signal builder (sine wave)",
                        ["q_i(t) = q_home,i + 20 deg * sin(2 pi 0.2 t + 0.5 i), gripper opens and closes",
                         ee_line()]))

    # 3. IK targets: world points, gripper pointing down, solved with our analytical IK.
    q = HOME_Q
    for xyz in [(-0.2, 0.6, 0.55), (0.4, 0.3, 0.6), (0.1, 0.5, 0.75)]:
        q = ik_tool_down(xyz, q)
        fk = forward_kinematics(q)[:3, 3] + ROBOT_BASE_POS
        cell.run(video, 4.0, lambda t, q=q: (q, GRIPPER_OPEN),
                 lambda t, xyz=xyz, q=q, fk=fk: (f"Input 3: IK target {xyz} m",
                                                 [f"IK -> q = {np.round(np.degrees(q), 1)} deg",
                                                  f"FK(q) = {np.round(fk, 3)} m", ee_line()]))
    video.close()


def main():
    sorting_video()
    joint_inputs_video()


if __name__ == "__main__":
    main()
