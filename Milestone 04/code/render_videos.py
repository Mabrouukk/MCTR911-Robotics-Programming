"""
Milestone 4 videos, rendered offscreen from the same torque-mode MuJoCo model as sim_bridge.py.

    .venv/bin/python "Milestone 04/code/render_videos.py"            both videos
    .venv/bin/python "Milestone 04/code/render_videos.py" control    only the trajectory / control video
    .venv/bin/python "Milestone 04/code/render_videos.py" sorting    only the application video

ms4_trajectory_control.mp4
    1. torque mode is real: zero torque -> the arm falls; g(q) alone -> it floats still
    2. joint-space test trajectory: PD only (sags behind), then computed torque (on the plan)
    3. task-space test trajectory: straight lines + a circle, computed torque, planned path vs flange trail
    4. a hidden 2 kg payload: PD + gravity vs computed torque
ms4_sorting_application.mp4
    the colour sorting of all six parts with computed-torque control.
"""
import json
import os
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control  # noqa: E402
import dynamics as dyn  # noqa: E402
import test_paths  # noqa: E402
from kinematics import HOME_Q, ROBOT_BASE_POS, forward_kinematics  # noqa: E402
from sim_bridge import Arm, Conveyor, build_model  # noqa: E402
from sorting_task import GRIPPER_OPEN, SortingApp  # noqa: E402

VIDEOS = os.path.join(HERE, "..", "videos")
RESULTS = os.path.join(HERE, "..", "report", "validation_results.json")
WIDTH, HEIGHT, FPS = 1280, 720, 30
PANEL_EVERY = 8
# Camera inside the fence, above the robot, so no fence post blocks the arm.
INSIDE_VIEW = dict(lookat=(0.15, 0.3, 0.6), distance=1.7, azimuth=-150, elevation=-35)


def _font(size, mono=False):
    paths = (("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf") if mono else
             ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for path in paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE, TEXT, MONO, SMALL = _font(28), _font(20), _font(15, mono=True), _font(16)
GOOD, WARN, BAD = (120, 230, 140), (255, 200, 0), (255, 110, 100)


class Video:
    """Fixed camera, frames piped into ffmpeg, with caption bar, data panel and an optional chart."""

    def __init__(self, model, filename, lookat, distance, azimuth, elevation):
        os.makedirs(VIDEOS, exist_ok=True)
        self.path = os.path.join(VIDEOS, filename)
        self.renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        self.camera = mujoco.MjvCamera()
        self.set_camera(lookat, distance, azimuth, elevation)
        self.markers = []          # (pos, radius, rgba) drawn into the 3D scene
        self._count = 0
        self._panel = None
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", self.path],
            stdin=subprocess.PIPE)

    def set_camera(self, lookat, distance, azimuth, elevation):
        self.camera.lookat[:] = lookat
        self.camera.distance = distance
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation

    def _add_markers(self):
        scene = self.renderer.scene
        for pos, radius, rgba in self.markers:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([radius, 0, 0]), np.asarray(pos, float), np.eye(3).flatten(),
                                np.asarray(rgba, np.float32))
            scene.ngeom += 1

    def frame(self, data, title, caption=(), panel=(), chart=None):
        self.renderer.update_scene(data, self.camera)
        self._add_markers()
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, WIDTH, 48 + 28 * len(caption)), fill=(0, 0, 0, 165))
        draw.text((20, 9), title, font=TITLE, fill="white")
        for i, line in enumerate(caption):
            draw.text((20, 50 + 28 * i), line, font=TEXT, fill=(230, 230, 230))
        if panel and (self._panel is None or self._count % PANEL_EVERY == 0
                      or [r[0] for r in panel] != [r[0] for r in self._panel]):
            self._panel = list(panel)
        panel = self._panel if panel else None
        self._count += 1
        if panel:
            h = 22 * len(panel) + 16
            y0 = HEIGHT - h - 12
            draw.rectangle((12, y0, 12 + 600, HEIGHT - 12), fill=(0, 0, 0, 170))
            for i, (label, value, color) in enumerate(panel):
                draw.text((24, y0 + 8 + 22 * i), label, font=MONO, fill=(200, 200, 200))
                draw.text((290, y0 + 8 + 22 * i), value, font=MONO, fill=color or "white")
        if chart:
            self._chart(draw, *chart)
        draw.text((WIDTH - 140, HEIGHT - 34), f"t = {data.time:5.1f} s", font=TEXT, fill="white")
        self.ffmpeg.stdin.write(image.tobytes())

    def _chart(self, draw, title, t, line, dots, unit, legend):
        """Small time chart, bottom right: actual (line) vs planned (dots)."""
        x0, y0, w, h = WIDTH - 470, HEIGHT - 250, 440, 190
        draw.rectangle((x0 - 10, y0 - 34, x0 + w + 10, y0 + h + 10), fill=(0, 0, 0, 170))
        draw.text((x0, y0 - 30), title, font=SMALL, fill="white")
        if len(t) < 2:
            return
        lo, hi = min(min(line), min(dots)), max(max(line), max(dots))
        pad = 0.1 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        tx = lambda v: x0 + w * (v - t[0]) / max(t[-1] - t[0], 1e-9)  # noqa: E731
        ty = lambda v: y0 + h - h * (v - lo) / (hi - lo)  # noqa: E731
        for a, b in list(zip(t, dots))[::6]:
            draw.ellipse((tx(a) - 3, ty(b) - 3, tx(a) + 3, ty(b) + 3), fill=(170, 170, 170))
        draw.line([(tx(a), ty(b)) for a, b in zip(t, line)], fill=(90, 160, 255), width=3)
        draw.text((x0, y0 + h - 18), f"{lo:+.1f} {unit}", font=SMALL, fill=(170, 170, 170))
        draw.text((x0, y0 + 2), f"{hi:+.1f} {unit}", font=SMALL, fill=(170, 170, 170))
        draw.text((x0 + w - 200, y0 + 2), legend, font=SMALL, fill=(220, 220, 220))

    def hold(self, data, seconds, *args, **kwargs):
        for _ in range(int(seconds * FPS)):
            self.frame(data, *args, **kwargs)

    def close(self):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        self.renderer.close()
        print(f"Saved {os.path.relpath(self.path)}")


class Sim:
    """The torque-mode cell; torque(t, q, dq) is called every physics step."""

    def __init__(self, payload=0.0, environment=True):
        self.model = build_model(environment=environment, payload_mass=payload)
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = WIDTH, HEIGHT
        self.data = mujoco.MjData(self.model)
        self.arm = Arm(self.model)
        self.conveyor = Conveyor(self.model) if environment else None
        self.steps_per_frame = round(1 / (FPS * self.model.opt.timestep))

    def reset(self, q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm.qpos] = q
        self.data.ctrl[self.arm.grip] = GRIPPER_OPEN
        self.arm.show_target(self.data, q)
        mujoco.mj_forward(self.model, self.data)

    def run(self, seconds, torque, on_frame, conveyor=False):
        t0 = self.data.time
        k = 0
        while self.data.time < t0 + seconds - 1e-9:
            q, dq, _ = self.arm.state(self.data)
            self.data.ctrl[self.arm.act] = torque(self.data.time - t0, q, dq)
            if conveyor:
                self.conveyor.step(self.data)
            mujoco.mj_step(self.model, self.data)
            k += 1
            if k % self.steps_per_frame == 0:
                on_frame(self.data.time - t0)


def flange_world(q):
    return forward_kinematics(q)[:3, 3] + ROBOT_BASE_POS


def err_color(mm, good=1.0, warn=10.0):
    return GOOD if mm < good else (WARN if mm < warn else BAD)


def control_video():
    sim = Sim()
    video = Video(sim.model, "ms4_trajectory_control.mp4", **INSIDE_VIEW)
    wait_q = test_paths.WAIT_Q

    # ---- 1. torque mode is real -------------------------------------------------
    sim.reset(wait_q)
    video.hold(sim.data, 2.5, "Milestone 4: the UR5e in torque mode",
               ["The motors now get torques only. No position servos, and MuJoCo's own gravity compensation is off.",
                "So the arm stays up only if our controller computes the right torques."])
    for title, caption, law in (
            ("1a. Zero torque: gravity wins",
             ["tau = 0  ->  the arm falls (brakes released, nothing holds it)."],
             lambda t, q, dq: np.zeros(6)),
            ("1b. Gravity compensation only: tau = g(q)",
             ["g(q) from our Lagrangian model (dP/dq) exactly cancels the weight of every link and the gripper,",
              "so the arm floats wherever it is, with no feedback at all."],
             lambda t, q, dq: dyn.NOMINAL.gravity(q))):
        sim.reset(wait_q)

        def frame(t, title=title, caption=caption):
            q, dq, _ = sim.arm.state(sim.data)
            drift = 1000 * np.linalg.norm(flange_world(q) - flange_world(wait_q))
            g = dyn.NOMINAL.gravity(q)
            video.frame(sim.data, title, caption, panel=[
                ("flange moved from start", f"{drift:8.1f} mm", err_color(drift)),
                ("g(q) shoulder / elbow", f"{g[1]:+6.1f} / {g[2]:+6.1f} N m", None),
                ("torque sent  shoulder / elbow", f"{sim.data.actuator_force[sim.arm.act][1]:+6.1f} / "
                                                  f"{sim.data.actuator_force[sim.arm.act][2]:+6.1f} N m", None)])
        sim.run(2.5 if "Zero" in title else 4.0, law, frame)

    # ---- 2. joint-space test: PD vs computed torque -------------------------------
    for name, caption in (
            ("pd", ["PD only: tau = Kp e + Kd e_dot. Gravity is a load it can only fight by letting the error grow:",
                    "the arm sags 2-3 deg below the plan (green ball = where the plan wants the flange)."]),
            ("ctc", ["Computed torque: tau = M(q)(q_ddot_d + Kd e_dot + Kp e) + C q_dot + g. The model gives the torque the",
                     "plan needs, the PD only fixes the rest: e_ddot + 40 e_dot + 400 e = 0 on every joint."])):
        plan = test_paths.joint_test_plan()
        sim.reset(wait_q)
        ctrl = control.make(name)
        hist = {"t": [], "q": [], "qd": []}
        worst = [0.0]

        def law(t, q, dq, plan=plan, ctrl=ctrl):
            q_d, dq_d, ddq_d, _, _, _ = plan.sample(t)
            sim.arm.show_target(sim.data, q_d)
            return ctrl(q, dq, q_d, dq_d, ddq_d)

        def frame(t, plan=plan, name=name, caption=caption, hist=hist, worst=worst):
            q, _, _ = sim.arm.state(sim.data)
            q_d, _, _, _, label, _ = plan.sample(t)
            e = 1000 * np.linalg.norm(flange_world(q_d) - flange_world(q))
            worst[0] = max(worst[0], e)
            hist["t"].append(t)
            hist["q"].append(np.degrees(q[1]))
            hist["qd"].append(np.degrees(q_d[1]))
            video.frame(sim.data, f"2. Joint-space trajectory, {control.make(name).name}", caption, panel=[
                ("segment", label, None),
                ("flange error now", f"{e:8.2f} mm", err_color(e)),
                ("worst so far", f"{worst[0]:8.2f} mm", err_color(worst[0])),
                ("joint error (max)", f"{np.degrees(np.abs(q_d - q)).max():8.3f} deg", None)],
                chart=("shoulder lift q2 [deg]", hist["t"], hist["q"], hist["qd"], "deg", "— actual   • planned"))
        sim.run(plan.T, law, frame)

    # ---- 3. task-space test with computed torque --------------------------------
    plan = test_paths.task_test_plan()
    sim.reset(wait_q)
    video.set_camera((0.35, 0.3, 0.6), 1.4, -150, -40)
    ctrl = control.make("ctc")
    video.markers = [(flange_world(plan.sample(t)[0]), 0.004, (1, 0.85, 0, 0.8)) for t in np.linspace(0, plan.T, 160)]
    trail = []
    worst = [0.0]

    def law3(t, q, dq):
        q_d, dq_d, ddq_d, _, _, _ = plan.sample(t)
        sim.arm.show_target(sim.data, q_d)
        return ctrl(q, dq, q_d, dq_d, ddq_d)

    def frame3(t):
        q, _, _ = sim.arm.state(sim.data)
        q_d, _, _, _, label, _ = plan.sample(t)
        p = flange_world(q)
        if not trail or np.linalg.norm(p - trail[-1]) > 0.006:
            trail.append(p)
            video.markers.append((p, 0.0035, (0.2, 0.55, 1.0, 0.9)))
        e = 1000 * np.linalg.norm(flange_world(q_d) - p)
        worst[0] = max(worst[0], e)
        video.frame(sim.data, "3. Task-space trajectory, computed torque",
                    ["Straight lines and a circle for the flange (yellow), turned into joints by IK, J^-1 and",
                     "J^-1(a - J_dot q_dot) (Milestone 3), then into torques. Blue = where the flange really went."],
                    panel=[("segment", label, None), ("flange error now", f"{e:8.3f} mm", err_color(e, 0.5, 5)),
                           ("worst so far", f"{worst[0]:8.3f} mm", err_color(worst[0], 0.5, 5))])
    sim.run(plan.T, law3, frame3)
    video.markers = []

    # ---- 4. hidden payload --------------------------------------------------------
    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    heavy = Sim(payload=2.0)
    video.close()
    video = None
    return heavy, results


def payload_video_part(heavy, results):
    """Appended to the control video: the 2 kg payload the controller does not know about."""
    path = os.path.join(VIDEOS, "ms4_part4.mp4")
    video = Video(heavy.model, "ms4_part4.mp4", **INSIDE_VIEW)
    for name in ("pdg", "ctc"):
        plan = test_paths.joint_test_plan()
        heavy.reset(test_paths.WAIT_Q)
        ctrl = control.make(name)
        worst = [0.0]

        def law(t, q, dq, plan=plan, ctrl=ctrl):
            q_d, dq_d, ddq_d, _, _, _ = plan.sample(t)
            heavy.arm.show_target(heavy.data, q_d)
            return ctrl(q, dq, q_d, dq_d, ddq_d)

        def frame(t, plan=plan, name=name, worst=worst):
            q, _, _ = heavy.arm.state(heavy.data)
            q_d = plan.sample(t)[0]
            e = 1000 * np.linalg.norm(flange_world(q_d) - flange_world(q))
            worst[0] = max(worst[0], e)
            video.frame(heavy.data, f"4. Hidden 2 kg payload (orange ball), {control.make(name).name}",
                        ["The model thinks the gripper is empty. The payload is a force the model does not know,",
                         "so only the feedback part can fight it: the error grows, but the arm still follows the plan."],
                        panel=[("flange error now", f"{e:8.2f} mm", err_color(e)),
                               ("worst so far", f"{worst[0]:8.2f} mm", err_color(worst[0]))])
        heavy.run(plan.T, law, frame)
    rob = results.get("robustness", {})
    lines = []
    for label in rob:
        lines.append(f"{label:40s} PD+g {rob[label]['pdg']['flange_err_max_mm']:6.2f} mm   "
                     f"computed torque {rob[label]['ctc']['flange_err_max_mm']:6.2f} mm")
    video.hold(heavy.data, 6.0, "Summary: worst flange error on the joint-space test (validate_control.py)",
               lines or ["run validate_control.py first for the numbers"])
    video.close()
    return path


def sorting_video():
    sim = Sim()
    video = Video(sim.model, "ms4_sorting_application.mp4", lookat=(0.15, 0.3, 0.6), distance=1.7,
                  azimuth=-150, elevation=-35)
    sim.reset(HOME_Q)
    app = SortingApp(HOME_Q)
    ctrl = control.make("ctc")
    state = {"step": "to the pick station", "worst_move": 0.0}

    def law(t, q, dq):
        q_d, dq_d, ddq_d, gripper, event = app.update(sim.data.time, sim.conveyor.station_color(sim.data))
        if event:
            state["step"] = event
        sim.arm.show_target(sim.data, q_d)
        sim.data.ctrl[sim.arm.grip] = gripper
        state["q_d"], state["dq_d"] = q_d, dq_d
        return ctrl(q, dq, q_d, dq_d, ddq_d)

    def frame(t):
        q, _, _ = sim.arm.state(sim.data)
        # The plan was sampled before the physics step, q is after it: move the plan on by one step
        # so both are from the same instant (else 2 ms of motion, up to 2 mm, reads as "error").
        q_now = state["q_d"] + state["dq_d"] * sim.model.opt.timestep if "q_d" in state else q
        e = 1000 * np.linalg.norm(flange_world(q_now) - flange_world(q))
        grasp = any((app.label or "").startswith(g) for g in ("close the gripper", "lift", "open the gripper"))
        if not grasp:
            state["worst_move"] = max(state["worst_move"], e)
        az = -150 + 25 * np.sin(2 * np.pi * sim.data.time / 60)
        video.set_camera((0.15, 0.3, 0.6), 1.7, az, -35)
        tau = sim.data.actuator_force[sim.arm.act]
        counts = ", ".join(f"{c}: {n}" for c, n in app.sorted.items())
        video.frame(sim.data, "Colour sorting under computed-torque control",
                    ["Pick and place go straight up and down (task space); transfers are joint-space quintics.",
                     f"Step: {(app.label or '')}   |   sorted: {counts}"],
                    panel=[("flange error now", f"{e:8.2f} mm", err_color(e)),
                           ("worst while moving", f"{state['worst_move']:8.2f} mm", err_color(state["worst_move"])),
                           ("torque shoulder / elbow", f"{tau[1]:+6.1f} / {tau[2]:+6.1f} N m", None),
                           ("gripper", "on the part" if grasp else "free", None)])
        if sum(app.sorted.values()) >= 6:
            raise StopIteration
    try:
        sim.run(120.0, law, frame, conveyor=True)
    except StopIteration:
        pass
    video.hold(sim.data, 3.0, "All six parts sorted", [f"sorted: {app.sorted} in {sim.data.time:.0f} s"])
    video.close()


def main():
    which = sys.argv[1:] or ["control", "sorting"]
    if "control" in which:
        heavy, results = control_video()
        part4 = payload_video_part(heavy, results)
        main_path = os.path.join(VIDEOS, "ms4_trajectory_control.mp4")
        tmp = os.path.join(VIDEOS, "_parts.txt")
        with open(tmp, "w") as f:
            f.write(f"file '{os.path.abspath(main_path)}'\nfile '{os.path.abspath(part4)}'\n")
        joined = os.path.join(VIDEOS, "_joined.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", tmp,
                        "-c", "copy", joined], check=True)
        os.replace(joined, main_path)
        os.remove(tmp)
        os.remove(part4)
        print(f"Joined part 4 into {os.path.relpath(main_path)}")
    if "sorting" in which:
        sorting_video()


if __name__ == "__main__":
    main()
