"""
Milestone 3 videos, rendered offscreen from the same MuJoCo model as sim_bridge.py.

    .venv/bin/python "Milestone 03/code/render_videos.py"              both videos
    .venv/bin/python "Milestone 03/code/render_videos.py" validation   only the kinematics video
    .venv/bin/python "Milestone 03/code/render_videos.py" gui          only the environment video

ms3_kinematics_validation.mp4
    1. joint angles in -> sensed flange pose vs FK, and IK of the sensed pose back to the angles
    2. joint rates and accelerations in -> sensed flange twist / acceleration vs J q_dot, J q_ddot + J_dot q_dot,
       and J^-1 of the sensed values back to the joint rates / accelerations
    3. a flange path in task space -> IK, J^-1, J^-1(...) -> joint commands; sensed path vs planned
       (angles only, then with velocity feed-forward)
    4. the wrist singularity: what happens to J^-1 as q5 -> 0
ms3_gui_environment.mp4
    the full cell (fence, people, chairs, tools, conveyors...) while the arm sorts parts.
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
import task_path  # noqa: E402
import velocity_kinematics as vk  # noqa: E402
from ee_sensors import read_ee  # noqa: E402
from kinematic_checks import kinematics_check  # noqa: E402
from kinematics import HOME_Q, ROBOT_BASE_POS, forward_kinematics, inverse_kinematics, wrap_angle  # noqa: E402
from sim_bridge import Arm, Conveyor, build_model  # noqa: E402
from sorting_task import GRIPPER_OPEN, SortingController  # noqa: E402

VIDEOS = os.path.join(HERE, "..", "videos")
RESULTS = os.path.join(HERE, "..", "report", "validation_results.json")
WIDTH, HEIGHT, FPS = 1280, 720, 30
KV_OVER_KP = 0.2      # position servo lag kd/kp of every UR5e joint in the model


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


def sci(x):
    return f"{x:.0e}"


def vec(x, digits=3):
    return "[" + " ".join(f"{c:+.{digits}f}" for c in x) + "]"


class Video:
    """Fixed camera, frames piped into ffmpeg, with caption bar, data panel and an optional chart."""

    def __init__(self, model, filename, lookat, distance, azimuth, elevation):
        os.makedirs(VIDEOS, exist_ok=True)
        self.path = os.path.join(VIDEOS, filename)
        self.renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        self.camera = mujoco.MjvCamera()
        self.set_camera(lookat, distance, azimuth, elevation)
        self.markers = []          # (pos, radius, rgba) drawn into the 3D scene
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", self.path],
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

    def frame(self, data, title, caption=(), panel=(), chart=None, t_label=None):
        self.renderer.update_scene(data, self.camera)
        self._add_markers()
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image, "RGBA")
        # Caption bar: what is being tested and what the numbers say.
        draw.rectangle((0, 0, WIDTH, 48 + 28 * len(caption)), fill=(0, 0, 0, 165))
        draw.text((20, 9), title, font=TITLE, fill="white")
        for i, line in enumerate(caption):
            draw.text((20, 50 + 28 * i), line, font=TEXT, fill=(230, 230, 230))
        # Data panel, bottom left: (label, value, colour) rows in a monospace font.
        if panel:
            h = 22 * len(panel) + 16
            y0 = HEIGHT - h - 12
            draw.rectangle((12, y0, 12 + 640, HEIGHT - 12), fill=(0, 0, 0, 170))
            for i, (label, value, color) in enumerate(panel):
                draw.text((24, y0 + 8 + 22 * i), label, font=MONO, fill=(200, 200, 200))
                draw.text((290, y0 + 8 + 22 * i), value, font=MONO, fill=color or "white")
        if chart:
            self._chart(draw, *chart)
        draw.text((WIDTH - 140, HEIGHT - 34), t_label or f"t = {data.time:5.1f} s", font=TEXT, fill="white")
        self.ffmpeg.stdin.write(image.tobytes())

    def _chart(self, draw, title, t, line, dots, unit):
        """Small time chart, bottom right: sensed value (line) vs equation (dots)."""
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
        draw.line([(tx(a), ty(b)) for a, b in zip(t, line)], fill=(90, 160, 255), width=3)
        for a, b in list(zip(t, dots))[::6]:
            draw.ellipse((tx(a) - 3, ty(b) - 3, tx(a) + 3, ty(b) + 3), fill=(255, 160, 60))
        draw.text((x0, y0 + h - 18), f"{lo:+.2f} {unit}", font=SMALL, fill=(170, 170, 170))
        draw.text((x0, y0 + 2), f"{hi:+.2f} {unit}", font=SMALL, fill=(170, 170, 170))
        draw.text((x0 + w - 230, y0 + 2), "— sensed   • equation", font=SMALL, fill=(220, 220, 220))

    def hold(self, data, seconds, *args, **kwargs):
        for _ in range(int(seconds * FPS)):
            self.frame(data, *args, **kwargs)

    def close(self):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        self.renderer.close()
        print(f"Saved {os.path.relpath(self.path)}")


class Sim:
    """The cell, stepped like sim_bridge.py; ctrl(t) is called every physics step."""

    def __init__(self):
        self.model = build_model()
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = WIDTH, HEIGHT
        self.data = mujoco.MjData(self.model)
        self.arm = Arm(self.model)
        self.conveyor = Conveyor(self.model)
        self.steps_per_frame = round(1 / (FPS * self.model.opt.timestep))
        self.reset(HOME_Q)

    def reset(self, q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm.qpos] = q
        self.data.ctrl[self.arm.act] = q
        self.data.ctrl[self.arm.grip] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)

    def sense(self):
        mujoco.mj_forward(self.model, self.data)
        q, dq, ddq = self.arm.state(self.data)
        ee = read_ee(self.model, self.data)
        return q, dq, ddq, ee, kinematics_check(q, dq, ddq, ee)

    def run(self, seconds, ctrl, on_frame, conveyor=False):
        t0 = self.data.time
        k = 0
        while self.data.time < t0 + seconds - 1e-9:
            self.data.ctrl[self.arm.act] = ctrl(self.data.time - t0)
            if conveyor:
                self.conveyor.step(self.data)
            mujoco.mj_step(self.model, self.data)
            k += 1
            if k % self.steps_per_frame == 0:
                on_frame(self.data.time - t0)


def blend(q_from, q_to, duration):
    """Smooth move with velocity feed-forward, then hold."""
    def ctrl(t):
        s = min(t / duration, 1.0)
        b = 0.5 - 0.5 * np.cos(np.pi * s)
        b_dot = 0.5 * np.pi / duration * np.sin(np.pi * s) if s < 1 else 0.0
        return q_from + b * (q_to - q_from) + KV_OVER_KP * b_dot * (q_to - q_from)
    return ctrl


def validation_video():
    sim = Sim()
    # Camera inside the fence, above it, so no fence post blocks the arm.
    video = Video(sim.model, "ms3_kinematics_validation.mp4", lookat=(0.1, 0.3, 0.6), distance=1.6,
                  azimuth=-150, elevation=-38)

    # --- Title card -------------------------------------------------------------------------
    video.hold(sim.data, 4.0, "Milestone 3: UR5e velocity and acceleration kinematics",
               ["We put joint angles, rates and accelerations into the MuJoCo simulator, read its flange",
                "sensors, and compare with our equations. Then the reverse: from the sensed flange values",
                "back to the joint values. Tolerance for a pass: 1e-6 (m, rad, m/s, rad/s), 1e-5 for accelerations."],
               t_label=" ")

    # --- Part 1: joint angles -> pose ---------------------------------------------------------
    poses = [HOME_Q, np.array([-1.2, -1.9, 1.8, -1.5, -1.57, 0.5]), np.array([-2.6, -1.3, 1.4, -2.0, -1.2, -0.8]),
             np.array([-1.8, -2.2, 2.2, -0.6, -2.0, 1.2])]
    q_cmd = HOME_Q
    for k, q_target in enumerate(poses):
        ctrl = blend(q_cmd, q_target, 1.5)
        q_cmd = q_target

        def frame(t, q_target=q_target, k=k):
            q, dq, ddq, ee, c = sim.sense()
            T = forward_kinematics(q)
            T_s = np.eye(4)
            T_s[:3, :3], T_s[:3, 3] = ee["R"], ee["p"]
            sols = inverse_kinematics(T_s)
            q_ik = sols[np.argmin(np.abs(wrap_angle(sols - q)).max(axis=1))]
            video.frame(sim.data, f"1. Position kinematics: joint angles in (pose {k + 1} of {len(poses)})", [
                "Forward: the simulator's flange sensor vs our FK(q) from the DH table.",
                "Inverse: our analytical IK of the SENSED flange pose gives back the joint angles.",
                f"Result: they agree to {sci(max(c['fk_pos'], 1e-16))} m and {sci(max(c['ik'], 1e-16))} rad (round-off): PASS"], [
                ("joint angles q (deg)", vec(np.degrees(q), 1), None),
                ("sensed flange p (m)", vec(ee["p"], 4), None),
                ("FK(q) p (m)", vec(T[:3, 3], 4), None),
                ("|difference| position", f"{c['fk_pos']:.1e} m", GOOD),
                ("orientation difference", f"{c['fk_rot']:.1e} rad", GOOD),
                ("IK(sensed pose) (deg)", vec(np.degrees(q_ik), 1), None),
                ("|IK - q| (max joint)", f"{c['ik']:.1e} rad", GOOD),
            ])
        sim.run(3.0, ctrl, frame)

    # --- Part 2: joint rates and accelerations -> twist and acceleration ---------------------
    amp = np.radians([40, 25, 30, 40, 40, 60])
    freq = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    q_start = q_cmd
    sim.run(2.0, blend(q_start, HOME_Q, 2.0), lambda t: video.frame(
        sim.data, "2. Velocity and acceleration kinematics", ["Moving back to home before the joint-space test..."]))
    hist = {"t": [], "s": [], "e": []}
    worst = dict.fromkeys(("fv_lin", "fv_ang", "fa_lin", "fa_ang", "iv", "ia"), 0.0)

    def frame2(t):
        q, dq, ddq, ee, c = sim.sense()
        for k in worst:
            worst[k] = max(worst[k], c[k])
        _, v, w, a, alpha = c["pred"]
        hist["t"].append(t), hist["s"].append(ee["v"][2]), hist["e"].append(v[2])
        keep = [i for i, x in enumerate(hist["t"]) if x > t - 4.0]
        video.frame(sim.data, "2. Velocity and acceleration kinematics: joint rates and accelerations in", [
            "Every joint follows a sine wave, so q_dot and q_ddot change all the time.",
            "Forward: sensed flange twist vs J q_dot, sensed acceleration vs J q_ddot + J_dot q_dot.",
            "Inverse: J^-1 of the sensed twist / acceleration gives back q_dot / q_ddot.  All agree: PASS"], [
            ("joint rates q_dot (rad/s)", vec(dq, 2), None),
            ("joint accel. q_ddot (rad/s2)", vec(ddq, 2), None),
            ("sensed v (m/s)", vec(ee["v"]), None),
            ("J q_dot   (m/s)", vec(v), None),
            ("sensed w (rad/s)", vec(ee["w"]), None),
            ("J q_dot   (rad/s)", vec(w), None),
            ("sensed a (m/s2)", vec(ee["a"]), None),
            ("J q_ddot + J_dot q_dot", vec(a), None),
            ("worst fwd v / w / a / alpha",
             f"{sci(worst['fv_lin'])} {sci(worst['fv_ang'])} {sci(worst['fa_lin'])} {sci(worst['fa_ang'])}", GOOD),
            ("worst inverse q_dot / q_ddot", f"{sci(worst['iv'])} rad/s  {sci(worst['ia'])} rad/s2", GOOD),
        ], chart=("flange vertical velocity v_z, last 4 s", [hist["t"][i] for i in keep],
                  [hist["s"][i] for i in keep], [hist["e"][i] for i in keep], "m/s"))
    sim.run(10.0, lambda t: HOME_Q + amp * np.sin(2 * np.pi * freq * t), frame2)

    # --- Part 3: task-space path -> IK / J^-1 -> joint commands ------------------------------
    path = task_path.plan_joint_path(sim.model.opt.timestep)
    video.set_camera((0.35, 0.3, 0.6), 1.5, -150, -50)
    planned_dots = [(p + ROBOT_BASE_POS, 0.006, (1.0, 0.85, 0.0, 0.9)) for p in path["p"][::50]]
    for run, feedforward in ((1, False), (2, True)):
        sim.reset(path["q"][0])
        trail, worst_err = [], [0.0]

        def ctrl(t, feedforward=feedforward):
            i = min(int(round(t / sim.model.opt.timestep)), len(path["t"]) - 1)
            ff = KV_OVER_KP * path["dq"][i] if (feedforward and t <= task_path.T_TOTAL) else 0.0
            return path["q"][i] + ff

        def frame3(t, run=run, feedforward=feedforward):
            q, dq, ddq, ee, c = sim.sense()
            p_ref, v_ref, _ = task_path.flange_target(t)
            err = np.linalg.norm(ee["p"] - p_ref)
            worst_err[0] = max(worst_err[0], err)
            trail.append(ee["p"] + ROBOT_BASE_POS)
            video.markers = planned_dots + [(p, 0.004, (0.2, 0.5, 1.0, 1.0)) for p in trail[::2]]
            dq_inv = vk.inverse_velocity(q, v_ref, np.zeros(3))
            caption = (["Planned flange path (yellow): line + circle, gripper down. IK, J^-1 v and J^-1 (a - J_dot q_dot)",
                        "turn it into joint commands. Run 1 sends only the angles: the position servos lag",
                        f"(kd/kp = 0.2 s), so the flange (blue) cuts inside the path. Worst so far {worst_err[0] * 1000:.1f} mm."]
                       if not feedforward else
                       ["Run 2 also sends (kd/kp) x J^-1 v: our inverse velocity cancels the servo lag",
                        "(velocity feed-forward). The flange (blue) now sits on the planned path:",
                        f"worst so far {worst_err[0] * 1000:.2f} mm, tolerance 5 mm: PASS."])
            video.frame(sim.data, f"3. Inverse kinematics driving the robot, run {run}: "
                        f"{'angles only' if not feedforward else 'angles + velocity feed-forward'}", caption, [
                ("planned flange p (m)", vec(p_ref), None),
                ("sensed flange p (m)", vec(ee["p"]), None),
                ("distance to plan", f"{err * 1000:6.2f} mm", GOOD if err < 0.005 else BAD),
                ("planned v (m/s)", vec(v_ref), None),
                ("sensed v (m/s)", vec(ee["v"]), None),
                ("J^-1 [v; 0] (rad/s)", vec(dq_inv, 2), None),
                ("sigma_min(J)", f"{c['sigma_min']:.3f} m/rad (far from singular)", None),
            ])
        sim.run(task_path.T_TOTAL + 1.0, ctrl, frame3)
        video.hold(sim.data, 1.5, f"3. Run {run} done", [
            f"Worst distance between the sensed and the planned flange: {worst_err[0] * 1000:.2f} mm "
            f"({'FAIL, the servo lag' if not feedforward else 'PASS'})."], t_label=" ")
    video.markers = []

    # --- Part 4: singularity -----------------------------------------------------------------
    video.set_camera((0.0, 0.2, 0.75), 1.3, -150, -30)
    sim.reset(HOME_Q)
    q_sing = HOME_Q.copy()
    q_sing[4] = -np.radians(0.3)
    x_dot = np.array([0.05, 0, 0, 0, 0, 0])

    def frame4(t):
        q, dq, ddq, ee, c = sim.sense()
        J = vk.jacobian(q)
        plain = np.abs(np.linalg.solve(J, x_dot)).max()
        damped, lam = vk.inverse_velocity(q, x_dot[:3], x_dot[3:], return_damping=True)
        near = c["sigma_min"] < vk.SINGULAR_EPS
        video.frame(sim.data, "4. Singularity: the wrist lines up (q5 -> 0)", [
            "det J = a2 a3 sin(q3) sin(q5) (a2 cos q2 + a3 cos(q2+q3) + d5 sin(q2+q3+q4)) -> 0 as q5 -> 0.",
            "Joints 4 and 6 then turn about the same line, so one flange direction is lost and plain J^-1",
            "asks for huge joint speeds. Damped least squares (lambda > 0 near the singularity) keeps them bounded."], [
            ("q5 (deg)", f"{np.degrees(q[4]):7.2f}", None),
            ("det J (closed form)", f"{vk.determinant_closed_form(q):+.2e}", WARN if near else None),
            ("sigma_min(J)", f"{c['sigma_min']:.4f} m/rad" + ("   NEAR SINGULAR" if near else ""), WARN if near else None),
            ("plain J^-1, 5 cm/s along x", f"{plain:9.2f} rad/s" + ("  > UR5e limit pi" if plain > np.pi else ""),
             BAD if plain > np.pi else None),
            ("damped, 5 cm/s along x", f"{np.abs(damped).max():9.2f} rad/s   (lambda = {lam:.3f})", GOOD),
        ])
    sim.run(7.0, blend(HOME_Q, q_sing, 6.0), frame4)

    # --- Summary -----------------------------------------------------------------------------
    res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else None
    lines = ["Our equations vs the simulator's sensors, forward and inverse, position / velocity / acceleration:",
             "position, velocity and acceleration all agree to 1e-13 or better (round-off): PASS.",
             "Inverse kinematics driving the robot along a task-space path: within 0.6 mm with velocity feed-forward.",
             "Near a singularity: plain J^-1 blows up, damped least squares stays within the UR5e speed limit."]
    if res:
        r = res["random_states"]
        lines[1] = (f"Worst over 1000 random states: FK {sci(r['fk_pos'])} m, J q_dot {sci(r['fv_lin'])} m/s, "
                    f"J q_ddot + J_dot q_dot {sci(r['fa_lin'])} m/s2, inverse {sci(r['ia'])} rad/s2: PASS.")
    video.set_camera((0.1, 0.3, 0.6), 1.6, -150, -38)
    video.hold(sim.data, 6.0, "Summary", lines, t_label=" ")
    video.close()


def gui_video():
    sim = Sim()
    video = Video(sim.model, "ms3_gui_environment.mp4", lookat=(0.0, 0.1, 0.6), distance=4.2,
                  azimuth=-120, elevation=-28)
    sorter = SortingController(HOME_Q)
    state = {"q": HOME_Q, "grip": GRIPPER_OPEN, "step": "waiting for a part"}
    shots = [
        # (seconds, lookat, distance, azimuth from -> to, elevation, title, caption)
        (14.0, (-0.2, 0.1, 0.5), 5.4, (-150, -115), -30, "The colour-sorting cell (Milestone 3 environment)",
         ["Guarded UR5e cell: fence with an open gate, light curtain on the infeed, warning sign.",
          "Infeed conveyor -> cell conveyor -> colour sensor -> UR5e -> red / green / blue bins."]),
        (8.0, (0.1, -0.9, 0.85), 3.4, (95, 115), -20, "Operator station",
         ["Seated operator at the HMI desk (dashboard with part counters, e-stop),",
          "control cabinet with a stack light (green = running), shelving rack and pallet behind."]),
        (8.0, (2.0, 0.2, 0.95), 2.1, (-75, -50), -30, "Maintenance workbench",
         ["Technician at the workbench: toolbox, wrench, screwdriver, hammer, spare finger pad,",
          "tool board on the wall; workbench chair and a stool; fire extinguisher by the fence."]),
        (8.0, (-1.6, 0.0, 0.6), 3.2, (-10, 15), -25, "Logistics side",
         ["Infeed conveyor with hopper and drive motor, pallet with boxes, shelving rack,",
          "and a worker walking along the fence with a clipboard."]),
        (14.0, (0.1, 0.3, 0.6), 1.6, (-155, -120), -38, "Inside the cell: the arm at work",
         ["The UR5e picks each part the colour sensor reports and drops it in the bin of its colour",
          "(Milestone 2 sorting logic, now with gravity compensation and the flange sensors of Milestone 3)."]),
    ]
    rate_steps = round(1 / (50 * sim.model.opt.timestep))
    step = [0]

    def ctrl(t):
        if step[0] % rate_steps == 0:
            q, g, event = sorter.update(sim.data.time, sim.conveyor.station_color(sim.data))
            state["q"], state["grip"] = q, g
            if event:
                state["step"] = event.split(" (")[0]
        step[0] += 1
        sim.data.ctrl[sim.arm.grip] = state["grip"]
        return state["q"]

    for seconds, lookat, dist, (az0, az1), elev, title, caption in shots:
        def frame(t, lookat=lookat, dist=dist, az0=az0, az1=az1, elev=elev, title=title, caption=caption,
                  seconds=seconds):
            video.set_camera(lookat, dist, az0 + (az1 - az0) * t / seconds, elev)
            counts = ", ".join(f"{c}: {n}" for c, n in sorter.sorted.items())
            video.frame(sim.data, title, caption + [f"Arm: {state['step']}   |   sorted so far: {counts}"])
        sim.run(seconds, ctrl, frame, conveyor=True)
    video.close()


def main():
    which = sys.argv[1:] or ["validation", "gui"]
    if "validation" in which:
        validation_video()
    if "gui" in which:
        gui_video()


if __name__ == "__main__":
    main()
