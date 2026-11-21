"""
MuJoCo simulator for Challenge 3, in TORQUE mode, running natively on the Mac.

It plays the part of the course mujoco_ros2 bridge with {"control_mode": "TORQUE"}:
  - the 7 arm actuators are torque motors: the first 7 values on /joint_commands are
    joint torques [N m], applied as they are (limited to the Panda's 87 / 12 N m);
  - gravity acts on the arm (no gravity compensation inside the simulator);
  - the joint damping of panda.xml (1 N m s/rad) stays: it is friction our model leaves out;
  - like a real Panda, the brakes hold the arm at Q_START until the first command arrives
    (and again if commands stop for 0.1 s).

The planned hand path is drawn as yellow dots and the hand pose at Q_END as an RGB frame.
When the trajectory is done (plus HOLD seconds) the simulator prints how closely the arm
tracked the plan and saves the log to results/sim_<tag>.npz.

The node's infrastructure (which we must not change) moves along the plan one sample per
timer tick. A Python timer at 250 Hz often fires a little late, so the plan then plays back
a bit slower than real time. To measure tracking and not timing, the reference is the plan
sample of the command being applied: command number n -> plan time (n - 1) x CONTROL_DT.
The playback speed (plan time / real time) is printed separately.

    --mass-scale link3=1.2     make a link 20 % heavier than the controller's model thinks
                               (the Team 5 debate question); can be given several times.
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ros_free_import import load_controller_node  # noqa: E402

cn = load_controller_node()

SCENE = os.path.join(HERE, "..", "..", "Challenge 01", "model", "scene.xml")
RESULTS = os.path.join(HERE, "..", "results")
PORT = 9994                # Challenge 2 uses 9997, Milestone 4 9995
CMD_SIZE = 9               # 7 torques + gripper + 1 spare, like the course bridge
STATE_HZ = 250.0
LOG_PERIOD = 0.004
REPORT_PERIOD = 1.0
HOLD = 2.0                 # s after the plan ends that we keep measuring
BRAKE_AFTER = 0.1
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]


def plan():
    return cn.plan_trajectory(cn.Q_START, cn.Q_END, cn.TRAJ_DURATION, cn.CONTROL_DT)


def build_model(mass_scale=None, show_plan=True):
    """Panda in torque mode. mass_scale: {"link3": 1.2, ...} changes the real arm, not the model."""
    spec = mujoco.MjSpec.from_file(SCENE)
    for i, act in enumerate(spec.actuators[:7]):
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.gainprm[:3] = [1.0, 0.0, 0.0]
        act.biasprm[:3] = [0.0, 0.0, 0.0]
        act.ctrlrange = [-cn.TORQUE_LIMIT[i], cn.TORQUE_LIMIT[i]]
        act.forcerange = [-cn.TORQUE_LIMIT[i], cn.TORQUE_LIMIT[i]]
    for name, factor in (mass_scale or {}).items():
        body = spec.body(name)
        body.mass *= factor
        body.fullinertia = [v * factor for v in body.fullinertia]
    if show_plan:
        _, q_plan, _, _ = plan()
        for q in q_plan[::25]:
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0, 0],
                                    pos=cn.forward_kinematics(q)[:3, 3], rgba=[1, 0.85, 0, 0.8],
                                    contype=0, conaffinity=0)
        T_end = cn.forward_kinematics(cn.Q_END)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T_end[:3, :3].flatten())
        frame = spec.worldbody.add_body(name="goal", pos=T_end[:3, 3], quat=quat)
        for axis, rgba in zip(np.eye(3), ([1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5])):
            frame.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, *(0.1 * axis)],
                           size=[0.005, 0, 0], rgba=rgba, contype=0, conaffinity=0)
    return spec.compile()


def reset_to_start(model, data):
    mujoco.mj_resetData(model, data)
    for i, name in enumerate(ARM_JOINTS):
        data.qpos[model.joint(name).qposadr[0]] = cn.Q_START[i]
    for name in FINGER_JOINTS:
        data.qpos[model.joint(name).qposadr[0]] = 0.04
    data.ctrl[:7] = 0.0
    data.ctrl[7] = cn.GRIPPER_CTRL
    mujoco.mj_forward(model, data)


class TrackingLog:
    """Planned vs actual joints and hand position against time since the first command."""

    def __init__(self):
        self.t_plan, self.q_plan, _, _ = plan()
        self.T = self.t_plan[-1]
        self.rows = []

    def reference(self, tau):
        q = np.array([np.interp(tau, self.t_plan, self.q_plan[:, j]) for j in range(7)])
        return q, cn.forward_kinematics(q)[:3, 3]

    def record(self, tau, q, hand, torque):
        q_ref, hand_ref = self.reference(tau)
        self.rows.append(np.concatenate([[tau], q, q_ref, hand, hand_ref, torque]))

    def arrays(self):
        a = np.array(self.rows)
        return dict(t=a[:, 0], q=a[:, 1:8], q_ref=a[:, 8:15], hand=a[:, 15:18], hand_ref=a[:, 18:21], tau=a[:, 21:28])

    def summary(self):
        a = self.arrays()
        e = np.degrees(np.abs(a["q"] - a["q_ref"]))
        hand = 1000 * np.linalg.norm(a["hand"] - a["hand_ref"], axis=1)
        moving = a["t"] <= self.T
        final = a["t"] >= a["t"][-1] - 0.05
        return {"max_joint_err_deg": float(e[moving].max()), "worst_joint": int(np.argmax(e[moving].max(axis=0))) + 1,
                "rms_joint_err_deg": float(np.sqrt(np.mean(e[moving] ** 2))),
                "max_hand_err_mm": float(hand[moving].max()), "final_joint_err_deg": float(e[final].max()),
                "final_hand_err_mm": float(hand[final].mean())}


class Bridge:
    """TCP server that panda_bridge_node.py (inside Docker) connects to. One JSON object per line."""

    def __init__(self, port):
        self._lock = threading.Lock()
        self._command = None
        self.last_command_wall = None
        self._conn = None
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", port))
        self._server.listen(1)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"Waiting for the ROS 2 bridge node on port {port}...")

    def _accept_loop(self):
        while True:
            conn, addr = self._server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"ROS 2 bridge connected from {addr}")
            self._conn = conn
            try:
                for line in conn.makefile("r"):
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    with self._lock:
                        self._command = msg
                        self.last_command_wall = time.time()
            except OSError:
                pass
            print("ROS 2 bridge disconnected")
            self._conn = None

    def pop_command(self):
        with self._lock:
            command, self._command = self._command, None
        return command

    def send_state(self, state):
        conn = self._conn
        if conn is None:
            return
        try:
            conn.sendall((json.dumps(state) + "\n").encode())
        except OSError:
            pass


def parse_mass_scale(items):
    out = {}
    for item in items or []:
        name, factor = item.split("=")
        out[name] = float(factor)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="no viewer; exit after the summary")
    parser.add_argument("--mass-scale", action="append", help="e.g. link3=1.2 (repeatable)")
    parser.add_argument("--tag", default="run", help="name of the saved log: results/sim_<tag>.npz")
    args = parser.parse_args()
    mass_scale = parse_mass_scale(args.mass_scale)

    model = build_model(mass_scale)
    data = mujoco.MjData(model)
    arm_qpos = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    arm_dof = [model.joint(j).dofadr[0] for j in ARM_JOINTS]
    finger_qpos = [model.joint(j).qposadr[0] for j in FINGER_JOINTS]
    finger_dof = [model.joint(j).dofadr[0] for j in FINGER_JOINTS]
    hand = model.body("hand").id
    reset_to_start(model, data)
    print("Control mode: TORQUE (the first 7 command values are joint torques in N m), gravity ON")
    if mass_scale:
        print(f"Real arm differs from the model: {mass_scale}")

    bridge = Bridge(PORT)
    log = TrackingLog()
    state_every = max(1, int(round(1.0 / (STATE_HZ * model.opt.timestep))))
    q_lock = data.qpos[arm_qpos].copy()          # brakes on at the start
    t0 = None
    wall_t0 = None
    n_cmd = 0                                    # torque commands received = plan samples played
    last_log = last_report = -1e9
    done = False
    step = 0

    def step_once():
        nonlocal t0, wall_t0, last_log, last_report, done, q_lock, step, n_cmd
        command = bridge.pop_command()
        if command is not None:
            cmd = command.get("cmd", [])
            if len(cmd) == CMD_SIZE:
                lo, hi = model.actuator_ctrlrange[:8, 0], model.actuator_ctrlrange[:8, 1]
                data.ctrl[:8] = np.clip(cmd[:8], lo, hi)
                q_lock = None
                n_cmd += 1
                if t0 is None:
                    t0, wall_t0 = data.time, time.time()
                    print(f"First torque command: brakes released, trajectory clock started ({log.T:.2f} s planned)")
        stale = bridge.last_command_wall is None or time.time() - bridge.last_command_wall > BRAKE_AFTER
        q, dq = data.qpos[arm_qpos], data.qvel[arm_dof]
        if stale and q_lock is None:
            q_lock = q.copy()
        if q_lock is not None:
            # Brakes: the joints are locked where they are (a real brake does not sag).
            data.ctrl[:7] = 0.0
            data.qpos[arm_qpos] = q_lock
            data.qvel[arm_dof] = 0.0

        mujoco.mj_step(model, data)
        step += 1
        if step % state_every == 0:
            bridge.send_state({"q": data.qpos[arm_qpos + finger_qpos].tolist(),
                               "dq": data.qvel[arm_dof + finger_dof].tolist()})

        if t0 is not None and not done:
            tau = (n_cmd - 1) * cn.CONTROL_DT          # plan time of the command being applied
            real = data.time - t0
            if tau - last_log >= LOG_PERIOD:
                last_log = tau
                log.record(tau, data.qpos[arm_qpos].copy(), data.xpos[hand].copy(), data.actuator_force[:7].copy())
            if tau - last_report >= REPORT_PERIOD:
                last_report = tau
                q_ref, hand_ref = log.reference(tau)
                print(f"t = {tau:4.1f} s   hand vs plan {1000 * np.linalg.norm(data.xpos[hand] - hand_ref):7.2f} mm   "
                      f"worst joint {np.degrees(np.abs(q_ref - data.qpos[arm_qpos])).max():.3f} deg")
            if real >= log.T + HOLD:
                done = True
                s = log.summary()
                print(f"\n=== TORQUE mode, tracking of the planned trajectory ===\n"
                      f"  during the motion (0 to {log.T:.2f} s): worst joint error {s['max_joint_err_deg']:.3f} deg "
                      f"(joint {s['worst_joint']}), RMS {s['rms_joint_err_deg']:.3f} deg, worst hand error "
                      f"{s['max_hand_err_mm']:.2f} mm\n"
                      f"  {HOLD:.0f} s after the end: joint error {s['final_joint_err_deg']:.4f} deg, hand error "
                      f"{s['final_hand_err_mm']:.3f} mm\n"
                      f"  plan playback: {min(tau, log.T + HOLD):.2f} s of plan in {real:.2f} s "
                      f"({min(tau, log.T + HOLD) / real:.3f} x real time, node timer at about "
                      f"{n_cmd / real:.0f} Hz); real-time factor of the simulation "
                      f"{real / (time.time() - wall_t0):.3f}")
                if n_cmd / real > 1.2 / cn.CONTROL_DT:
                    print(f"  WARNING: {n_cmd / real:.0f} commands/s, more than one controller sends "
                          f"({1 / cn.CONTROL_DT:.0f}/s). Is an old controller_node still running in another "
                          f"container? ROS 2 nodes in different containers see each other.")
                os.makedirs(RESULTS, exist_ok=True)
                path = os.path.join(RESULTS, f"sim_{args.tag}.npz")
                np.savez(path, **log.arrays(), T=log.T)
                print(f"Log saved to {os.path.relpath(path, os.path.join(HERE, '..', '..'))}")

    def realtime_loop(still_running, sync=lambda: None):
        wall0, sim0, last_sync = time.time(), data.time, 0.0
        while still_running():
            step_once()
            now = time.time()
            if now - last_sync >= 1 / 60:
                last_sync = now
                sync()
            ahead = (data.time - sim0) - (now - wall0)
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -0.05:
                wall0, sim0 = time.time(), data.time      # don't race to catch up

    if args.headless:
        realtime_loop(lambda: not done)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = (0.4, 0.1, 0.4)
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            realtime_loop(viewer.is_running, viewer.sync)


if __name__ == "__main__":
    main()
