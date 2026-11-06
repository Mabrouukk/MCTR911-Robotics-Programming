"""
MuJoCo simulator for Challenge 2, running natively on the Mac.

It plays the part of the course mujoco_ros2 bridge, in either control mode:

  --mode position   the arm actuators are position servos (as in panda.xml),
                    so the 7 arm values on /joint_commands are joint angles.
  --mode velocity   the arm actuators are swapped for velocity servos, so the
                    7 arm values are joint velocities. Nothing turns them back
                    into a position: the physics integrates them, exactly like
                    "relying on the bridge's internal integration".

The --mode flag is our stand-in for {"control_mode": ...} in the course
launch file. The node still has to publish the matching thing (q or dq).

The arm starts at Q_START from trajectory_node.py, so both modes begin from
the planned start. Yellow dots show the planned hand path and an RGB frame
marks the hand pose at Q_END. When the trajectory is done the simulator
prints how closely the arm tracked it and saves the log to results/.

Commands come in and joint states go out over a TCP socket that
panda_bridge_node.py (ROS 2, inside Docker) connects to.

Run with mjpython for the viewer on macOS; scripts/run_ch2.sh does this.
--headless runs the same loop without a window (used for automated tests).
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
from ros_free_import import load_trajectory_node  # noqa: E402

tn = load_trajectory_node()

SCENE = os.path.join(HERE, "..", "..", "Challenge 01", "model", "scene.xml")
RESULTS = os.path.join(HERE, "..", "results")
PORT = 9997                # Challenge 1 uses 9998, Milestone 2 uses 9999
CMD_SIZE = 9               # 7 arm joints + gripper + 1 spare, like the course bridge
STATE_PERIOD = 1 / 30
LOG_PERIOD = 0.01          # tracking log, in simulated seconds
REPORT_PERIOD = 1.0
SETTLE_TIME = 2.0          # how long after the plan ends we measure the final error
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]

# Velocity servo gain per joint (N m s/rad): torque = kv * (qd_cmd - qd).
# Same values as the damping (kd) of the position servos in panda.xml.
VELOCITY_GAIN = np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0])


def plan():
    """The planned trajectory, exactly as the node computes it."""
    return tn.plan_trajectory(tn.Q_START, tn.Q_END, tn.TRAJ_DURATION, tn.TRAJ_DT)


def build_model(mode, gravcomp=True, show_plan=True):
    spec = mujoco.MjSpec.from_file(SCENE)
    if gravcomp:
        # Like the real Panda controller, cancel gravity, so the error we see
        # comes from the control mode, not from the arm sagging.
        for body in spec.bodies:
            if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
                body.gravcomp = 1

    if mode == "velocity":
        # MuJoCo's <velocity> actuator: force = kv * ctrl - kv * qvel.
        for i, act in enumerate(spec.actuators[:7]):
            kv = VELOCITY_GAIN[i]
            act.gainprm[:3] = [kv, 0.0, 0.0]
            act.biasprm[:3] = [0.0, 0.0, -kv]
            act.ctrlrange = [-tn.QD_MAX[i], tn.QD_MAX[i]]

    if show_plan:
        _, q_plan, _, _ = plan()
        # Planned hand path as small dots (drawn only, never collide).
        for q in q_plan[::4]:
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0, 0],
                pos=tn.forward_kinematics(q)[:3, 3], rgba=[1, 0.85, 0, 0.8],
                contype=0, conaffinity=0)
        # Hand pose at Q_END: red = x, green = y, blue = z.
        T_end = tn.forward_kinematics(tn.Q_END)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T_end[:3, :3].flatten())
        frame = spec.worldbody.add_body(name="goal", pos=T_end[:3, 3], quat=quat)
        for axis, rgba in zip(np.eye(3), ([1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5])):
            frame.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, *(0.1 * axis)],
                           size=[0.005, 0, 0], rgba=rgba, contype=0, conaffinity=0)
    return spec.compile()


def reset_to_start(model, data, mode):
    """Arm at Q_START, still, gripper open, actuators holding that state."""
    mujoco.mj_resetData(model, data)
    for i, name in enumerate(ARM_JOINTS):
        data.qpos[model.joint(name).qposadr[0]] = tn.Q_START[i]
    for name in FINGER_JOINTS:
        data.qpos[model.joint(name).qposadr[0]] = 0.04
    data.ctrl[:7] = tn.Q_START if mode == "position" else 0.0
    data.ctrl[7] = tn.GRIPPER_CTRL
    mujoco.mj_forward(model, data)


def apply_command(model, data, cmd):
    """Same rule as the course bridge: exactly 9 values, else ignored."""
    if len(cmd) != CMD_SIZE:
        return False
    lo, hi = model.actuator_ctrlrange[:8, 0], model.actuator_ctrlrange[:8, 1]
    data.ctrl[:8] = np.clip(cmd[:8], lo, hi)
    return True


class TrackingLog:
    """Planned vs actual joints and hand position, against time since the first command."""

    def __init__(self):
        self.t_plan, self.q_plan, _, _ = plan()
        self.T = self.t_plan[-1]
        self.rows = []

    def reference(self, tau):
        q = np.array([np.interp(tau, self.t_plan, self.q_plan[:, j]) for j in range(7)])
        return q, tn.forward_kinematics(q)[:3, 3]

    def record(self, tau, q, hand):
        q_ref, hand_ref = self.reference(tau)
        self.rows.append(np.concatenate([[tau], q, q_ref, hand, hand_ref]))

    def arrays(self):
        a = np.array(self.rows)
        return dict(t=a[:, 0], q=a[:, 1:8], q_ref=a[:, 8:15], hand=a[:, 15:18], hand_ref=a[:, 18:21])

    def summary(self):
        a = self.arrays()
        joint_err = np.degrees(np.abs(a["q"] - a["q_ref"]))
        hand_err = 1000 * np.linalg.norm(a["hand"] - a["hand_ref"], axis=1)
        moving = a["t"] <= self.T
        final = a["t"] >= a["t"][-1] - 0.05
        worst = int(np.argmax(joint_err[moving].max(axis=0)))
        return {
            "max_joint_err_deg": float(joint_err[moving].max()),
            "max_joint_err_joint": worst + 1,
            "max_hand_err_mm": float(hand_err[moving].max()),
            "at_T_hand_err_mm": float(np.interp(self.T, a["t"], hand_err)),
            "final_joint_err_deg": float(joint_err[final].max()),
            "final_hand_err_mm": float(hand_err[final].mean()),
        }


def format_summary(mode, s, T):
    return (f"\n=== {mode.upper()} mode, tracking of the planned trajectory ===\n"
            f"  during the motion (0 to {T:.2f} s): worst joint error {s['max_joint_err_deg']:.2f} deg "
            f"(joint {s['max_joint_err_joint']}), worst hand error {s['max_hand_err_mm']:.1f} mm\n"
            f"  at t = T:      hand error {s['at_T_hand_err_mm']:.1f} mm\n"
            f"  {SETTLE_TIME:.0f} s after T: worst joint error {s['final_joint_err_deg']:.3f} deg, "
            f"hand error {s['final_hand_err_mm']:.2f} mm (drift that stays)\n")


class Bridge:
    """TCP server that panda_bridge_node.py (inside Docker) connects to. One JSON object per line."""

    def __init__(self, port):
        self._lock = threading.Lock()
        self._command = None
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["position", "velocity"], default="position")
    parser.add_argument("--headless", action="store_true", help="no viewer; exit after the summary")
    parser.add_argument("--no-gravcomp", action="store_true", help="let gravity act on the arm")
    args = parser.parse_args()

    model = build_model(args.mode, gravcomp=not args.no_gravcomp)
    data = mujoco.MjData(model)
    arm_qpos = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    finger_qpos = [model.joint(j).qposadr[0] for j in FINGER_JOINTS]
    arm_dof = [model.joint(j).dofadr[0] for j in ARM_JOINTS]
    finger_dof = [model.joint(j).dofadr[0] for j in FINGER_JOINTS]
    hand = model.body("hand").id
    reset_to_start(model, data, args.mode)

    print(f"Control mode: {args.mode.upper()}  "
          f"(the node must publish {'q_traj (joint angles)' if args.mode == 'position' else 'dq_traj (joint velocities)'})")
    print(f"Gravity compensation: {'off' if args.no_gravcomp else 'on'}")

    bridge = Bridge(PORT)
    log = TrackingLog()
    t0 = None                    # sim time of the first command
    wall_t0 = None               # wall time of the first command
    last_state = last_log = last_report = -1e9
    done = False

    def step_once():
        nonlocal t0, wall_t0, last_state, last_log, last_report, done
        command = bridge.pop_command()
        if command is not None and apply_command(model, data, command.get("cmd", [])) and t0 is None:
            t0 = data.time
            wall_t0 = time.time()
            print(f"First command received, trajectory clock started ({log.T:.2f} s planned)")

        mujoco.mj_step(model, data)

        if data.time - last_state >= STATE_PERIOD:
            last_state = data.time
            bridge.send_state({"q": data.qpos[arm_qpos + finger_qpos].tolist(),
                               "dq": data.qvel[arm_dof + finger_dof].tolist()})

        if t0 is not None and not done:
            tau = data.time - t0
            if tau - last_log >= LOG_PERIOD:
                last_log = tau
                log.record(tau, data.qpos[arm_qpos].copy(), data.xpos[hand].copy())
            if tau - last_report >= REPORT_PERIOD:
                last_report = tau
                _, hand_ref = log.reference(tau)
                print(f"t = {tau:4.1f} s   hand vs plan: {1000 * np.linalg.norm(data.xpos[hand] - hand_ref):6.1f} mm")
            if tau >= log.T + SETTLE_TIME:
                done = True
                s = log.summary()
                print(format_summary(args.mode, s, log.T))
                print(f"Real-time factor during the run: {tau / (time.time() - wall_t0):.3f} (1.000 = real time)")
                os.makedirs(RESULTS, exist_ok=True)
                tag = args.mode + ("_nogravcomp" if args.no_gravcomp else "")
                path = os.path.join(RESULTS, f"sim_{tag}.npz")
                np.savez(path, **log.arrays(), T=log.T)
                print(f"Log saved to {os.path.relpath(path, os.path.join(HERE, '..', '..'))}")

    def realtime_loop(still_running, sync=lambda: None):
        # Keep simulated time locked to the wall clock. The node sends one
        # command every 50 ms of WALL time, so if the simulation ran slower
        # than real time the plan would play back too fast in sim time.
        # Sleeping a fixed amount per step lets small oversleeps add up, so
        # we instead sleep until the wall clock catches up with data.time,
        # and redraw the viewer at about 60 Hz.
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

    if args.headless:
        realtime_loop(lambda: not done)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            realtime_loop(viewer.is_running, viewer.sync)


if __name__ == "__main__":
    main()
