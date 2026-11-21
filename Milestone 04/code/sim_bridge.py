"""
MuJoCo side of Milestone 4: the UR5e colour-sorting cell, now in TORQUE mode.

What changed from Milestone 3:
  - the six arm actuators are plain torque motors (tau = ctrl), limited to the UR5e's
    150 / 28 N m. The servo springs of the XML are gone, so nothing holds the arm
    up except the torques our controller sends;
  - MuJoCo's built-in gravity compensation is OFF: gravity really acts on the arm,
    and our controller must cancel it with g(q) from the Lagrangian model;
  - brakes: like a real UR5e, when no controller is sending torques (start-up, or no
    command for BRAKE_AFTER s) the joints are held by the brakes;
  - a see-through green ball shows where the plan wants the flange to be; the live
    panel shows the controller, the tracking error and the motor torques.

Commands come over TCP from mujoco_bridge_node.py (ROS 2 in Docker):
    {"tau": [6 torques N m], "gripper": 0..255, "q_d": [6 planned angles], "label": "...", "ctrl": "..."}
q_d / label / ctrl are only for the display. State goes back at STATE_HZ.
"""
import argparse
import json
import os
import socket
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from dynamics import TORQUE_LIMIT
from ee_sensors import add_ee_sensors, read_ee
from kinematics import HOME_Q, ROBOT_BASE_POS, forward_kinematics
from sorting_task import PART_CENTER_Z, STATION_XY

HERE = os.path.dirname(os.path.abspath(__file__))
CAD = os.path.join(HERE, "..", "..", "Milestone 01", "cad_models")
PORT = 9995                # Milestone 3 uses 9996, Milestone 2 9999, the challenges 9998 / 9997
STATE_HZ = 250.0           # joint state stream to the controller
PANEL_PERIOD = 0.25
BRAKE_AFTER = 0.1          # s without a torque command -> brakes on
# The brake as a stiff spring-damper on each joint, scaled by the joint's inertia so it is
# equally stiff (and numerically stable) on the heavy shoulder and the light wrist.
BRAKE_INERTIA = np.array([2.0, 2.5, 1.1, 0.16, 0.13, 0.10])
BRAKE_WN = 60.0
BRAKE_KP, BRAKE_KD = BRAKE_INERTIA * BRAKE_WN**2, BRAKE_INERTIA * 2 * BRAKE_WN
MAX_LAG = 0.05             # s; if the simulation falls further behind the wall clock, it does not race to catch up

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
ARM_ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
GRIPPER_ACTUATOR = "gripper_fingers_actuator"

BELT_SPEED = 0.08  # m/s
BELT_X = (-1.05, 0.15)
BELT_Y = (0.45, 0.75)
PART_GAP = 0.07
COLOR_NAMES = ("red", "green", "blue")


def build_model(environment=True, payload_mass=0.0, mass_scale=None):
    """
    The cell with the UR5e in torque mode and gravity acting on it.

    payload_mass: extra mass [kg] bolted inside the gripper (robustness tests; the
                  controller's model does not know about it).
    mass_scale:   {body name: factor} to make the real arm differ from our model.
    """
    arm = mujoco.MjSpec.from_file(os.path.join(CAD, "ur5e", "scene.xml"))
    gripper = mujoco.MjSpec.from_file(os.path.join(CAD, "robotiq_2f85", "2f85.xml"))

    arm.option.cone = gripper.option.cone
    arm.option.impratio = gripper.option.impratio
    arm.body("base").pos = ROBOT_BASE_POS
    add_ee_sensors(arm)
    for name, act in zip(ARM_ACTUATORS, arm.actuators):
        assert act.name == name
    for i, act in enumerate(arm.actuators):
        # Torque motor: force = gain * ctrl, no position / velocity feedback inside.
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.gainprm[:3] = [1.0, 0.0, 0.0]
        act.biasprm[:3] = [0.0, 0.0, 0.0]
        act.ctrlrange = [-TORQUE_LIMIT[i], TORQUE_LIMIT[i]]
        act.forcerange = [-TORQUE_LIMIT[i], TORQUE_LIMIT[i]]
    for name, factor in (mass_scale or {}).items():
        body = arm.body(name)
        body.mass *= factor
        body.inertia = [v * factor for v in body.inertia]
    if payload_mass > 0:
        # A lump of metal between the pads, rigidly fixed to the gripper base.
        holder = gripper.body("base")
        holder.add_body(name="payload", pos=[0, 0, 0.156 - 0.0038 - 0.007]).add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], mass=payload_mass,
            rgba=[0.9, 0.6, 0.1, 1], contype=0, conaffinity=0)
    arm.attach(gripper, prefix="gripper_", site=arm.site("attachment_site"))
    if environment:
        env = mujoco.MjSpec.from_file(os.path.join(HERE, "environment.xml"))
        arm.attach(env, prefix="env_", frame=arm.worldbody.add_frame())
    # Where the plan wants the flange: a ghost ball moved every step (mocap, never collides).
    target = arm.worldbody.add_body(name="flange_target", mocap=True)
    target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], rgba=[0.1, 0.9, 0.3, 0.35],
                    contype=0, conaffinity=0, group=1)
    return arm.compile()


class Arm:
    """Addresses of the arm's joints and actuators in the compiled model."""

    def __init__(self, model):
        self.qpos = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.dof = [model.joint(j).dofadr[0] for j in ARM_JOINTS]
        self.act = [model.actuator(a).id for a in ARM_ACTUATORS]
        self.grip = model.actuator(GRIPPER_ACTUATOR).id
        self.target = model.body("flange_target").mocapid[0]

    def state(self, data):
        return data.qpos[self.qpos].copy(), data.qvel[self.dof].copy(), data.qacc[self.dof].copy()

    def show_target(self, data, q_d):
        data.mocap_pos[self.target] = forward_kinematics(q_d)[:3, 3] + ROBOT_BASE_POS


class Brakes:
    """Holds the joints where they were when the brakes came on (no torque from the motors)."""

    def __init__(self):
        self.q_lock = None

    def engage(self, q):
        if self.q_lock is None:
            self.q_lock = np.array(q)

    def release(self):
        self.q_lock = None

    @property
    def on(self):
        return self.q_lock is not None

    def force(self, q, dq):
        return BRAKE_KP * (self.q_lock - q) - BRAKE_KD * dq


class Conveyor:
    """Moves the parts on the belt to the pick station, where they queue up (as in Milestone 2/3)."""

    def __init__(self, model):
        self._parts = []
        for b in range(model.nbody):
            if model.body(b).name.startswith("env_part_"):
                joint = model.body_jntadr[b]
                rgb = model.mat_rgba[model.geom_matid[model.body_geomadr[b]], :3]
                self._parts.append((model.jnt_qposadr[joint], model.jnt_dofadr[joint], COLOR_NAMES[np.argmax(rgb)]))

    def _on_belt(self, data):
        on_belt = []
        for qadr, dadr, color in self._parts:
            x, y, z = data.qpos[qadr:qadr + 3]
            if BELT_X[0] < x < BELT_X[1] and BELT_Y[0] < y < BELT_Y[1] and abs(z - PART_CENTER_Z) < 0.01:
                on_belt.append((x, dadr, color))
        return sorted(on_belt, reverse=True)

    def step(self, data):
        limit = STATION_XY[0]
        for x, dadr, _ in self._on_belt(data):
            data.qvel[dadr] = BELT_SPEED if x < limit - 0.002 else 0.0
            limit = min(limit, x) - PART_GAP

    def station_color(self, data):
        for x, dadr, color in self._on_belt(data):
            if abs(x - STATION_XY[0]) < 0.005 and abs(data.qvel[dadr]) < 0.01:
                return color
        return ""


class Bridge:
    """TCP server that the ROS 2 bridge node (inside Docker) connects to. One JSON object per line."""

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
                        # Torques and the display reference arrive as separate messages:
                        # merge them so none is lost between two physics steps.
                        self._command = {**(self._command or {}), **msg}
                        if "tau" in msg:
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


def panel_text(ctrl, label, q, q_d, tau, ee_p, brakes_on, t):
    e = np.degrees(np.abs(q_d - q)).max() if q_d is not None else 0.0
    flange_err = 1000 * np.linalg.norm(forward_kinematics(q_d)[:3, 3] - ee_p) if q_d is not None else 0.0
    load = np.abs(tau) / TORQUE_LIMIT * 100
    left = "\n".join(["TORQUE CONTROL", "controller", "step", "joint error (max)", "flange error",
                      "torque  base/shoulder/elbow", "torque  wrist 1/2/3", "brakes", "sim time"])
    right = "\n".join([
        "",
        ctrl or "-",
        (label or "-")[:34],
        f"{e:.3f} deg",
        f"{flange_err:.2f} mm",
        f"{tau[0]:+6.1f} {tau[1]:+6.1f} {tau[2]:+6.1f} N m",
        f"{tau[3]:+6.2f} {tau[4]:+6.2f} {tau[5]:+6.2f} N m",
        "ON (no controller)" if brakes_on else f"off   (peak load {load.max():.0f}% of limit)",
        f"{t:.1f} s",
    ])
    return left, right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="no viewer window (automated tests)")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    parser.add_argument("--payload", type=float, default=0.0, help="hidden extra mass in the gripper [kg]")
    args = parser.parse_args()

    model = build_model(payload_mass=args.payload)
    data = mujoco.MjData(model)
    arm = Arm(model)
    conveyor = Conveyor(model)
    brakes = Brakes()

    data.qpos[arm.qpos] = HOME_Q
    mujoco.mj_forward(model, data)
    arm.show_target(data, HOME_Q)

    bridge = Bridge(PORT)
    display = {"q_d": None, "label": "", "ctrl": ""}
    tau = np.zeros(6)
    last = {"state": 0.0, "panel": 0.0, "sync": 0.0}
    state_every = max(1, int(round(1.0 / (STATE_HZ * model.opt.timestep))))
    step = 0

    def run(viewer):
        nonlocal tau, step
        wall0, sim0 = time.time(), data.time
        while (viewer is None or viewer.is_running()) and (args.duration is None or data.time - sim0 < args.duration):
            command = bridge.pop_command()
            if command is not None:
                if "tau" in command and len(command["tau"]) == 6:
                    tau = np.clip(np.asarray(command["tau"], dtype=float), -TORQUE_LIMIT, TORQUE_LIMIT)
                    brakes.release()
                if "gripper" in command:
                    lo, hi = model.actuator_ctrlrange[arm.grip]
                    data.ctrl[arm.grip] = np.clip(float(command["gripper"]), lo, hi)
                if command.get("q_d") is not None:
                    display["q_d"] = np.asarray(command["q_d"], dtype=float)
                    arm.show_target(data, display["q_d"])
                display["label"] = command.get("label", display["label"])
                display["ctrl"] = command.get("ctrl", display["ctrl"])

            q, dq, _ = arm.state(data)
            stale = bridge.last_command_wall is None or time.time() - bridge.last_command_wall > BRAKE_AFTER
            if stale:
                brakes.engage(q)
            data.qfrc_applied[arm.dof] = brakes.force(q, dq) if brakes.on else 0.0
            data.ctrl[arm.act] = 0.0 if brakes.on else tau

            conveyor.step(data)
            mujoco.mj_step(model, data)
            step += 1

            now = time.time()
            if step % state_every == 0:
                q, dq, ddq = arm.state(data)
                bridge.send_state({
                    "t": data.time, "q": q.tolist(), "dq": dq.tolist(), "ddq": ddq.tolist(),
                    "tau": data.actuator_force[arm.act].tolist(), "brakes": brakes.on,
                    "gripper": float(data.ctrl[arm.grip]),
                    "ee_p": (data.site_xpos[model.site("attachment_site").id] - ROBOT_BASE_POS).tolist(),
                    "station": conveyor.station_color(data),
                })
            if viewer is not None and now - last["panel"] >= PANEL_PERIOD:
                last["panel"] = now
                mujoco.mj_forward(model, data)
                q, dq, _ = arm.state(data)
                ee = read_ee(model, data)
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                  *panel_text(display["ctrl"], display["label"], q, display["q_d"],
                                              data.actuator_force[arm.act], ee["p"], brakes.on, data.time)))
            if viewer is not None and now - last["sync"] >= 1 / 60:
                last["sync"] = now
                viewer.sync()

            ahead = (data.time - sim0) - (time.time() - wall0)
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -MAX_LAG:
                # Running slower than real time for a moment: carry on from here instead of
                # racing (a remote controller could not keep up with a burst of fast steps).
                wall0, sim0 = time.time(), data.time

    if args.headless:
        run(None)
    else:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.lookat[:] = (0.1, 0.3, 0.6)
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = -150
            viewer.cam.elevation = -38
            run(viewer)


if __name__ == "__main__":
    main()
