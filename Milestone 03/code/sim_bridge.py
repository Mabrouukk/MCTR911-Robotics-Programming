"""
MuJoCo side of Milestone 3: the UR5e colour-sorting cell, running natively on the Mac.

New in Milestone 3 (on top of Milestone 2's cell, conveyor and colour sensor):
  - six MuJoCo sensors on the flange (ee_sensors.py): position, orientation,
    linear/angular velocity and linear/angular acceleration;
  - the state sent to ROS 2 now also carries the joint velocities and
    accelerations and all of those flange readings, from the same instant;
  - a live "kinematics check" panel in the viewer: it compares what the
    sensors measure with what our equations (velocity_kinematics.py) predict
    from the joint state, forward and inverse, while the arm moves;
  - gravity compensation on the arm, like the UR5e's own controller does,
    so the position servos are not pulled down by gravity;
  - the bigger environment (environment.xml): fence, people, chairs, tools...

Commands come in over TCP from mujoco_bridge_node.py (ROS 2 in Docker):
{"q": [6 joint angles], "gripper": 0..255}.
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

from ee_sensors import add_ee_sensors, read_ee
from kinematic_checks import Panel, kinematics_check
from kinematics import HOME_Q, ROBOT_BASE_POS
from sorting_task import PART_CENTER_Z, STATION_XY

HERE = os.path.dirname(os.path.abspath(__file__))
CAD = os.path.join(HERE, "..", "..", "Milestone 01", "cad_models")
PORT = 9996                # Milestone 2 uses 9999, the challenges 9998 / 9997
STATE_PERIOD = 1 / 30
PANEL_PERIOD = 0.5

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
ARM_ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
ARM_BODIES = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
GRIPPER_ACTUATOR = "gripper_fingers_actuator"

BELT_SPEED = 0.08  # m/s
BELT_X = (-1.05, 0.15)
BELT_Y = (0.45, 0.75)
PART_GAP = 0.07  # centre-to-centre spacing of parts queued behind each other
COLOR_NAMES = ("red", "green", "blue")


def build_model(gravity_compensation=True):
    arm = mujoco.MjSpec.from_file(os.path.join(CAD, "ur5e", "scene.xml"))
    gripper = mujoco.MjSpec.from_file(os.path.join(CAD, "robotiq_2f85", "2f85.xml"))
    environment = mujoco.MjSpec.from_file(os.path.join(HERE, "environment.xml"))

    arm.option.cone = gripper.option.cone
    arm.option.impratio = gripper.option.impratio
    arm.body("base").pos = ROBOT_BASE_POS
    add_ee_sensors(arm)
    if gravity_compensation:
        for name in ARM_BODIES:
            arm.body(name).gravcomp = 1
        for body in gripper.bodies:
            body.gravcomp = 1
    arm.attach(gripper, prefix="gripper_", site=arm.site("attachment_site"))
    arm.attach(environment, prefix="env_", frame=arm.worldbody.add_frame())
    return arm.compile()


class Arm:
    """Addresses of the arm's joints and actuators in the compiled model."""

    def __init__(self, model):
        self.qpos = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.dof = [model.joint(j).dofadr[0] for j in ARM_JOINTS]
        self.act = [model.actuator(a).id for a in ARM_ACTUATORS]
        self.grip = model.actuator(GRIPPER_ACTUATOR).id

    def state(self, data):
        """Sensed joint state: angles, rates and accelerations (the 'encoders')."""
        return data.qpos[self.qpos].copy(), data.qvel[self.dof].copy(), data.qacc[self.dof].copy()


class Conveyor:
    """Moves the parts lying on the belt towards the pick station, where they queue up.

    The belt geom is static, so every step the parts resting on it get the belt's speed,
    unless they have reached the end stop or the part in front of them.
    A colour sensor at the station reports the colour of the part waiting there.
    """

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
    """TCP server that the ROS2 bridge node (inside Docker) connects to."""

    def __init__(self, port):
        self._lock = threading.Lock()
        self._command = None
        self._conn = None
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", port))
        self._server.listen(1)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"Waiting for the ROS2 bridge node on port {port}...")

    def _accept_loop(self):
        while True:
            conn, addr = self._server.accept()
            print(f"ROS2 bridge connected from {addr}")
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
            print("ROS2 bridge disconnected")
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
    parser.add_argument("--headless", action="store_true", help="no viewer window (automated tests)")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    parser.add_argument("--no-panel", action="store_true", help="hide the live kinematics panel")
    args = parser.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    arm = Arm(model)
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    ee_site = model.site("attachment_site").id
    conveyor = Conveyor(model)

    data.qpos[arm.qpos] = HOME_Q
    data.ctrl[arm.act] = HOME_Q
    mujoco.mj_forward(model, data)

    bridge = Bridge(PORT)
    last = {"state": 0.0, "panel": 0.0, "sync": 0.0}
    live_panel = Panel()

    def run(viewer):
        wall0, sim0 = time.time(), data.time
        while (viewer is None or viewer.is_running()) and (args.duration is None or data.time - sim0 < args.duration):
            command = bridge.pop_command()
            if command is not None:
                if "q" in command and len(command["q"]) == 6:
                    q = np.asarray(command["q"], dtype=float)
                    data.ctrl[arm.act] = np.clip(q, ctrl_lo[arm.act], ctrl_hi[arm.act])
                if "gripper" in command:
                    data.ctrl[arm.grip] = np.clip(float(command["gripper"]), ctrl_lo[arm.grip], ctrl_hi[arm.grip])

            conveyor.step(data)
            mujoco.mj_step(model, data)

            now = time.time()
            send = now - last["state"] >= STATE_PERIOD
            panel = viewer is not None and not args.no_panel and now - last["panel"] >= PANEL_PERIOD
            if send or panel:
                # mj_step leaves qacc and the sensors at the state BEFORE the
                # step; mj_forward recomputes them for the current state, so
                # joints and flange readings are from the same instant.
                mujoco.mj_forward(model, data)
                q, dq, ddq = arm.state(data)
                ee = read_ee(model, data)
            if send:
                last["state"] = now
                bridge.send_state({
                    "t": data.time,
                    "q": q.tolist(), "dq": dq.tolist(), "ddq": ddq.tolist(),
                    "gripper": float(data.ctrl[arm.grip]),
                    "ee": data.site_xpos[ee_site].tolist(),          # world frame, as in Milestone 2
                    "ee_p": ee["p"].tolist(), "ee_R": ee["R"].flatten().tolist(),
                    "ee_v": ee["v"].tolist(), "ee_w": ee["w"].tolist(),
                    "ee_a": ee["a"].tolist(), "ee_alpha": ee["alpha"].tolist(),
                    "station": conveyor.station_color(data),
                })
            if panel:
                last["panel"] = now
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                  *live_panel.update(ee, kinematics_check(q, dq, ddq, ee))))
            if viewer is not None and now - last["sync"] >= 1 / 60:
                last["sync"] = now
                viewer.sync()

            # Keep simulated time locked to the wall clock (commands arrive in wall time).
            ahead = (data.time - sim0) - (time.time() - wall0)
            if ahead > 0:
                time.sleep(ahead)

    if args.headless:
        run(None)
    else:
        # Side menus closed so the 3D view gets the whole window (Tab / Shift+Tab open them).
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            # Start looking at the robot, from above inside the fence.
            viewer.cam.lookat[:] = (0.1, 0.3, 0.6)
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = -150
            viewer.cam.elevation = -38
            run(viewer)


if __name__ == "__main__":
    main()
