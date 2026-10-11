"""
MuJoCo simulator for Challenge 1, running natively on the Mac.

It plays the part of the course mujoco_ros2 bridge in POSITION mode:
  - the arm actuators in panda.xml are position servos, so the first 7
    command values are target joint angles and the 8th is the gripper (0-255),
  - commands come in, and joint states go out, over a TCP socket that
    panda_bridge_node.py (ROS 2, inside Docker) connects to.

A transparent RGB frame marks TARGET_POSE, so you can see the hand land on it,
and every 2 s the terminal prints how far the simulated hand is from it.

Run with mjpython (needed for the viewer on macOS); scripts/run_ch1.sh does this.
"""
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
sys.path.insert(0, os.path.join(HERE, "..", "panda_kinematics"))
from panda_kinematics.kinematics import TARGET_POSE, pose_error  # noqa: E402

SCENE = os.path.join(HERE, "..", "model", "scene.xml")
PORT = 9998                # Milestone 2 uses 9999
CMD_SIZE = 9               # 7 arm joints + gripper + 1 spare, like the course bridge
STATE_PERIOD = 1 / 30
REPORT_PERIOD = 2.0
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]


def build_model():
    spec = mujoco.MjSpec.from_file(SCENE)
    # The arm actuators are plain spring-damper servos (kp, kd, no integral
    # term), so without help gravity bends the arm ~0.5 deg at joint 2, about
    # 8 mm at the hand. A real Panda's controller compensates gravity; gravcomp
    # does the same here, so what you see is what the IK commanded.
    for body in spec.bodies:
        if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
            body.gravcomp = 1
    # Target marker: a mocap body (moved by hand, not by physics) with three
    # axes, red = x, green = y, blue = z. No collisions, so it never gets in the way.
    target = spec.worldbody.add_body(name="target", mocap=True)
    for axis, rgba in zip(np.eye(3), ([1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5])):
        target.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, *(0.1 * axis)],
                        size=[0.005, 0, 0], rgba=rgba, contype=0, conaffinity=0)
    return spec.compile()


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
    model = build_model()
    data = mujoco.MjData(model)

    arm_qpos = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    finger_qpos = [model.joint(j).qposadr[0] for j in FINGER_JOINTS]
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    hand = model.body("hand").id

    # Start at the 'home' keyframe from panda.xml, holding still.
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.mocap_pos[0] = TARGET_POSE[:3, 3]
    mujoco.mju_mat2Quat(data.mocap_quat[0], TARGET_POSE[:3, :3].flatten())
    mujoco.mj_forward(model, data)

    bridge = Bridge(PORT)
    last_state = last_report = 0.0
    commanded = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            command = bridge.pop_command()
            if command is not None:
                cmd = command.get("cmd", [])
                # Same rule as the course bridge: anything but exactly 9 values is ignored.
                if len(cmd) == CMD_SIZE:
                    data.ctrl[:8] = np.clip(cmd[:8], ctrl_lo[:8], ctrl_hi[:8])
                    commanded = True

            mujoco.mj_step(model, data)
            viewer.sync()

            if step_start - last_state >= STATE_PERIOD:
                last_state = step_start
                bridge.send_state({"q": data.qpos[arm_qpos + finger_qpos].tolist()})

            if commanded and step_start - last_report >= REPORT_PERIOD:
                last_report = step_start
                T = np.eye(4)
                T[:3, :3] = data.xmat[hand].reshape(3, 3)
                T[:3, 3] = data.xpos[hand]
                e = pose_error(TARGET_POSE, T)
                print(f"hand vs TARGET_POSE: {np.linalg.norm(e[:3]) * 1000:7.2f} mm, "
                      f"{np.degrees(np.linalg.norm(e[3:])):6.2f} deg")

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
