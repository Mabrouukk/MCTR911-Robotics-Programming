import json
import os
import socket
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from kinematics import HOME_Q, ROBOT_BASE_POS

HERE = os.path.dirname(os.path.abspath(__file__))
CAD = os.path.join(HERE, "..", "..", "Milestone 01", "cad_models")
PORT = 9999
STATE_PERIOD = 1 / 30

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
ARM_ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
GRIPPER_ACTUATOR = "gripper_fingers_actuator"


def build_model():
    arm = mujoco.MjSpec.from_file(os.path.join(CAD, "ur5e", "scene.xml"))
    gripper = mujoco.MjSpec.from_file(os.path.join(CAD, "robotiq_2f85", "2f85.xml"))
    environment = mujoco.MjSpec.from_file(os.path.join(HERE, "environment.xml"))

    arm.option.cone = gripper.option.cone
    arm.option.impratio = gripper.option.impratio
    arm.body("base").pos = ROBOT_BASE_POS
    arm.attach(gripper, prefix="gripper_", site=arm.site("attachment_site"))
    arm.attach(environment, prefix="env_", frame=arm.worldbody.add_frame())
    return arm.compile()


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
    model = build_model()
    data = mujoco.MjData(model)

    qpos_adr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    arm_act = [model.actuator(a).id for a in ARM_ACTUATORS]
    grip_act = model.actuator(GRIPPER_ACTUATOR).id
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    ee_site = model.site("attachment_site").id

    data.qpos[qpos_adr] = HOME_Q
    data.ctrl[arm_act] = HOME_Q
    mujoco.mj_forward(model, data)

    bridge = Bridge(PORT)
    last_state_time = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            command = bridge.pop_command()
            if command is not None:
                if "q" in command and len(command["q"]) == 6:
                    q = np.asarray(command["q"], dtype=float)
                    data.ctrl[arm_act] = np.clip(q, ctrl_lo[arm_act], ctrl_hi[arm_act])
                if "gripper" in command:
                    data.ctrl[grip_act] = np.clip(float(command["gripper"]), ctrl_lo[grip_act], ctrl_hi[grip_act])

            mujoco.mj_step(model, data)
            viewer.sync()

            if step_start - last_state_time >= STATE_PERIOD:
                last_state_time = step_start
                bridge.send_state({
                    "q": data.qpos[qpos_adr].tolist(),
                    "gripper": float(data.ctrl[grip_act]),
                    "ee": data.site_xpos[ee_site].tolist(),
                })

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
