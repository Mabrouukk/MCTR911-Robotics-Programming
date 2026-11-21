"""
ROS 2 bridge for Milestone 4 (runs in Docker): links ROS 2 to the torque-mode MuJoCo
simulator on the Mac (sim_bridge.py) over TCP.

In:
    /ur5e/effort_commands     Float64MultiArray: 6 joint torques [N m], optional 7th = gripper 0-255
    /ur5e/joint_reference     JointState: the planned q_d (position), q_dot_d (velocity), q_ddot_d (effort);
                              header.frame_id carries "<controller>|<step label>". Display only.
Out (stamped with the simulator time):
    /ur5e/joint_states        JointState: position [rad], velocity [rad/s], effort = applied torque [N m]
    /ur5e/joint_accelerations Float64MultiArray: [sim time, 6 joint accelerations]
    /ur5e/ee_position         PointStamped, base frame: sensed flange position
    /ur5e/brakes              Bool: True while the brakes hold the arm
    /cell/station_color       String: "red", "green", "blue" or ""
"""
import json
import os
import socket
import threading
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String

SIM_HOST = os.environ.get("SIM_HOST", "host.docker.internal")
SIM_PORT = 9995
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
# Control traffic: only the newest message matters, an old torque is worse than none.
FAST = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class MujocoBridgeNode(Node):
    def __init__(self):
        super().__init__("mujoco_bridge")
        self._sock = None
        self._sock_lock = threading.Lock()
        self._state_pub = self.create_publisher(JointState, "/ur5e/joint_states", FAST)
        self._acc_pub = self.create_publisher(Float64MultiArray, "/ur5e/joint_accelerations", 10)
        self._ee_pub = self.create_publisher(PointStamped, "/ur5e/ee_position", 10)
        self._brake_pub = self.create_publisher(Bool, "/ur5e/brakes", 10)
        self._color_pub = self.create_publisher(String, "/cell/station_color", 10)
        self.create_subscription(Float64MultiArray, "/ur5e/effort_commands", self._on_effort, FAST)
        self.create_subscription(JointState, "/ur5e/joint_reference", self._on_reference, 10)
        self._n = 0
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _send(self, command):
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return
        try:
            sock.sendall((json.dumps(command) + "\n").encode())
        except OSError:
            pass

    def _on_effort(self, msg):
        values = list(msg.data)
        if len(values) not in (6, 7):
            self.get_logger().warn(f"Expected 6 or 7 values on /ur5e/effort_commands, got {len(values)}",
                                   throttle_duration_sec=2.0)
            return
        command = {"tau": values[:6]}
        if len(values) == 7:
            command["gripper"] = values[6]
        self._send(command)

    def _on_reference(self, msg):
        ctrl, _, label = msg.header.frame_id.partition("|")
        self._send({"q_d": list(msg.position), "ctrl": ctrl, "label": label})

    def _connect(self):
        while rclpy.ok():
            try:
                sock = socket.create_connection((SIM_HOST, SIM_PORT))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.get_logger().info(f"Connected to MuJoCo simulator at {SIM_HOST}:{SIM_PORT}")
                return sock
            except OSError:
                self.get_logger().info("Simulator not reachable, retrying in 1 s (is sim_bridge.py running on the Mac?)")
                time.sleep(1.0)
        return None

    def _read_loop(self):
        while rclpy.ok():
            sock = self._connect()
            if sock is None:
                return
            with self._sock_lock:
                self._sock = sock
            try:
                for line in sock.makefile("r"):
                    self._publish_state(json.loads(line))
            except (OSError, json.JSONDecodeError):
                pass
            with self._sock_lock:
                self._sock = None
            self.get_logger().warn("Lost connection to the simulator, reconnecting...")

    def _publish_state(self, state):
        stamp = Time(seconds=state["t"]).to_msg()
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = JOINT_NAMES
        joints.position = state["q"]
        joints.velocity = state["dq"]
        joints.effort = state["tau"]
        self._state_pub.publish(joints)
        self._n += 1
        if self._n % 5:                      # the slow topics at 50 Hz
            return
        self._acc_pub.publish(Float64MultiArray(data=[state["t"]] + state["ddq"]))
        ee = PointStamped()
        ee.header.stamp, ee.header.frame_id = stamp, "ur5e_base"
        ee.point.x, ee.point.y, ee.point.z = state["ee_p"]
        self._ee_pub.publish(ee)
        self._brake_pub.publish(Bool(data=bool(state["brakes"])))
        self._color_pub.publish(String(data=state.get("station", "")))


def main():
    rclpy.init()
    node = MujocoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
