import json
import socket
import threading
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

SIM_HOST = "host.docker.internal"
SIM_PORT = 9999
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class MujocoBridgeNode(Node):
    """Links ROS2 to the MuJoCo simulator running natively on the Mac (sim_bridge.py).

    /ur5e/joint_commands (Float64MultiArray: 6 joint angles [rad], optional 7th = gripper 0-255)
        -> forwarded to the simulator
    simulator state -> /ur5e/joint_states (JointState) and /ur5e/ee_position (PointStamped, world frame)
    """

    def __init__(self):
        super().__init__("mujoco_bridge")
        self._sock = None
        self._sock_lock = threading.Lock()
        self._state_pub = self.create_publisher(JointState, "/ur5e/joint_states", 10)
        self._ee_pub = self.create_publisher(PointStamped, "/ur5e/ee_position", 10)
        self.create_subscription(Float64MultiArray, "/ur5e/joint_commands", self._on_command, 10)
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _connect(self):
        while rclpy.ok():
            try:
                sock = socket.create_connection((SIM_HOST, SIM_PORT))
                self.get_logger().info(f"Connected to MuJoCo simulator at {SIM_HOST}:{SIM_PORT}")
                return sock
            except OSError:
                self.get_logger().info("Simulator not reachable, retrying in 1 s (is sim_bridge.py running on the Mac?)")
                time.sleep(1.0)
        return None

    def _on_command(self, msg):
        values = list(msg.data)
        if len(values) not in (6, 7):
            self.get_logger().warn(f"Expected 6 or 7 values on /ur5e/joint_commands, got {len(values)}")
            return
        command = {"q": values[:6]}
        if len(values) == 7:
            command["gripper"] = values[6]
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return
        try:
            sock.sendall((json.dumps(command) + "\n").encode())
        except OSError:
            pass

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
        stamp = self.get_clock().now().to_msg()

        joints = JointState()
        joints.header.stamp = stamp
        joints.name = JOINT_NAMES
        joints.position = state["q"]
        self._state_pub.publish(joints)

        ee = PointStamped()
        ee.header.stamp = stamp
        ee.header.frame_id = "world"
        ee.point.x, ee.point.y, ee.point.z = state["ee"]
        self._ee_pub.publish(ee)


def main():
    rclpy.init()
    node = MujocoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
