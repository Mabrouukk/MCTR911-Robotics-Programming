"""
ROS 2 side of the Challenge 2 simulator (runs inside Docker).

Stands in for the course mujoco_ros2 bridge, with the same topics, so the
trajectory_node runs unchanged against it:

  /joint_commands (Float64MultiArray, exactly 9 values) -> forwarded to panda_sim.py
  panda_sim.py joint positions and velocities           -> /joint_state (JointState, 9 joints)

The bridge itself does not care about the control mode: panda_sim.py --mode
decides whether the 7 arm values are angles or velocities.
"""
import json
import os
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

SIM_HOST = os.environ.get("SIM_HOST", "host.docker.internal")
SIM_PORT = 9997
CMD_SIZE = 9
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]


class PandaBridgeNode(Node):
    def __init__(self):
        super().__init__("panda_bridge")
        self._sock = None
        self._sock_lock = threading.Lock()
        self._state_pub = self.create_publisher(JointState, "joint_state", 10)
        self.create_subscription(Float64MultiArray, "joint_commands", self._on_command, 10)
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _connect(self):
        while rclpy.ok():
            try:
                sock = socket.create_connection((SIM_HOST, SIM_PORT))
                self.get_logger().info(f"Connected to MuJoCo simulator at {SIM_HOST}:{SIM_PORT}")
                return sock
            except OSError:
                self.get_logger().info("Simulator not reachable, retrying in 1 s (is panda_sim.py running on the Mac?)")
                time.sleep(1.0)
        return None

    def _on_command(self, msg):
        if len(msg.data) != CMD_SIZE:
            self.get_logger().warn(
                f"Expected {CMD_SIZE} values on /joint_commands, got {len(msg.data)}; ignored",
                throttle_duration_sec=2.0)
            return
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return
        try:
            sock.sendall((json.dumps({"cmd": list(msg.data)}) + "\n").encode())
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
                    state = JointState()
                    state.header.stamp = self.get_clock().now().to_msg()
                    state.name = JOINT_NAMES
                    msg = json.loads(line)
                    state.position = msg["q"]
                    state.velocity = msg.get("dq", [])
                    self._state_pub.publish(state)
            except (OSError, json.JSONDecodeError):
                pass
            with self._sock_lock:
                self._sock = None
            self.get_logger().warn("Lost connection to the simulator, reconnecting...")


def main():
    rclpy.init()
    node = PandaBridgeNode()
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
