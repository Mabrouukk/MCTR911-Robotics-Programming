import json
import os
import socket
import threading
import time

import rclpy
from geometry_msgs.msg import AccelStamped, PointStamped, PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

# The bridge between ROS 2 (in Docker) and the MuJoCo simulator on the Mac (sim_bridge.py).
SIM_HOST = os.environ.get("SIM_HOST", "host.docker.internal")
SIM_PORT = 9996
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def matrix_to_quat(R):
    """3x3 rotation matrix (row-major list of 9) to a quaternion (x, y, z, w)."""
    m = [R[0:3], R[3:6], R[6:9]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = 2.0 * (tr + 1.0) ** 0.5
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * (1.0 + m[i][i] - m[j][j] - m[k][k]) ** 0.5
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (m[j][i] + m[i][j]) / s
    q[k] = (m[k][i] + m[i][k]) / s
    return (q[0], q[1], q[2], (m[k][j] - m[j][k]) / s)


class MujocoBridgeNode(Node):
    """Links ROS 2 to the MuJoCo simulator (Milestone 3).

    In:  /ur5e/joint_commands      Float64MultiArray: 6 joint angles [rad], optional 7th = gripper 0-255
    Out (all stamped with the simulator time of one and the same instant):
         /ur5e/joint_states        JointState: position [rad] and velocity [rad/s]
         /ur5e/joint_accelerations Float64MultiArray: [sim time, 6 joint accelerations rad/s^2]
         /ur5e/ee_pose             PoseStamped, base frame "ur5e_base": sensed flange pose
         /ur5e/ee_twist            TwistStamped, base frame: sensed flange linear / angular velocity
         /ur5e/ee_accel            AccelStamped, base frame: sensed flange linear / angular acceleration
         /ur5e/ee_position         PointStamped, world frame (as in Milestone 2)
         /cell/station_color       String: "red", "green", "blue" or "" when empty
    """

    def __init__(self):
        super().__init__("mujoco_bridge")
        self._sock = None
        self._sock_lock = threading.Lock()
        self._state_pub = self.create_publisher(JointState, "/ur5e/joint_states", 10)
        self._acc_pub = self.create_publisher(Float64MultiArray, "/ur5e/joint_accelerations", 10)
        self._pose_pub = self.create_publisher(PoseStamped, "/ur5e/ee_pose", 10)
        self._twist_pub = self.create_publisher(TwistStamped, "/ur5e/ee_twist", 10)
        self._accel_pub = self.create_publisher(AccelStamped, "/ur5e/ee_accel", 10)
        self._ee_pub = self.create_publisher(PointStamped, "/ur5e/ee_position", 10)
        self._color_pub = self.create_publisher(String, "/cell/station_color", 10)
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
        # Stamp everything with the simulator time, so a subscriber can match
        # the joint readings and the flange readings of the same instant.
        sim_t = state["t"]
        stamp = Time(seconds=sim_t).to_msg()

        joints = JointState()
        joints.header.stamp = stamp
        joints.name = JOINT_NAMES
        joints.position = state["q"]
        joints.velocity = state["dq"]
        self._state_pub.publish(joints)
        self._acc_pub.publish(Float64MultiArray(data=[sim_t] + state["ddq"]))

        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = stamp, "ur5e_base"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = state["ee_p"]
        o = pose.pose.orientation
        o.x, o.y, o.z, o.w = matrix_to_quat(state["ee_R"])
        self._pose_pub.publish(pose)

        twist = TwistStamped()
        twist.header.stamp, twist.header.frame_id = stamp, "ur5e_base"
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = state["ee_v"]
        twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z = state["ee_w"]
        self._twist_pub.publish(twist)

        accel = AccelStamped()
        accel.header.stamp, accel.header.frame_id = stamp, "ur5e_base"
        accel.accel.linear.x, accel.accel.linear.y, accel.accel.linear.z = state["ee_a"]
        accel.accel.angular.x, accel.accel.angular.y, accel.accel.angular.z = state["ee_alpha"]
        self._accel_pub.publish(accel)

        ee = PointStamped()
        ee.header.stamp, ee.header.frame_id = stamp, "world"
        ee.point.x, ee.point.y, ee.point.z = state["ee"]
        self._ee_pub.publish(ee)

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
