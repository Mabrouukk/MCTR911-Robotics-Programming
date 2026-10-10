import argparse
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematics import HOME_Q, ROBOT_BASE_POS, forward_kinematics  # noqa: E402
from sorting_task import GRIPPER_CLOSED, GRIPPER_OPEN, SortingController, ik_tool_down  # noqa: E402

RATE_HZ = 50.0


class JointCommandPublisher(Node):
    def __init__(self, args):
        super().__init__("joint_command_publisher")
        self._args = args
        self._pub = self.create_publisher(Float64MultiArray, "/ur5e/joint_commands", 10)
        self._t0 = self.get_clock().now()

        if args.mode == "constant":
            q = np.radians(args.deg) if args.deg else HOME_Q
            self._fixed = (q, args.gripper)
            self.get_logger().info(f"Constant: q = {np.round(np.degrees(q), 1)} deg, gripper = {args.gripper}")
        elif args.mode == "ik":
            q = ik_tool_down(args.xyz, HOME_Q)
            reached = forward_kinematics(q)[:3, 3] + ROBOT_BASE_POS
            self._fixed = (q, args.gripper)
            self.get_logger().info(f"IK target {args.xyz} -> q = {np.round(np.degrees(q), 1)} deg")
            self.get_logger().info(f"FK check: flange lands at {np.round(reached, 4)}")
        elif args.mode == "sort":
            self._sorter = SortingController(HOME_Q)
            self._station_color = ""
            self.create_subscription(String, "/cell/station_color", self._on_station_color, 10)
            self.get_logger().info("Colour sorting: waiting for a part at the pick station")
        elif args.mode == "sine":
            self._amp = np.radians(args.amp_deg)
            self._phase = np.arange(6) * 0.5

        self.create_timer(1.0 / RATE_HZ, self._tick)

    def _on_station_color(self, msg):
        self._station_color = msg.data

    def _tick(self):
        t = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        mode = self._args.mode

        if mode in ("constant", "ik"):
            q, gripper = self._fixed
        elif mode == "sine":
            w = 2 * np.pi * self._args.freq
            q = HOME_Q + self._amp * np.sin(w * t + self._phase)
            gripper = GRIPPER_CLOSED * 0.5 * (1 - np.cos(w * t))
        else:
            q, gripper, event = self._sorter.update(t, self._station_color)
            if event:
                self.get_logger().info(event)

        msg = Float64MultiArray()
        msg.data = [float(v) for v in q] + [float(gripper)]
        self._pub.publish(msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Publish UR5e joint commands on /ur5e/joint_commands")
    sub = parser.add_subparsers(dest="mode", required=True)

    constant = sub.add_parser("constant", help="hold a fixed joint configuration (constant input)")
    constant.add_argument("--deg", type=float, nargs=6, help="6 joint angles in degrees (default: home pose)")
    constant.add_argument("--gripper", type=float, default=GRIPPER_OPEN, help="0 = open, 255 = closed")

    sine = sub.add_parser("sine", help="sine wave on every joint around home (signal builder)")
    sine.add_argument("--amp-deg", type=float, default=20.0)
    sine.add_argument("--freq", type=float, default=0.2, help="Hz")

    sub.add_parser("sort", help="colour sorting: pick each part the station sensor reports and drop it in its bin")

    ik = sub.add_parser("ik", help="move the wrist flange to a world xyz with the gripper pointing down")
    ik.add_argument("--xyz", type=float, nargs=3, required=True, help="world coordinates in meters")
    ik.add_argument("--gripper", type=float, default=GRIPPER_OPEN)

    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = JointCommandPublisher(args)
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
