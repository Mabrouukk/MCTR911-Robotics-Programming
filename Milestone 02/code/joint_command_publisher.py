import argparse
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematics import HOME_Q, ROBOT_BASE_POS, forward_kinematics, inverse_kinematics, wrap_angle  # noqa: E402

RATE_HZ = 50.0
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0
TOOL_POINTING_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

# (label, world xyz of the wrist flange or None for home, gripper command)
PICK_PLACE_SEQUENCE = [
    ("home", None, GRIPPER_OPEN),
    ("above conveyor", (-0.2, 0.6, 0.70), GRIPPER_OPEN),
    ("down to conveyor", (-0.2, 0.6, 0.55), GRIPPER_OPEN),
    ("close gripper", (-0.2, 0.6, 0.55), GRIPPER_CLOSED),
    ("lift", (-0.2, 0.6, 0.70), GRIPPER_CLOSED),
    ("above pallet", (0.6, -0.15, 0.60), GRIPPER_CLOSED),
    ("down to pallet", (0.6, -0.15, 0.40), GRIPPER_CLOSED),
    ("open gripper", (0.6, -0.15, 0.40), GRIPPER_OPEN),
    ("retreat", (0.6, -0.15, 0.60), GRIPPER_OPEN),
]


def ik_tool_down(xyz_world, reference_q):
    """Joint angles that put the wrist flange at xyz_world with the gripper pointing down.

    Picks the IK solution nearest to reference_q and unwraps it so the arm takes the short way round.
    """
    T = np.eye(4)
    T[:3, :3] = TOOL_POINTING_DOWN
    T[:3, 3] = np.asarray(xyz_world, dtype=float) - ROBOT_BASE_POS
    solutions = inverse_kinematics(T)
    if len(solutions) == 0:
        raise ValueError(f"Target {tuple(xyz_world)} is out of the UR5e's reach")
    deltas = wrap_angle(solutions - reference_q)
    return reference_q + deltas[np.argmin(np.abs(deltas).sum(axis=1))]


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
        elif args.mode == "sequence":
            self._steps = []
            q = HOME_Q
            for label, xyz, gripper in PICK_PLACE_SEQUENCE:
                q = HOME_Q if xyz is None else ik_tool_down(xyz, q)
                self._steps.append((label, q, gripper))
            self._current_step = -1
        elif args.mode == "sine":
            self._amp = np.radians(args.amp_deg)
            self._phase = np.arange(6) * 0.5

        self.create_timer(1.0 / RATE_HZ, self._tick)

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
            index = int(t // self._args.period) % len(self._steps)
            label, q, gripper = self._steps[index]
            if index != self._current_step:
                self._current_step = index
                self.get_logger().info(f"Step {index + 1}/{len(self._steps)}: {label}")

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

    sequence = sub.add_parser("sequence", help="step through pick-and-place waypoints (signal builder)")
    sequence.add_argument("--period", type=float, default=2.5, help="seconds per step")

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
