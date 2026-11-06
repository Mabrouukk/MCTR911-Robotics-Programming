"""
ROS 2 node (Milestone 3): drives the UR5e and checks our kinematic equations
against the simulator's sensors, live, through ROS 2.

    python3 kinematics_validator.py forward   joint-space input: sine waves on every joint
    python3 kinematics_validator.py inverse   task-space input: a flange path turned into joint
                                              commands by IK, J^-1 and J^-1(... - J_dot q_dot)
    python3 kinematics_validator.py sort      Milestone 2's colour sorting, with the checks running

For every instant the bridge reports, it matches the joint readings
(/ur5e/joint_states, /ur5e/joint_accelerations) with the flange readings
(/ur5e/ee_pose, /ur5e/ee_twist, /ur5e/ee_accel) by their simulator time stamp
and runs:
    forward: FK(q), J q_dot, J q_ddot + J_dot q_dot          vs  sensed flange pose / twist / acceleration
    inverse: IK(pose), J^-1 twist, J^-1 (accel - J_dot q_dot) vs  sensed q / q_dot / q_ddot
Once a second it prints the worst errors so far; at the end, a PASS/FAIL summary.
"""
import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import AccelStamped, PoseStamped, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_path  # noqa: E402
from kinematic_checks import kinematics_check  # noqa: E402
from kinematics import HOME_Q  # noqa: E402
from sorting_task import GRIPPER_OPEN, SortingController  # noqa: E402

RATE_HZ = 50.0
# Same "simple tolerance" as validate_kinematics.py.
TOL = {"fk_pos": 1e-6, "fk_rot": 1e-6, "fv_lin": 1e-6, "fv_ang": 1e-6, "fa_lin": 1e-5, "fa_ang": 1e-5,
       "ik": 1e-6, "iv": 1e-6, "ia": 1e-5}
LABELS = {"fk_pos": "forward position  FK(q) vs sensed position [m]",
          "fk_rot": "forward position  FK(q) vs sensed orientation [rad]",
          "fv_lin": "forward velocity  J q_dot vs sensed linear velocity [m/s]",
          "fv_ang": "forward velocity  J q_dot vs sensed angular velocity [rad/s]",
          "fa_lin": "forward accel.    J q_ddot + J_dot q_dot vs sensed linear acc. [m/s^2]",
          "fa_ang": "forward accel.    J q_ddot + J_dot q_dot vs sensed angular acc. [rad/s^2]",
          "ik": "inverse position  IK(sensed pose) vs sensed q [rad]",
          "iv": "inverse velocity  J^-1 sensed twist vs sensed q_dot [rad/s]",
          "ia": "inverse accel.    J^-1 (sensed acc. - J_dot q_dot) vs sensed q_ddot [rad/s^2]"}
PATH_TOL = 0.005      # 5 mm: the flange following the planned path in inverse mode

# Joint sine waves for the forward test (same as validate_kinematics.py).
AMP = np.radians([40, 25, 30, 40, 40, 60])
FREQ = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
FORWARD_DURATION = 8.0
LEAD_MOVE = 3.0       # inverse mode: seconds to move smoothly from home to the path start,
LEAD_IN = 3.5         # then hold there until the path starts at LEAD_IN


def stamp_key(stamp):
    return round(stamp.sec + stamp.nanosec * 1e-9, 4)


class KinematicsValidator(Node):
    def __init__(self, mode):
        super().__init__("kinematics_validator")
        self.mode = mode
        self.pub = self.create_publisher(Float64MultiArray, "/ur5e/joint_commands", 10)
        self.create_subscription(JointState, "/ur5e/joint_states", self.on_joints, 10)
        self.create_subscription(Float64MultiArray, "/ur5e/joint_accelerations", self.on_acc, 10)
        self.create_subscription(PoseStamped, "/ur5e/ee_pose", self.on_pose, 10)
        self.create_subscription(TwistStamped, "/ur5e/ee_twist", self.on_twist, 10)
        self.create_subscription(AccelStamped, "/ur5e/ee_accel", self.on_accel, 10)
        self.samples = {}           # sim time -> partial sample, filled by the callbacks
        self.worst = dict.fromkeys(TOL, 0.0)
        self.worst_path = 0.0
        self.worst_path_t = 0.0
        self.n = 0
        self.min_sigma = np.inf
        self.last_report = -1.0
        self.t0 = None              # wall time the commands started
        self.sim_t0 = None          # simulator time the commands started
        self.last_sim_t = None
        self.last_recv = None       # wall time that last state arrived
        self.done = False

        if mode == "inverse":
            self.path = task_path.plan_joint_path(0.002)
            # Position servos lag by (kd/kp) q_dot = 0.2 s * q_dot; adding that term to
            # the command cancels it (velocity feed-forward, from our inverse velocity).
            self.kv_over_kp = 0.2
            self.get_logger().info(f"Inverse test: {LEAD_IN:.0f} s to the path start, then {task_path.T_TOTAL:.1f} s "
                                   "of joint commands from IK / J^-1 / J^-1(a - J_dot q_dot)")
        elif mode == "sort":
            self.sorter = SortingController(HOME_Q)
            self.station = ""
            self.create_subscription(String, "/cell/station_color", lambda m: setattr(self, "station", m.data), 10)
        self.create_timer(1.0 / RATE_HZ, self.tick)

    # ---------------------------------------------------------------- commands
    def tick(self):
        if self.last_sim_t is None:
            return                              # wait for the simulator
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            # The simulator runs locked to the wall clock, so its time "now" is the
            # time of the last state plus how long ago that state arrived (up to
            # 33 ms at 30 Hz; ignoring this shows up as millimetres of fake error).
            self.t0, self.sim_t0 = now, self.last_sim_t + (now - self.last_recv)
            self.get_logger().info(f"Starting the {self.mode} test")
        t = now - self.t0
        gripper = GRIPPER_OPEN
        if self.mode == "forward":
            q = HOME_Q + AMP * np.sin(2 * np.pi * FREQ * min(t, FORWARD_DURATION))
            if t > FORWARD_DURATION + 1.0:
                self.finish()
        elif self.mode == "inverse":
            if t < LEAD_IN:
                # Smoothly from home to the first point of the path (with the same
                # velocity feed-forward), then hold so the arm is settled there.
                tm = min(t, LEAD_MOVE)
                s = 0.5 - 0.5 * np.cos(np.pi * tm / LEAD_MOVE)
                s_dot = 0.5 * np.pi / LEAD_MOVE * np.sin(np.pi * tm / LEAD_MOVE)
                q = HOME_Q + s * (self.path["q"][0] - HOME_Q) + self.kv_over_kp * s_dot * (self.path["q"][0] - HOME_Q)
            else:
                # Each command is held for one timer period (zero-order hold), which delays
                # it by half a period on average, so aim at the middle of the hold window.
                tp = min(t - LEAD_IN + 0.5 / RATE_HZ, task_path.T_TOTAL)
                q = np.array([np.interp(tp, self.path["t"], self.path["q"][:, j]) for j in range(6)])
                dq = np.array([np.interp(tp, self.path["t"], self.path["dq"][:, j]) for j in range(6)])
                q = q + self.kv_over_kp * dq
            if t > LEAD_IN + task_path.T_TOTAL + 1.5:
                self.finish()
        else:
            q, gripper, event = self.sorter.update(t, self.station)
            if event:
                self.get_logger().info(event)
        self.pub.publish(Float64MultiArray(data=[float(v) for v in q] + [float(gripper)]))

    # ---------------------------------------------------------------- sensing
    def _put(self, key, field, value):
        sample = self.samples.setdefault(key, {})
        sample[field] = value
        if len(sample) == 5:
            del self.samples[key]
            self.evaluate(key, sample)
        if len(self.samples) > 100:            # drop instants that never completed
            for old in sorted(self.samples)[:50]:
                del self.samples[old]

    def on_joints(self, msg):
        key = stamp_key(msg.header.stamp)
        self.last_sim_t = key
        self.last_recv = self.get_clock().now().nanoseconds * 1e-9
        self._put(key, "joints", (np.array(msg.position), np.array(msg.velocity)))

    def on_acc(self, msg):
        self._put(round(msg.data[0], 4), "ddq", np.array(msg.data[1:7]))

    def on_pose(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        x, y, z, w = o.x, o.y, o.z, o.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        self._put(stamp_key(msg.header.stamp), "pose", (np.array([p.x, p.y, p.z]), R))

    def on_twist(self, msg):
        lin, ang = msg.twist.linear, msg.twist.angular
        self._put(stamp_key(msg.header.stamp), "twist", (np.array([lin.x, lin.y, lin.z]), np.array([ang.x, ang.y, ang.z])))

    def on_accel(self, msg):
        lin, ang = msg.accel.linear, msg.accel.angular
        self._put(stamp_key(msg.header.stamp), "accel", (np.array([lin.x, lin.y, lin.z]), np.array([ang.x, ang.y, ang.z])))

    def evaluate(self, sim_t, s):
        if self.t0 is None or self.done:
            return
        (q, dq), ddq = s["joints"], s["ddq"]
        (p, R), (v, w), (a, alpha) = s["pose"], s["twist"], s["accel"]
        ee = {"p": p, "R": R, "v": v, "w": w, "a": a, "alpha": alpha}
        c = kinematics_check(q, dq, ddq, ee)
        self.n += 1
        self.min_sigma = min(self.min_sigma, c["sigma_min"])
        for k in TOL:
            if k in ("ik", "iv", "ia") and c["sigma_min"] < 0.02:
                continue                           # inverse not unique right at a singularity
            self.worst[k] = max(self.worst[k], c[k])
        path_err = None
        if self.mode == "inverse":
            tp = sim_t - self.sim_t0 - LEAD_IN
            if tp >= 0:
                p_ref, _, _ = task_path.flange_target(tp)
                path_err = np.linalg.norm(p - p_ref)
                if path_err > self.worst_path:
                    self.worst_path, self.worst_path_t = path_err, tp
        if sim_t - self.last_report >= 1.0:
            self.last_report = sim_t
            line = (f"t={sim_t - self.sim_t0:5.1f}s | forward: pos {c['fk_pos']:.0e} m, vel {c['fv_lin']:.0e} m/s, "
                    f"acc {c['fa_lin']:.0e} m/s2 | inverse: IK {c['ik']:.0e} rad, J^-1v {c['iv']:.0e} rad/s, "
                    f"J^-1a {c['ia']:.0e} rad/s2 | sigma_min {c['sigma_min']:.3f}")
            if path_err is not None:
                line += f" | path error {path_err * 1000:.2f} mm"
            self.get_logger().info(line)

    def finish(self):
        if self.done:
            return
        self.done = True
        lines = [f"===== {self.mode.upper()} test: {self.n} instants checked, "
                 f"smallest sigma_min {self.min_sigma:.3f} m/rad ====="]
        ok_all = True
        for k, label in LABELS.items():
            ok = self.worst[k] < TOL[k]
            ok_all &= ok
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {label:76s} worst {self.worst[k]:.1e}  (tol {TOL[k]:.0e})")
        if self.mode == "inverse":
            ok = self.worst_path < PATH_TOL
            ok_all &= ok
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {'flange follows the planned path (via ROS 2, 50 Hz commands) [m]':76s} "
                         f"worst {self.worst_path:.1e}  (tol {PATH_TOL:.0e}), at t = {self.worst_path_t:.2f} s of the path")
        lines.append("  ALL PASSED" if ok_all else "  SOME CHECKS FAILED")
        self.get_logger().info("\n".join(lines))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "forward"
    if mode not in ("forward", "inverse", "sort"):
        raise SystemExit("usage: kinematics_validator.py [forward|inverse|sort]")
    rclpy.init()
    node = KinematicsValidator(mode)
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
