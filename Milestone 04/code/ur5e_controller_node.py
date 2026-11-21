"""
ROS 2 node (Milestone 4): trajectory planning + motor-level torque control of the UR5e, closed
loop through ROS 2 and the MuJoCo simulator.

    python3 ur5e_controller_node.py sort  [ctc|pdg|pd]   the colour-sorting application (default)
    python3 ur5e_controller_node.py joint [ctc|pdg|pd]   joint-space test trajectory (quintic + LSPB)
    python3 ur5e_controller_node.py task  [ctc|pdg|pd]   task-space test trajectory (lines + circle)

Loop: every /ur5e/joint_states message (250 Hz, stamped with the simulator time) ->
      sample the plan at that time -> control law -> /ur5e/effort_commands.
The arm starts on its brakes at the home pose; the node first plans a joint-space move to
the waiting pose above the pick station, then the test (or the sorting).

At the end of a test (or on Ctrl+C) it prints the tracking summary and saves the log to
Milestone 04/results/ros_<mode>_<controller>.npz.
"""
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control  # noqa: E402
import test_paths  # noqa: E402
from kinematics import forward_kinematics  # noqa: E402
from sorting_task import SortingApp, plan_to_station  # noqa: E402
from trajectories import Plan  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
FAST = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
REFERENCE_EVERY = 5          # publish the reference (display) at 250 / 5 = 50 Hz
PARTS_IN_CELL = 6
GRASP_LABELS = ("close the gripper", "lift", "open the gripper")   # contact with the part


class TestRun:
    """A fixed plan (joint or task test) played once, preceded by the move to the station."""

    def __init__(self, mode, q_start):
        to_station = plan_to_station(q_start)
        test = test_paths.PLANS[mode](to_station.q_end)
        self.plan = Plan(q_start)
        for step in to_station.steps + test.steps:
            self.plan.add(*step)
        self.test_start = to_station.T
        self.label = None

    def update(self, t, seen_color):
        q, dq, ddq, gripper, label, _ = self.plan.sample(t)
        event = label if label != self.label else None
        self.label = label
        return q, dq, ddq, gripper, event

    def done(self, t):
        return t > self.plan.T


class ControllerNode(Node):
    def __init__(self, mode, ctrl_name):
        super().__init__("ur5e_controller")
        self.mode, self.ctrl_name = mode, ctrl_name
        self.controller = control.make(ctrl_name)
        if hasattr(self.controller, "dt"):
            self.controller.dt = 1 / 250.0          # integral term: one control period (state rate)
        self.pub = self.create_publisher(Float64MultiArray, "/ur5e/effort_commands", FAST)
        self.ref_pub = self.create_publisher(JointState, "/ur5e/joint_reference", 10)
        self.create_subscription(JointState, "/ur5e/joint_states", self.on_state, FAST)
        self.station = ""
        self.create_subscription(String, "/cell/station_color", lambda m: setattr(self, "station", m.data), 10)
        self.task = None
        self.t0 = None
        self.n = 0
        self.rows = []
        self.labels = []
        self.compute_ms = []
        self.finished = False
        self.get_logger().info(f"{mode} run with the {self.controller.name} controller, waiting for the simulator...")

    def on_state(self, msg):
        if len(msg.position) < 6:
            return
        t_sim = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q, dq = np.array(msg.position[:6]), np.array(msg.velocity[:6])
        if self.task is None:
            self.t0 = t_sim
            self.task = SortingApp(q) if self.mode == "sort" else TestRun(self.mode, q)
            self.get_logger().info("Brakes released, controller running")
        t = t_sim - self.t0
        c0 = time.perf_counter()
        q_d, dq_d, ddq_d, gripper, event = self.task.update(t, self.station)
        tau = self.controller(q, dq, q_d, dq_d, ddq_d)
        self.pub.publish(Float64MultiArray(data=[float(v) for v in tau] + [float(gripper)]))
        self.compute_ms.append((time.perf_counter() - c0) * 1000)
        if event:
            self.get_logger().info(f"t = {t:5.1f} s  {event}")
        label = getattr(self.task, "label", "") or ""
        if self.n % REFERENCE_EVERY == 0:
            ref = JointState()
            ref.header.stamp = msg.header.stamp
            ref.header.frame_id = f"{self.controller.name}|{label}"
            ref.position, ref.velocity, ref.effort = q_d.tolist(), dq_d.tolist(), ddq_d.tolist()
            self.ref_pub.publish(ref)
        self.n += 1
        if label not in self.labels:
            self.labels.append(label)
        self.rows.append(np.concatenate([[t], q, dq, q_d, dq_d, tau, np.array(msg.effort[:6]) if len(msg.effort) >= 6
                                         else np.zeros(6), [self.labels.index(label), gripper]]))
        if isinstance(self.task, TestRun) and self.task.done(t) and not self.finished:
            self.finish()
        if self.mode == "sort" and sum(self.task.sorted.values()) == PARTS_IN_CELL and not self.finished:
            self.finish()                       # every part is in a bin; keep holding the arm

    def finish(self):
        if self.finished or not self.rows:
            return
        self.finished = True
        a = np.array(self.rows)
        t, q, q_d, tau = a[:, 0], a[:, 1:7], a[:, 13:19], a[:, 25:31]
        label_idx = a[:, 37].astype(int)
        e = np.degrees(np.abs(q_d - q))
        flange = 1000 * np.array([np.linalg.norm(forward_kinematics(qd)[:3, 3] - forward_kinematics(qq)[:3, 3])
                                  for qd, qq in zip(q_d, q)])
        if isinstance(self.task, TestRun):
            window = t >= self.task.test_start
            what = f"during the {self.mode}-space test ({t[window][0]:.1f} s to {t[-1]:.1f} s)"
        else:
            grasp = np.array([any(self.labels[i].startswith(g) for g in GRASP_LABELS) for i in label_idx])
            window = (t > 0.3) & ~grasp
            what = "while moving, gripper not touching a part"
        dt = np.diff(t)
        lines = [
            f"===== {self.mode.upper()} run, {self.controller.name} controller =====",
            f"  control loop: {len(t)} steps, {1 / np.median(dt):.0f} Hz median, compute {np.median(self.compute_ms):.2f} ms median "
            f"({np.max(self.compute_ms):.1f} ms max)",
            f"  {what}:",
            f"    joint error   max {e[window].max():.3f} deg (joint {np.argmax(e[window].max(axis=0)) + 1}), "
            f"RMS {np.sqrt(np.mean(e[window] ** 2)):.3f} deg",
            f"    flange error  max {flange[window].max():.2f} mm, RMS {np.sqrt(np.mean(flange[window] ** 2)):.2f} mm",
            f"    peak torque   {np.round(np.abs(tau[window]).max(axis=0), 1).tolist()} N m",
        ]
        if self.mode == "sort":
            lines.append(f"  during the grasps (pads squeezing the part): joint error max "
                         f"{e[~window & (t > 0.3)].max():.3f} deg" if (~window & (t > 0.3)).any() else "")
            lines.append(f"  parts sorted: {self.task.sorted}")
        else:
            lines.append(f"  at the end: joint error {e[-1].max():.4f} deg, flange error {flange[-1]:.3f} mm")
        self.get_logger().info("\n".join(lines))
        os.makedirs(RESULTS, exist_ok=True)
        path = os.path.join(RESULTS, f"ros_{self.mode}_{self.ctrl_name}.npz")
        np.savez(path, t=t, q=q, dq=a[:, 7:13], q_d=q_d, dq_d=a[:, 19:25], tau=tau, tau_applied=a[:, 31:37],
                 label=label_idx, labels=np.array(self.labels), gripper=a[:, 38], compute_ms=np.array(self.compute_ms))
        self.get_logger().info(f"log saved to Milestone 04/results/{os.path.basename(path)}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sort"
    ctrl = sys.argv[2] if len(sys.argv) > 2 else "ctc"
    if mode not in ("sort", "joint", "task") or ctrl not in control.CONTROLLERS:
        raise SystemExit("usage: ur5e_controller_node.py [sort|joint|task] [ctc|ctci|pdg|pd]")
    rclpy.init()
    node = ControllerNode(mode, ctrl)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
