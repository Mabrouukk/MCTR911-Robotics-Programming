#!/usr/bin/env python3
import numpy as np

# ROS 2 is only needed by the node at the bottom of the file. Guarding the
# import lets the maths be imported and tested on a machine without ROS 2
# (see Challenge 01/verify/verify_kinematics.py).
try:
    import rclpy
    from rclpy.executors import ExternalShutdownException #ben shut down 
    from rclpy.node import Node
    from sensor_msgs.msg import JointState # messege 34an n3rf el joint state bta3 el robot
    from std_msgs.msg import Float64MultiArray
except ImportError:
    rclpy = None
    Node = object

"""
In the setup.py, change the entry point to include this file's name without the brackets

    entry_points={
        'console_scripts': [
            'kinematics = panda_kinematics.kinematics:main',
        ],
    },
"""
# ==========================================================
# KINEMATICS IMPLEMENTATION — EDIT ONLY THIS SECTION
# ==========================================================

# Standard DH parameters for links 1 through 7.
#
# Format: one row per link, columns are [alpha, a, d].
# theta is the joint variable and is not stored here.
#
# Convention: A_i = Rz(theta_i) . Tz(d_i) . Tx(a_i) . Rx(alpha_i)
# Frame i sits on the axis of joint i+1, so z_(i-1) is the axis joint i turns about.
#
# How each row comes out of panda.xml. In MuJoCo a child body is placed with
# T(pos) . R(quat) relative to its parent, then turns about its own z (axis="0 0 1").
# quat "1 -1 0 0" is Rx(-90 deg), "1 1 0 0" is Rx(+90 deg). Writing the chain
# link0 -> link7 out and regrouping it into Rz . Tz . Tx . Rx blocks gives:
#
#   i | alpha |    a    |   d   | where it comes from in panda.xml
#   1 | -pi/2 |  0      | 0.333 | link1 pos z=0.333, link2 quat Rx(-90)
#   2 | +pi/2 |  0      | 0     | link3 quat Rx(+90)
#   3 | +pi/2 |  0.0825 | 0.316 | link3 pos y=-0.316 (becomes +z after the Rx(+90)), link4 pos x=0.0825, quat Rx(+90)
#   4 | -pi/2 | -0.0825 | 0     | link5 pos x=-0.0825, quat Rx(-90)
#   5 | +pi/2 |  0      | 0.384 | link5 pos y=0.384 (becomes +z after the Rx(-90)), link6 quat Rx(+90)
#   6 | +pi/2 |  0.088  | 0     | link7 pos x=0.088, quat Rx(+90)
#   7 |  0    |  0      | 0     | frame 7 is exactly the link7 body; the 0.107 to the hand is in T_L7_HAND
#
# No theta offsets are needed: at q = 0 every DH frame lines up with its MuJoCo body.
DH_PARAMS = [
    [-np.pi / 2,  0.0,    0.333],  # link 1 shoulder # kol row le kol joint andena 7 rows 7 joints , kol row (alpha,a,d)
    [ np.pi / 2,  0.0,    0.0  ],  # link 2 joints 2 # el alpha el joint el baady twisted ad eh ala x 
    [ np.pi / 2,  0.0825, 0.316],  # link 3 # el x heya el sideway distance along x axis, d heya el distance along z axis
    [-np.pi / 2, -0.0825, 0.0  ],  # link 4 # tab leh mafish hena theta 34an el theta betetghayar de el change el robot bylfo (q) w dah bytghy kol ma el robot yethrk fa el table de bas el hagat el sabta 
    [ np.pi / 2,  0.0,    0.384],  # link 5 # el distances de kolaha meters
    [ np.pi / 2,  0.088,  0.0  ],  # link 6
    [ 0.0,        0.0,    0.0  ],  # link 7
]

# Fixed transform from link7 to the 'hand' body.
# 4x4 numpy array.
#
# <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
# quat = (cos(t/2), 0, 0, sin(t/2)) with t = -45 deg, a pure rotation about z.
_c, _s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
T_L7_HAND = np.array([ #awel talata columns dol rotation matrix, el 4th column de translation vector
    [_c,  -_s,  0.0, 0.0  ],
    [_s,   _c,  0.0, 0.0  ],
    [0.0,  0.0, 1.0, 0.107],
    [0.0,  0.0, 0.0, 1.0  ],
])

# Joint limits for joints 1 through 7, in radians.
# numpy arrays of shape (7,).
#
# Joints 1, 3, 5, 7 use the class default range="-2.8973 2.8973";
# joints 2, 4, 6 override it on their own <joint> tag.
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# The target pose the node will try to reach.
# 4x4 numpy array in the 'link0' frame.
#
# TARGET_JOINTS is a configuration inside the limits, picked so the hand is in
# front of the robot, above the table and pointing down. TARGET_POSE is set to
# forward_kinematics(TARGET_JOINTS) right after the functions below, so the
# target is reachable by construction.
TARGET_JOINTS = np.array([0.5, 0.3, -0.2, -2.0, 0.1, 2.3, 0.4]) # de el angles bta3t el joints el hadeha lel forward kinematics 3shan ygeb el pose bta3t el hand we el FK btrg3 ek oriantation
TARGET_POSE = None #DE el hayb2a feha el xyz towards el haga el ana 3ayz awslha S

# The 'home' keyframe from panda.xml. Used as a fallback IK seed.
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
#law ayez degrees np.degrees(TARGET_JOINTS)
# Upper bound on how far the hand origin can be from the shoulder (where
# joints 1 and 2 meet, 0.333 above link0). Triangle inequality over the three
# rigid segments shoulder->elbow, elbow->wrist, wrist->hand:
#   |(0.0825, 0.316)| + |(0.0825, 0.384)| + |(0.088, 0.107)| = 0.858 m
# Anything further away is rejected before iterating. Because it is an upper
# bound it never rejects a reachable target.
SHOULDER = np.array([0.0, 0.0, 0.333])
MAX_REACH = np.hypot(0.0825, 0.316) + np.hypot(0.0825, 0.384) + np.hypot(0.088, 0.107)

# IK settings.
IK_POS_TOL = 1e-4        # 0.1 mm
IK_ROT_TOL = 1e-3        # ~0.06 deg
IK_MAX_ITER = 200        # per seed
IK_MAX_STEP = 0.2        # rad, largest joint change allowed per iteration
IK_LAMBDA_BAR = 1e-4     # damping floor, keeps J^T J + lambda I invertible at singularities
IK_N_RANDOM_SEEDS = 10   # extra restarts if the given seed fails


def dh_transform(alpha, a, d, theta):
    """
    Single-link transformation matrix for standard DH.

    alpha: twist angle (rad)
    a:     link length (m)
    d:     link offset (m)
    theta: joint angle (rad)

    Returns a 4x4 numpy array.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def _fk_frames(q):
    """Frames 0..7 in link0 (frame 0 is the base), plus the hand frame."""
    T = np.eye(4)
    frames = [T]
    for (alpha, a, d), theta in zip(DH_PARAMS, q):
        T = T @ dh_transform(alpha, a, d, theta)
        frames.append(T)
    return frames, T @ T_L7_HAND


def forward_kinematics(q):
    """
    Forward kinematics for the Panda arm.

    q: numpy array of 7 joint angles (rad).

    Returns a 4x4 homogeneous transform of the 'hand' frame
    relative to 'link0'.
    """
    return _fk_frames(np.asarray(q, dtype=float))[1]


def jacobian(q):
    """
    6x7 geometric Jacobian of the hand frame in link0, rows [v; w].

    Joint i turns about z_(i-1) through o_(i-1), so its column is
    [z_(i-1) x (p_hand - o_(i-1)); z_(i-1)].
    Also returns the hand pose, since it falls out of the same pass.
    """
    frames, T_hand = _fk_frames(q)
    p = T_hand[:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z = frames[i][:3, 2]
        o = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J, T_hand


def _rotation_error(R_target, R):
    """
    Axis-angle vector (in link0) that rotates R onto R_target.

    Goes through a quaternion so it stays well defined at 0 and 180 deg,
    where the arccos(trace) formula loses precision.
    """
    R_err = R_target @ R.T
    w = np.sqrt(max(0.0, 1.0 + np.trace(R_err))) / 2.0
    v = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])
    if w > 1e-6:
        v = v / (2.0 * w)                     # v = sin(t/2) * axis
    else:
        # 180 deg: the vee() part vanishes. Here R_err = 2 a a^T - I, so any
        # non-zero column of (R_err + I) / 2 is the axis (up to sign, and at
        # 180 deg either sign is the same rotation).
        M = (R_err + np.eye(3)) / 2.0
        axis = M[:, int(np.argmax(np.diag(M)))]
        return np.pi * axis / np.linalg.norm(axis)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, w)
    return angle * v / s


def pose_error(target_pose, T):
    """6-vector [position error (m); orientation error (rad)] in link0."""
    return np.concatenate([
        target_pose[:3, 3] - T[:3, 3],
        _rotation_error(target_pose[:3, :3], T[:3, :3]),
    ])


def _lm_step(J, e):
    """
    One Levenberg-Marquardt step with Sugihara's error-scaled damping.

    dq = (J^T J + lambda I)^-1 J^T e,   lambda = 0.5 |e|^2 + IK_LAMBDA_BAR

    Far from the target lambda is large, so the step is short and cautious.
    Close to it lambda shrinks to IK_LAMBDA_BAR and the step becomes plain
    Gauss-Newton, so convergence is fast and lands exactly on the target
    (a fixed damping would leave a small steady-state error). At a singularity
    J^T J loses rank, but adding lambda I keeps it invertible, so the step
    stays finite: it just makes no progress along the lost direction.

    dq minimises |J dq - e|^2 + lambda |dq|^2, so the damping also penalises
    large joint changes: each step reduces the error with as little joint
    motion as it can. That is what keeps the answer close to the seed among
    the many solutions of a 7-joint arm.
    """
    lam = 0.5 * e @ e + IK_LAMBDA_BAR
    return np.linalg.solve(J.T @ J + lam * np.eye(7), J.T @ e)


def _solve_from(target_pose, q_start):
    """Iterate from q_start. Returns (q, converged)."""
    q = np.clip(q_start, Q_MIN, Q_MAX)
    for _ in range(IK_MAX_ITER):
        J, T = jacobian(q)
        e = pose_error(target_pose, T)
        if np.linalg.norm(e[:3]) < IK_POS_TOL and np.linalg.norm(e[3:]) < IK_ROT_TOL:
            return q, True

        dq = _lm_step(J, e)

        # Joints sitting on a limit and being pushed further into it are
        # frozen (their Jacobian column removed) and the step is recomputed,
        # so the other joints make up for them instead of the solve stalling
        # against the clamp.
        at_limit = ((q <= Q_MIN + 1e-9) & (dq < 0)) | ((q >= Q_MAX - 1e-9) & (dq > 0))
        if at_limit.any():
            J_free = J.copy()
            J_free[:, at_limit] = 0.0
            dq = _lm_step(J_free, e)
            dq[at_limit] = 0.0

        # Step size limit: a linearisation is only good for small steps.
        biggest = np.max(np.abs(dq))
        if biggest > IK_MAX_STEP:
            dq *= IK_MAX_STEP / biggest

        # Clamp inside the loop so every iterate (and the answer) is within limits.
        q = np.clip(q + dq, Q_MIN, Q_MAX)
    return q, False


def inverse_kinematics(target_pose, q0):
    """
    Inverse kinematics for the Panda arm.

    Any correct method is acceptable — numerical, analytical,
    or a mix of both. See the guide, Section 8.

    target_pose: 4x4 desired pose of the 'hand' frame in 'link0'.
    q0:          numpy array of 7 joint angles to seed the solve.

    Returns a tuple (q_solution, converged_flag), where
    q_solution is a numpy array of 7 joint angles and
    converged_flag is a bool.

    Method: Levenberg-Marquardt (damped least squares) on the full 6D pose
    error, starting from the seed, with joint limits clamped every iteration.
    Each step is the smallest joint change that reduces the error, so the
    answer is the solution nearest the seed, i.e. nearest where the arm is now.

    If the seed does not converge (e.g. it is stuck against a limit), the solve
    is restarted from the home pose and a few fixed random poses; among the
    restarts that converge, the one closest to the seed is returned.

    On failure the clamped seed is returned with False, so a caller that
    publishes it anyway makes the arm hold still instead of jumping.
    """
    target_pose = np.asarray(target_pose, dtype=float)
    q_ref = np.clip(np.asarray(q0, dtype=float)[:7], Q_MIN, Q_MAX)

    # Reachability, quick reject: further than the arm can possibly stretch.
    if np.linalg.norm(target_pose[:3, 3] - SHOULDER) > MAX_REACH:
        return q_ref, False

    q, ok = _solve_from(target_pose, q_ref)
    if ok:
        return q, True

    # Fixed generator so the same call always gives the same answer.
    rng = np.random.default_rng(0)
    seeds = [Q_HOME] + [rng.uniform(Q_MIN, Q_MAX) for _ in range(IK_N_RANDOM_SEEDS)]
    best = None
    for seed in seeds:
        q, ok = _solve_from(target_pose, seed)
        if ok and (best is None or np.linalg.norm(q - q_ref) < np.linalg.norm(best - q_ref)):
            best = q
    if best is not None:
        return best, True

    # Reachable distance but no joint vector within limits gives this pose
    # (bad orientation for that spot, inside the shoulder region, ...).
    return q_ref, False


TARGET_POSE = forward_kinematics(TARGET_JOINTS)


# =================================================================
# DO NOT MODIFY BELOW (OR MODIFY, IF YOU KNOW WHAT YOU ARE DOING)
# =================================================================

CMD_SIZE = 9
GRIPPER_CTRL = 255.0


class PandaKinematicsNode(Node):
    def __init__(self):
        super().__init__('panda_kinematics')
        self.q_current = np.zeros(7)
        # Changes to the template: wait for a real joint state before solving
        # (the zeros placeholder is a poor seed, q4 = 0 is outside its limits),
        # and solve once per target instead of every tick. Re-solving from a
        # moving seed every 50 ms can hop between redundant solutions and make
        # the arm jitter; the cached answer keeps the command fixed.
        self.have_state = False
        self.q_cmd = None

        self.create_subscription(
            JointState, 'joint_state', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, 'joint_commands', 10)

        self.create_timer(0.05, self.tick)

        if TARGET_POSE is not None:
            self.get_logger().info(
                f"Target pose:\n{np.round(TARGET_POSE, 4)}")
        else:
            self.get_logger().warn(
                "TARGET_POSE is None — set it before running.")

    def on_joint_state(self, msg):
        self.q_current = np.array(msg.position[:7])
        self.have_state = True

    def tick(self):
        if TARGET_POSE is None or not self.have_state:
            return

        if self.q_cmd is None:
            q_sol, ok = inverse_kinematics(TARGET_POSE, self.q_current)
            if not ok:
                self.get_logger().warn(
                    "IK did not converge.",
                    throttle_duration_sec=2.0)
            else:
                self.q_cmd = q_sol
                err = pose_error(TARGET_POSE, forward_kinematics(q_sol))
                self.get_logger().info(
                    f"IK solution (rad): {np.round(q_sol, 4).tolist()}\n"
                    f"FK check: position error {np.linalg.norm(err[:3]) * 1000:.3f} mm, "
                    f"orientation error {np.degrees(np.linalg.norm(err[3:])):.4f} deg")
        else:
            q_sol = self.q_cmd

        cmd = Float64MultiArray()
        cmd.data = q_sol.tolist() + [GRIPPER_CTRL, 0.0]
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PandaKinematicsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C: stop quietly instead of printing a traceback
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
