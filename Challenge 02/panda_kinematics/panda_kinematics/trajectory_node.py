#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np


# ==========================================================
#  EDIT ONLY THIS SECTION
# ==========================================================

# ---------------------------------------------------------------
# Copied from Challenge 1 (kinematics.py): DH table, FK, IK.
# ---------------------------------------------------------------

# Standard DH, one row per link: [alpha, a, d]. theta is the joint variable.
# A_i = Rz(theta_i) . Tz(d_i) . Tx(a_i) . Rx(alpha_i)
# Frame i sits on the axis of joint i+1, so joint i turns about z_(i-1).
DH_PARAMS = [
    [-np.pi / 2,  0.0,    0.333],  # link 1
    [ np.pi / 2,  0.0,    0.0  ],  # link 2
    [ np.pi / 2,  0.0825, 0.316],  # link 3
    [-np.pi / 2, -0.0825, 0.0  ],  # link 4
    [ np.pi / 2,  0.0,    0.384],  # link 5
    [ np.pi / 2,  0.088,  0.0  ],  # link 6
    [ 0.0,        0.0,    0.0  ],  # link 7
]

# link7 -> hand: <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
# = 0.107 m along z, then -45 deg about z.
_c, _s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
T_L7_HAND = np.array([
    [_c,  -_s,  0.0, 0.0  ],
    [_s,   _c,  0.0, 0.0  ],
    [0.0,  0.0, 1.0, 0.107],
    [0.0,  0.0, 0.0, 1.0  ],
])

# Joint position limits (rad), from the <joint> tags in panda.xml.
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# Joint velocity (rad/s) and acceleration (rad/s^2) limits of the real Panda,
# from the Franka datasheet. panda.xml has no velocity limits, so we take them
# from the robot itself. The planner never uses more than LIMIT_FRACTION of them.
QD_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
LIMIT_FRACTION = 0.8

# The 'home' keyframe from panda.xml. Used as a fallback IK seed.
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# IK settings (same as Challenge 1).
SHOULDER = np.array([0.0, 0.0, 0.333])
MAX_REACH = np.hypot(0.0825, 0.316) + np.hypot(0.0825, 0.384) + np.hypot(0.088, 0.107)
IK_POS_TOL = 1e-4
IK_ROT_TOL = 1e-3
IK_MAX_ITER = 200
IK_MAX_STEP = 0.2
IK_LAMBDA_BAR = 1e-4
IK_N_RANDOM_SEEDS = 10


def dh_transform(alpha, a, d, theta):
    """Single-link standard DH transform. Returns a 4x4 numpy array."""
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
    """4x4 pose of the 'hand' frame in 'link0' for 7 joint angles q (rad)."""
    return _fk_frames(np.asarray(q, dtype=float))[1]


def _geometric_jacobian(q):
    """
    Full 6x7 geometric Jacobian of the hand, rows [v; w], plus the hand pose.
    Column i: [z_(i-1) x (p - o_(i-1)); z_(i-1)]. Used by the IK.
    """
    frames, T_hand = _fk_frames(np.asarray(q, dtype=float))
    p = T_hand[:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z = frames[i][:3, 2]
        o = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J, T_hand


def _rotation_error(R_target, R):
    """Axis-angle vector (in link0) that rotates R onto R_target."""
    R_err = R_target @ R.T
    w = np.sqrt(max(0.0, 1.0 + np.trace(R_err))) / 2.0
    v = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])
    if w > 1e-6:
        v = v / (2.0 * w)
    else:
        M = (R_err + np.eye(3)) / 2.0
        axis = M[:, int(np.argmax(np.diag(M)))]
        return np.pi * axis / np.linalg.norm(axis)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3)
    return 2.0 * np.arctan2(s, w) * v / s


def pose_error(target_pose, T):
    """6-vector [position error (m); orientation error (rad)] in link0."""
    return np.concatenate([
        target_pose[:3, 3] - T[:3, 3],
        _rotation_error(target_pose[:3, :3], T[:3, :3]),
    ])


def _lm_step(J, e):
    """Levenberg-Marquardt step, lambda = 0.5 |e|^2 + IK_LAMBDA_BAR."""
    lam = 0.5 * e @ e + IK_LAMBDA_BAR
    return np.linalg.solve(J.T @ J + lam * np.eye(7), J.T @ e)


def _solve_from(target_pose, q_start):
    """Iterate from q_start. Returns (q, converged)."""
    q = np.clip(q_start, Q_MIN, Q_MAX)
    for _ in range(IK_MAX_ITER):
        J, T = _geometric_jacobian(q)
        e = pose_error(target_pose, T)
        if np.linalg.norm(e[:3]) < IK_POS_TOL and np.linalg.norm(e[3:]) < IK_ROT_TOL:
            return q, True
        dq = _lm_step(J, e)
        at_limit = ((q <= Q_MIN + 1e-9) & (dq < 0)) | ((q >= Q_MAX - 1e-9) & (dq > 0))
        if at_limit.any():
            J_free = J.copy()
            J_free[:, at_limit] = 0.0
            dq = _lm_step(J_free, e)
            dq[at_limit] = 0.0
        biggest = np.max(np.abs(dq))
        if biggest > IK_MAX_STEP:
            dq *= IK_MAX_STEP / biggest
        q = np.clip(q + dq, Q_MIN, Q_MAX)
    return q, False


def inverse_kinematics(target_pose, q0):
    """
    Levenberg-Marquardt IK on the full 6D pose, joint limits clamped every
    iteration. Seeds: q0 first, then home and 10 fixed random poses.
    Returns (q_solution, converged_flag).
    """
    target_pose = np.asarray(target_pose, dtype=float)
    q_ref = np.clip(np.asarray(q0, dtype=float)[:7], Q_MIN, Q_MAX)
    if np.linalg.norm(target_pose[:3, 3] - SHOULDER) > MAX_REACH:
        return q_ref, False
    q, ok = _solve_from(target_pose, q_ref)
    if ok:
        return q, True
    rng = np.random.default_rng(0)
    seeds = [Q_HOME] + [rng.uniform(Q_MIN, Q_MAX) for _ in range(IK_N_RANDOM_SEEDS)]
    best = None
    for seed in seeds:
        q, ok = _solve_from(target_pose, seed)
        if ok and (best is None or np.linalg.norm(q - q_ref) < np.linalg.norm(best - q_ref)):
            best = q
    if best is not None:
        return best, True
    return q_ref, False


# ---------------------------------------------------------------
# Challenge 2: velocity and acceleration kinematics.
# ---------------------------------------------------------------

def compute_jacobian(q):
    """
    Linear geometric Jacobian of the 'hand' frame in the base frame.

    q: numpy array of 7 joint angles (rad).

    Returns a 3x7 numpy array mapping joint velocities to
    linear end-effector velocities (x, y, z) in the base frame.

    All 7 joints are revolute. Joint i turns about the axis z_(i-1), which
    passes through the point o_(i-1) (both read from the DH frame i-1).
    If only joint i turns, at rate qd_i, the hand point p moves on a circle
    around that axis, with velocity qd_i * z_(i-1) x (p - o_(i-1)).
    So column i is

        Jv_i = z_(i-1) x (p - o_(i-1))

    and the hand velocity is the sum over all joints: p_dot = Jv @ q_dot.
    """
    frames, T_hand = _fk_frames(np.asarray(q, dtype=float))
    p = T_hand[:3, 3]
    Jv = np.zeros((3, 7))
    for i in range(7):
        z = frames[i][:3, 2]     # axis of joint i+1
        o = frames[i][:3, 3]     # a point on that axis
        Jv[:, i] = np.cross(z, p - o)
    return Jv


def compute_jacobian_dot(q, dq):
    """
    Time derivative of the linear geometric Jacobian.

    q:  numpy array of 7 joint angles (rad).
    dq: numpy array of 7 joint velocities (rad/s).

    Returns a 3x7 numpy array. Analytical or numerical,
    your choice.

    Analytical. Differentiate each column Jv_i = z x (p - o) with the
    product rule (z = z_(i-1), o = o_(i-1)):

        d/dt Jv_i = z_dot x (p - o) + z x (p_dot - o_dot)

    z is fixed in frame i-1, so it turns with that frame's angular velocity
    w:  z_dot = w x z.  The other pieces are:
        w     = sum over joints j before i of  qd_j * z_(j-1)
        p_dot = Jv @ dq                       (hand velocity)
        o_dot = velocity of the point o       (built up link by link)

    Both w and o_dot are built recursively, going out from the base:
        w_k     = w_(k-1) + qd_k * z_(k-1)
        o_dot_k = o_dot_(k-1) + w_k x (o_k - o_(k-1))

    Checked against central finite differences of compute_jacobian in
    verify/verify_trajectory.py.

    Used for acceleration kinematics:  p_ddot = Jv @ ddq + Jv_dot @ dq.
    """
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    frames, T_hand = _fk_frames(q)
    p = T_hand[:3, 3]
    p_dot = compute_jacobian(q) @ dq

    Jv_dot = np.zeros((3, 7))
    w = np.zeros(3)        # angular velocity of frame i (the base does not move)
    o_dot = np.zeros(3)    # linear velocity of the origin o_i
    for i in range(7):
        z = frames[i][:3, 2]
        o = frames[i][:3, 3]
        z_dot = np.cross(w, z)
        Jv_dot[:, i] = np.cross(z_dot, p - o) + np.cross(z, p_dot - o_dot)
        # Move on to frame i+1: joint i+1 adds its spin about z_i.
        w = w + dq[i] * z
        o_dot = o_dot + np.cross(w, frames[i + 1][:3, 3] - o)
    return Jv_dot


def _quintic_profile(s):
    """
    Normalised quintic h(s), s in [0, 1], and its first two derivatives in s.

    h(0) = 0, h(1) = 1, and h' = h'' = 0 at both ends. These are the six
    conditions that fix the six coefficients of a 5th-order polynomial:
        h(s)   = 10 s^3 - 15 s^4 + 6 s^5
        h'(s)  = 30 s^2 (1 - s)^2           >= 0, so h only goes up
        h''(s) = 60 s (1 - s)(1 - 2 s)
    """
    h = 10 * s**3 - 15 * s**4 + 6 * s**5
    hd = 30 * s**2 * (1 - s)**2
    hdd = 60 * s * (1 - s) * (1 - 2 * s)
    return h, hd, hdd


# Peak values of |h'| and |h''| on [0, 1]:
#   h'  peaks at s = 1/2:             30/16       = 1.875
#   h'' peaks at s = 1/2 -+ sqrt(3)/6: 10/sqrt(3) = 5.7735
QUINTIC_PEAK_VEL = 1.875
QUINTIC_PEAK_ACC = 10.0 / np.sqrt(3.0)


def minimum_duration(q_start, q_end):
    """
    Shortest T for which the quintic keeps every joint within LIMIT_FRACTION
    of its velocity and acceleration limit. With D = |q_end - q_start| per joint:
        peak velocity     = 1.875  * D / T     ->  T >= 1.875 D / v_max
        peak acceleration = 5.7735 * D / T^2   ->  T >= sqrt(5.7735 D / a_max)
    """
    D = np.abs(np.asarray(q_end, float) - np.asarray(q_start, float))
    T_vel = np.max(QUINTIC_PEAK_VEL * D / (LIMIT_FRACTION * QD_MAX))
    T_acc = np.max(np.sqrt(QUINTIC_PEAK_ACC * D / (LIMIT_FRACTION * QDD_MAX)))
    return max(T_vel, T_acc)


def plan_trajectory(q_start, q_end, T, dt):
    """
    Generate a trajectory from q_start to q_end over time T.

    Method is up to you. See the guide.

    q_start: numpy array of 7 joint angles (start).
    q_end:   numpy array of 7 joint angles (end).
    T:       total time of the trajectory (seconds).
    dt:      time step between samples (seconds).

    Returns:
        t_array:  1D array of times, length N.
        q_array:  (N, 7) array of joint positions.
        dq_array: (N, 7) array of joint velocities.
        ddq_array:(N, 7) array of joint accelerations.

    Method: quintic polynomial in joint space, all joints on the same clock.

        q(t) = q_start + h(t/T) * (q_end - q_start)

    - Boundary conditions: q(0) = q_start, q(T) = q_end, and the velocity
      and acceleration are zero at both ends. So the arm starts and stops
      without a jump in velocity or acceleration (a cubic would jump in
      acceleration at t = 0 and t = T).
    - All joints share one h(s), so they start and finish together and the
      motion is a straight line in joint space.
    - No overshoot: h only goes up (h' >= 0), so every joint moves one way
      only and always stays between its start and end value. If both ends
      are inside the joint limits, the whole path is too.
    - Velocity and acceleration limits: the peaks are known in closed form
      (see minimum_duration). If T is too short for them, T is stretched to
      the shortest safe duration, rather than sending a motion the robot
      cannot follow.
    - No Jacobian inverse is used, so a singularity on the way cannot make
      the joint velocities blow up.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)

    for name, q in (("q_start", q_start), ("q_end", q_end)):
        if np.any(q < Q_MIN) or np.any(q > Q_MAX):
            raise ValueError(f"{name} is outside the joint limits: {q}")

    T_safe = minimum_duration(q_start, q_end)
    if T < T_safe:
        print(f"[plan_trajectory] T = {T:.2f} s would exceed {LIMIT_FRACTION:.0%} of a "
              f"velocity or acceleration limit; stretched to {T_safe:.2f} s.")
        T = T_safe

    # Whole number of steps, so the last sample lands exactly on q_end.
    n_steps = int(np.ceil(T / dt - 1e-9))
    T = n_steps * dt
    t_array = np.arange(n_steps + 1) * dt

    s = t_array / T
    h, hd, hdd = _quintic_profile(s)
    delta = q_end - q_start
    q_array = q_start + np.outer(h, delta)
    dq_array = np.outer(hd / T, delta)          # chain rule: d/dt = (1/T) d/ds
    ddq_array = np.outer(hdd / T**2, delta)
    return t_array, q_array, dq_array, ddq_array


# ==========================================================
# INFRASTRUCTURE — DO NOT MODIFY BELOW
# ==========================================================

CMD_SIZE = 9
GRIPPER_CTRL = 255.0

# UPDATE THESE BASED ON YOUR ASSIGNED VARIANT
# You must choose reachable configurations within joint limits.
#
# Team 5, The Folded Approach.
# Start: arm stretched out in front (elbow almost straight, q4 = -0.5),
#        hand at about (0.72, 0.22, 0.50) m, gripper pointing down.
# End:   arm folded (elbow bent to q4 = -2.95, 0.12 rad from its limit),
#        hand pulled back to (0.25, 0, 0.40) m, 0.25 m from the base axis, gripper still
#        pointing down. The tightest gap at the end is upper arm (link2) to
#        forearm (link4); verify/verify_trajectory.py checks every sample of
#        the motion for self-collision with the MuJoCo collision meshes.
Q_START = np.array([0.3,  0.9, 0.0, -0.5,  0.0, 1.4,  0.785])
Q_END   = np.array([0.0, -1.0, 0.0, -2.95, 0.0, 1.95, 0.785])

TRAJ_DURATION = 4.0
TRAJ_DT       = 0.05


class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_node')
        self.q_current = np.zeros(7)
        self.dq_current = np.zeros(7)

        self.create_subscription(
            JointState, 'joint_state', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, 'joint_commands', 10)

        self.t_array = None
        self.q_traj = None
        self.dq_traj = None
        self.ddq_traj = None
        self.step_idx = 0

        # Wait 0.5s before starting to ensure the sim is up
        self.create_timer(0.5, self.start_trajectory_once)

    def on_joint_state(self, msg):
        self.q_current = np.array(msg.position[:7])
        if hasattr(msg, 'velocity') and len(msg.velocity) >= 7:
            self.dq_current = np.array(msg.velocity[:7])

    def start_trajectory_once(self):
        if self.t_array is not None:
            return

        result = plan_trajectory(Q_START, Q_END, TRAJ_DURATION, TRAJ_DT)
        if result is None:
            return

        t_arr, q_arr, dq_arr, ddq_arr = result

        if q_arr.shape[1] != 7 or dq_arr.shape[1] != 7 or ddq_arr.shape[1] != 7:
            self.get_logger().error(
                "plan_trajectory returned wrong shape. Expected (N, 7).")
            return

        self.t_array = t_arr
        self.q_traj = q_arr
        self.dq_traj = dq_arr
        self.ddq_traj = ddq_arr

        self.get_logger().info(
            f"Trajectory generated: {len(t_arr)} samples over "
            f"{TRAJ_DURATION:.1f} s at {1.0/TRAJ_DT:.0f} Hz.")

        self.create_timer(TRAJ_DT, self.tick)

    def tick(self):
        if self.step_idx >= len(self.t_array):
            # Hold final target
            cmd = Float64MultiArray()
            
            # ==================================================
            # DEPENDING ON YOUR BRIDGE MODE, UNCOMMENT ONE BELOW:
            # ==================================================
            
            # -- POSITION MODE (Default) --
            cmd.data = Q_END.tolist() + [GRIPPER_CTRL, 0.0]
            
            # -- VELOCITY MODE --
            # cmd.data = [0.0] * 7 + [GRIPPER_CTRL, 0.0]
            
            self.cmd_pub.publish(cmd)
            return

        cmd = Float64MultiArray()
        
        # ==================================================
        # DEPENDING ON YOUR BRIDGE MODE, UNCOMMENT ONE BELOW:
        # ==================================================
        
        # -- POSITION MODE (Default) --
        q_target = self.q_traj[self.step_idx]
        
        # -- VELOCITY MODE --
        # q_target = self.dq_traj[self.step_idx]
        
        cmd.data = q_target.tolist() + [GRIPPER_CTRL, 0.0]
        self.cmd_pub.publish(cmd)

        self.step_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
