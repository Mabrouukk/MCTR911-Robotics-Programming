"""
Velocity and acceleration kinematics of the UR5e (Milestone 3).

Builds on the DH table and forward kinematics of Milestone 2 (kinematics.py).
The end effector is the tool flange ("attachment_site" in the MuJoCo model),
the same point Milestone 2's FK returns. All vectors are in the robot base
frame {0}. In the cell the base sits at ROBOT_BASE_POS, a pure translation,
so velocities and accelerations are the same in the world frame.

    forward velocity      x_dot  = J(q) q_dot                       (twist from joint rates)
    inverse velocity      q_dot  = J(q)^-1 x_dot                    (joint rates from a twist)
    forward acceleration  x_ddot = J(q) q_ddot + J_dot(q, q_dot) q_dot
    inverse acceleration  q_ddot = J(q)^-1 (x_ddot - J_dot(q, q_dot) q_dot)

x_dot = [v; w] stacks the flange's linear velocity v (m/s) and angular
velocity w (rad/s). x_ddot = [a; alpha] is the same for accelerations.
"""
import numpy as np

from kinematics import A, ALPHA, D, forward_kinematics

# Below this smallest singular value of J (in m/rad), J is treated as near
# singular and the inverse switches to damped least squares (see _solve).
SINGULAR_EPS = 0.02
DAMPING_MAX = 0.02


# ---------------------------------------------------------------------------
# The Jacobian
# ---------------------------------------------------------------------------

def jacobian(q):
    """
    6x6 geometric Jacobian of the flange in the base frame, rows [v; w].

    Every UR5e joint is revolute. Joint i turns about z_(i-1), the z axis of
    DH frame i-1, which passes through that frame's origin o_(i-1). If only
    joint i moves, at rate q_dot_i:
        - the whole flange turns about that axis:      w = z_(i-1) q_dot_i
        - the flange point moves on a circle round it: v = z_(i-1) x (p - o_(i-1)) q_dot_i
    so column i of J is
        J_i = [ z_(i-1) x (p - o_(i-1)) ;  z_(i-1) ]
    z_(i-1) and o_(i-1) are the 3rd and 4th columns of T_0^(i-1) = A_1 ... A_(i-1),
    which forward_kinematics already builds.
    """
    frames = forward_kinematics(np.asarray(q, dtype=float), return_all=True)
    p = frames[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z = frames[i][:3, 2]
        o = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def jacobian_dot(q, dq):
    """
    Time derivative of the geometric Jacobian, 6x6, analytical.

    Differentiate each column with the product rule (z = z_(i-1), o = o_(i-1)):
        d/dt [z x (p - o)] = z_dot x (p - o) + z x (p_dot - o_dot)
        d/dt z             = z_dot
    z is fixed in frame i-1, so it turns with that frame's angular velocity:
        z_dot = w_(i-1) x z
    and p_dot is the flange velocity J q_dot. The angular velocity w_k and the
    origin velocity o_dot_k of each frame are built outwards from the base:
        w_k     = w_(k-1) + q_dot_k z_(k-1)
        o_dot_k = o_dot_(k-1) + w_k x (o_k - o_(k-1))
    """
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    frames = forward_kinematics(q, return_all=True)
    p = frames[6][:3, 3]
    p_dot = jacobian(q)[:3] @ dq

    J_dot = np.zeros((6, 6))
    w = np.zeros(3)        # angular velocity of frame i (base frame is still)
    o_dot = np.zeros(3)    # linear velocity of origin o_i
    for i in range(6):
        z = frames[i][:3, 2]
        o = frames[i][:3, 3]
        z_dot = np.cross(w, z)
        J_dot[:3, i] = np.cross(z_dot, p - o) + np.cross(z, p_dot - o_dot)
        J_dot[3:, i] = z_dot
        w = w + dq[i] * z
        o_dot = o_dot + np.cross(w, frames[i + 1][:3, 3] - o)
    return J_dot


# ---------------------------------------------------------------------------
# Singularities
# ---------------------------------------------------------------------------

def singular_values(q):
    return np.linalg.svd(jacobian(q), compute_uv=False)


def determinant_closed_form(q):
    """
    det J for the UR5e in closed form:

        det J = a2 a3 sin(q3) sin(q5) (a2 cos(q2) + a3 cos(q2 + q3) + d5 sin(q2 + q3 + q4))

    (checked against np.linalg.det in validate_kinematics.py). It is zero in
    exactly three cases, the three UR5e singularities:
      - elbow:    sin(q3) = 0, the arm is fully stretched or folded flat,
      - wrist:    sin(q5) = 0, the axes of joints 4 and 6 line up,
      - shoulder: a2 cos(q2) + a3 cos(q2+q3) + d5 sin(q2+q3+q4) = 0, the wrist
                  centre sits on the axis of joint 1.
    """
    q = np.asarray(q, dtype=float)
    a2, a3, d5 = A[1], A[2], D[4]
    shoulder = a2 * np.cos(q[1]) + a3 * np.cos(q[1] + q[2]) + d5 * np.sin(q[1] + q[2] + q[3])
    return a2 * a3 * np.sin(q[2]) * np.sin(q[4]) * shoulder


def singularity_report(q):
    """Distance from each of the three singularities, as the factors of det J."""
    q = np.asarray(q, dtype=float)
    a2, a3, d5 = A[1], A[2], D[4]
    return {
        "elbow  sin(q3)": float(np.sin(q[2])),
        "wrist  sin(q5)": float(np.sin(q[4])),
        "shoulder (m)": float(a2 * np.cos(q[1]) + a3 * np.cos(q[1] + q[2]) + d5 * np.sin(q[1] + q[2] + q[3])),
        "sigma_min": float(singular_values(q)[-1]),
    }


def _solve(J, x):
    """
    Solve J y = x for y.

    Away from singularities J is invertible and this is the exact inverse.
    When the smallest singular value drops below SINGULAR_EPS, the plain
    inverse would ask for huge joint rates (they grow like 1/sigma_min), so we
    switch to damped least squares,
        y = J^T (J J^T + lambda^2 I)^-1 x,
    with lambda growing smoothly from 0 at sigma_min = SINGULAR_EPS to
    DAMPING_MAX at the singularity. Along the lost direction the result then
    goes to zero instead of infinity; every other direction stays exact.
    Returns (y, lambda) so the caller can see when damping was used.
    """
    sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
    if sigma_min >= SINGULAR_EPS:
        return np.linalg.solve(J, x), 0.0
    lam = DAMPING_MAX * np.sqrt(1.0 - (sigma_min / SINGULAR_EPS) ** 2)
    return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), x), lam


# ---------------------------------------------------------------------------
# Forward and inverse velocity kinematics
# ---------------------------------------------------------------------------

def forward_velocity(q, dq):
    """Flange twist from joint rates. Returns (v [m/s], w [rad/s]) in the base frame."""
    twist = jacobian(q) @ np.asarray(dq, dtype=float)
    return twist[:3], twist[3:]


def inverse_velocity(q, v, w, return_damping=False):
    """Joint rates (rad/s) that give the flange linear velocity v and angular velocity w."""
    dq, lam = _solve(jacobian(q), np.concatenate([v, w]))
    return (dq, lam) if return_damping else dq


# ---------------------------------------------------------------------------
# Forward and inverse acceleration kinematics
# ---------------------------------------------------------------------------

def forward_acceleration(q, dq, ddq):
    """
    Flange acceleration from joint accelerations. Returns (a [m/s^2], alpha [rad/s^2]).

    Differentiating x_dot = J q_dot in time gives x_ddot = J q_ddot + J_dot q_dot.
    The second term is the acceleration the flange has even when every joint
    turns at constant speed (centripetal and Coriolis-like terms).
    """
    acc = jacobian(q) @ np.asarray(ddq, dtype=float) + jacobian_dot(q, dq) @ np.asarray(dq, dtype=float)
    return acc[:3], acc[3:]


def inverse_acceleration(q, dq, a, alpha, return_damping=False):
    """Joint accelerations (rad/s^2) that give the flange accelerations a and alpha."""
    rhs = np.concatenate([a, alpha]) - jacobian_dot(q, dq) @ np.asarray(dq, dtype=float)
    ddq, lam = _solve(jacobian(q), rhs)
    return (ddq, lam) if return_damping else ddq


# ---------------------------------------------------------------------------
# Small helpers used by the validation scripts and nodes
# ---------------------------------------------------------------------------

def rotation_error(R_target, R):
    """Smallest rotation angle (rad) between two rotation matrices.

    atan2 of the sine and cosine parts stays accurate for tiny angles,
    where arccos of the trace alone loses half the digits."""
    E = R_target @ R.T
    s = 0.5 * np.linalg.norm([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    c = (np.trace(E) - 1.0) / 2.0
    return float(np.arctan2(s, c))


def quat_to_matrix(quat):
    """MuJoCo quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
