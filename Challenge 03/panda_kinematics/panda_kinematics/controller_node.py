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
# Kinematics, copied from Challenges 1 and 2 (DH table, frames).
# ---------------------------------------------------------------

# Standard DH, one row per link: [alpha, a, d]. theta is the joint variable.
# A_i = Rz(theta_i) . Tz(d_i) . Tx(a_i) . Rx(alpha_i); joint i turns about z_(i-1).
DH_PARAMS = [
    [-np.pi / 2,  0.0,    0.333],  # link 1
    [ np.pi / 2,  0.0,    0.0  ],  # link 2
    [ np.pi / 2,  0.0825, 0.316],  # link 3
    [-np.pi / 2, -0.0825, 0.0  ],  # link 4
    [ np.pi / 2,  0.0,    0.384],  # link 5
    [ np.pi / 2,  0.088,  0.0  ],  # link 6
    [ 0.0,        0.0,    0.0  ],  # link 7
]

# link7 -> hand: 0.107 m along z, then -45 deg about z (panda.xml).
_c, _s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
T_L7_HAND = np.array([
    [_c,  -_s,  0.0, 0.0  ],
    [_s,   _c,  0.0, 0.0  ],
    [0.0,  0.0, 1.0, 0.107],
    [0.0,  0.0, 0.0, 1.0  ],
])

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
QD_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
LIMIT_FRACTION = 0.8
TORQUE_LIMIT = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])   # forcerange in panda.xml [N m]


def dh_transform(alpha, a, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def fk_frames(q):
    """DH frames 0..7 in the base (link0) frame."""
    T = np.eye(4)
    frames = [T]
    for (alpha, a, d), theta in zip(DH_PARAMS, q):
        T = T @ dh_transform(alpha, a, d, theta)
        frames.append(T)
    return frames


def forward_kinematics(q):
    """Pose of the 'hand' frame in the base frame."""
    return fk_frames(np.asarray(q, dtype=float))[7] @ T_L7_HAND


# ---------------------------------------------------------------
# Challenge 3 (a): dynamics by the Lagrangian method.
# ---------------------------------------------------------------
#
# Link parameters from the <inertial> tags of panda.xml:
#   mass [kg], CoM in the MuJoCo link frame [m], fullinertia = Ixx Iyy Izz Ixy Ixz Iyz [kg m^2].
# R_DH, P_DH: the MuJoCo link frame seen from our DH frame of the same link. Both frames
# are fixed to the same link, so this is a constant rotation + shift (read off the XML
# body chain and our DH table).
_X = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]     # MuJoCo link frame = DH frame turned +90 deg about x
_Y = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]     # ... turned -90 deg about x
_XML_LINKS = [
    dict(mass=4.970684, com=[0.003875, 0.002081, -0.04762],
         I=[0.70337, 0.70661, 0.0091170, -0.00013900, 0.0067720, 0.019169], R_DH=_X, P_DH=[0, 0, 0]),
    dict(mass=0.646926, com=[-0.003141, -0.02872, 0.003495],
         I=[0.0079620, 2.8110e-2, 2.5995e-2, -3.925e-3, 1.0254e-2, 7.04e-4], R_DH=_Y, P_DH=[0, 0, 0]),
    dict(mass=3.228604, com=[2.7518e-2, 3.9252e-2, -6.6502e-2],
         I=[3.7242e-2, 3.6155e-2, 1.083e-2, -4.761e-3, -1.1396e-2, -1.2805e-2], R_DH=_Y, P_DH=[-0.0825, 0, 0]),
    dict(mass=3.587895, com=[-5.317e-2, 1.04419e-1, 2.7454e-2],
         I=[2.5853e-2, 1.9552e-2, 2.8323e-2, 7.796e-3, -1.332e-3, 8.641e-3], R_DH=_X, P_DH=[0.0825, 0, 0]),
    dict(mass=1.225946, com=[-1.1953e-2, 4.1065e-2, -3.8437e-2],
         I=[3.5549e-2, 2.9474e-2, 8.627e-3, -2.117e-3, -4.037e-3, 2.29e-4], R_DH=_Y, P_DH=[0, 0, 0]),
    dict(mass=1.666555, com=[6.0149e-2, -1.4117e-2, -1.0517e-2],
         I=[1.964e-3, 4.354e-3, 5.433e-3, 1.09e-4, -1.158e-3, 3.41e-4], R_DH=_Y, P_DH=[-0.088, 0, 0]),
    dict(mass=7.35522e-01, com=[1.0517e-2, -4.252e-3, 6.1597e-2],
         I=[1.2516e-2, 1.0027e-2, 4.815e-3, -4.28e-4, -1.196e-3, -7.41e-4], R_DH=np.eye(3), P_DH=[0, 0, 0]),
]
ARMATURE = 0.1          # rotor inertia at every joint [kg m^2] (panda.xml)
JOINT_DAMPING = 1.0     # viscous friction at every joint [N m s/rad] (panda.xml); NOT part of M, C, g
GRAVITY = np.array([0.0, 0.0, -9.81])


def _full_inertia(I):
    xx, yy, zz, xy, xz, yz = I
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _combine(bodies):
    """Rigid bodies bolted together -> one body: mass, CoM, inertia about the CoM (parallel axis)."""
    m = sum(b[0] for b in bodies)
    c = sum(b[0] * b[1] for b in bodies) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in bodies:
        r = ci - c
        I += Ii + mi * (r @ r * np.eye(3) - np.outer(r, r))
    return m, c, I


def _build_links(mass_scale=None):
    """(mass, CoM, inertia) of links 1..7 in our DH frames. The hand and both fingers
    (fingers open, 4 cm each) are bolted onto link 7 as one rigid body."""
    links = []
    for i, L in enumerate(_XML_LINKS):
        R, p = np.array(L["R_DH"], dtype=float), np.array(L["P_DH"], dtype=float)
        k = 1.0 if mass_scale is None else mass_scale[i]
        links.append((k * L["mass"], R @ np.array(L["com"]) + p, k * R @ _full_inertia(L["I"]) @ R.T))
    # hand frame in the link-7 frame is T_L7_HAND; fingers sit 0.0584 m along the hand's z, 4 cm to each side.
    Rh, ph = T_L7_HAND[:3, :3], T_L7_HAND[:3, 3]
    hand = (0.73, Rh @ np.array([-0.01, 0.0, 0.03]) + ph, Rh @ np.diag([0.001, 0.0025, 0.0017]) @ Rh.T)
    finger_I = Rh @ np.diag([2.375e-6, 2.375e-6, 7.5e-7]) @ Rh.T
    left = (0.015, Rh @ np.array([0.0, 0.04, 0.0584]) + ph, finger_I)
    right = (0.015, Rh @ np.array([0.0, -0.04, 0.0584]) + ph, finger_I)
    links[6] = _combine([links[6], hand, left, right])
    return links


LINKS = _build_links()


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _geometry(q, links=LINKS):
    """Joint axes z_j and points o_j (j = 0..6), and for each link: mass, CoM, inertia in the base frame."""
    F = fk_frames(np.asarray(q, dtype=float))
    Z = np.array([F[j][:3, 2] for j in range(7)])
    O = np.array([F[j][:3, 3] for j in range(7)])
    world = []
    for i, (m, c, I) in enumerate(links):
        R = F[i + 1][:3, :3]                      # link i+1 is fixed to DH frame i+1
        world.append((m, R @ c + F[i + 1][:3, 3], R @ I @ R.T))
    return Z, O, world


def _link_jacobians(Z, O, i, c):
    """Jv, Jw (3x7) of the CoM c of link i: only joints 1..i+1 move it."""
    Jv, Jw = np.zeros((3, 7)), np.zeros((3, 7))
    for j in range(i + 1):
        Jv[:, j] = np.cross(Z[j], c - O[j])
        Jw[:, j] = Z[j]
    return Jv, Jw


def mass_matrix(q, links=LINKS):
    """M(q) from the kinetic energy K = 1/2 qd^T M qd:
    M = sum_i ( m_i Jv_i^T Jv_i + Jw_i^T (R_i I_i R_i^T) Jw_i ) + armature."""
    Z, O, world = _geometry(q, links)
    M = ARMATURE * np.eye(7)
    for i, (m, c, Iw) in enumerate(world):
        Jv, Jw = _link_jacobians(Z, O, i, c)
        M += m * Jv.T @ Jv + Jw.T @ Iw @ Jw
    return M


def gravity_vector(q, links=LINKS):
    """g(q) = dP/dq with P = sum_i m_i g0 z_ci:  g = - sum_i m_i Jv_i^T g_vec."""
    Z, O, world = _geometry(q, links)
    g = np.zeros(7)
    for i, (m, c, _) in enumerate(world):
        Jv, _ = _link_jacobians(Z, O, i, c)
        g -= m * Jv.T @ GRAVITY
    return g


def mass_matrix_derivatives(q, links=LINKS):
    """
    dM/dq_k (k = 0..6), closed form. Turning joint k rigidly rotates everything after it
    about z_k:  dz_j/dq_k = z_k x z_j,  do_j/dq_k = z_k x (o_j - o_k)  (j > k),
    dc_i/dq_k = z_k x (c_i - o_k),  d(R I R^T)/dq_k = S I_w - I_w S  (S = [z_k]x).
    Then the product rule on m Jv^T Jv + Jw^T I_w Jw.
    """
    Z, O, world = _geometry(q, links)
    D = np.zeros((7, 7, 7))
    later = np.triu(np.ones((7, 7)), 1)[:, :, None]
    dz_all = np.cross(Z[:, None, :], Z[None, :, :]) * later
    do_all = np.cross(Z[:, None, :], O[None, :, :] - O[:, None, :]) * later
    for i, (m, c, Iw) in enumerate(world):
        n = i + 1
        Jv, Jw = _link_jacobians(Z, O, i, c)
        Zn, On = Z[:n], O[:n]
        dz, do = dz_all[:n, :n], do_all[:n, :n]
        dc = np.cross(Zn, c - On)
        dJv = np.transpose(np.cross(dz, c - On[None]) + np.cross(Zn[None], dc[:, None] - do), (0, 2, 1))
        dJw = np.transpose(dz, (0, 2, 1))
        S = np.array([_skew(z) for z in Zn])
        dIw = S @ Iw - Iw @ S
        Jvn, Jwn = Jv[:, :n], Jw[:, :n]
        A = m * np.transpose(dJv, (0, 2, 1)) @ Jvn + np.transpose(dJw, (0, 2, 1)) @ Iw @ Jwn
        D[:n, :n, :n] += A + np.transpose(A, (0, 2, 1)) + Jwn.T @ dIw @ Jwn
    return D


def coriolis_matrix(q, dq, links=LINKS):
    """C(q, qd) from the Christoffel symbols: C_kj = 1/2 sum_i (dM_kj/dq_i + dM_ki/dq_j - dM_ij/dq_k) qd_i."""
    D = mass_matrix_derivatives(q, links)
    dq = np.asarray(dq, dtype=float)
    return 0.5 * (np.einsum("ikj,i->kj", D, dq) + np.einsum("jki,i->kj", D, dq) - np.einsum("kij,i->kj", D, dq))


def compute_dynamics(q, dq, ddq=None):
    """
    Lagrangian dynamics of the Panda (7 links + hand).

    q, dq: numpy arrays of 7 joint angles (rad) and velocities (rad/s).
    ddq:   joint accelerations. The template passes them, but M and C q_dot + g do not
           depend on them (the torque is M @ ddq + h), so they are not used here.

    Returns:
        M: (7, 7) mass / inertia matrix M(q)
        h: (7,)   C(q, qd) qd + g(q)  (Coriolis / centrifugal + gravity)

    so that  tau = M @ ddq + h.  The joint damping of panda.xml (1 N m s/rad) is
    friction, not part of the Lagrangian; the controller has to deal with it.
    Checked against MuJoCo in verify/verify_dynamics.py.
    """
    q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
    M = mass_matrix(q)
    h = coriolis_matrix(q, dq) @ dq + gravity_vector(q)
    return M, h


# ---------------------------------------------------------------
# Challenge 3 (b): the fuzzy logic controller (FLC).
# ---------------------------------------------------------------
#
# Decoupled MIMO: 7 FLCs, one per joint, all with the same membership functions and
# rule base, each with its own scaling gains.
#   inputs:  E  = error / GE                 (normalised to [-1, 1], clipped)
#            DE = (delta_error / dt) / GDE   (rate of change of the error, normalised)
#   output:  u in [-1, 1], torque correction = GU * u
# Five fuzzy sets on every normalised universe:
#   LN large negative, MN medium negative, S small, MP medium positive, LP large positive
# Triangles centred at -1, -0.5, 0, 0.5, 1, each reaching zero at its neighbours'
# centres, so any value belongs to at most two sets and the memberships add up to 1.
# On the inputs LN and LP are shoulders (stay at 1 beyond -1 / +1: "large" stays large).
# On the output the universe runs to +-1.5 so LN and LP are full triangles and the
# centroid can actually reach +-1.
SETS = ("LN", "MN", "S", "MP", "LP")
CENTERS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
WIDTH = 0.5
OUT_UNIVERSE = np.linspace(-1.5, 1.5, 301)

# Rule base, 5 x 5 = 25 rules: RULES[i][j] = output set for (E is SETS[i]) AND (DE is SETS[j]).
# A PD-like table: the output grows with the error and with its rate, and the two can
# cancel (error big but already shrinking fast -> push less, to avoid overshoot).
#            DE: LN    MN    S     MP    LP
RULES = [["LN", "LN", "LN", "MN", "S"],    # E = LN
         ["LN", "LN", "MN", "S",  "MP"],   # E = MN
         ["LN", "MN", "S",  "MP", "LP"],   # E = S
         ["MN", "S",  "MP", "LP", "LP"],   # E = MP
         ["S",  "MP", "LP", "LP", "LP"]]   # E = LP
_RULE_IDX = np.array([[SETS.index(r) for r in row] for row in RULES])


def membership(x, shoulders=True):
    """Membership of each value in x (shape (n,)) in the 5 sets -> (n, 5)."""
    x = np.asarray(x, dtype=float)[:, None]
    mu = np.clip(1.0 - np.abs(x - CENTERS) / WIDTH, 0.0, 1.0)
    if shoulders:
        mu[:, 0] = np.where(x[:, 0] <= -1.0, 1.0, mu[:, 0])
        mu[:, 4] = np.where(x[:, 0] >= 1.0, 1.0, mu[:, 4])
    return mu


_OUT_MU = membership(OUT_UNIVERSE, shoulders=False)          # (301, 5): output sets on the universe


# Inference method.
#   "sum-product": AND = product, implication = scale the output set, aggregation = sum.
#   "min-max":     AND = min, implication = cut the output set, aggregation = max (classic Mamdani).
# Both end with the centroid of the aggregated shape. We use sum-product: with our
# overlapping triangles it makes the control surface a smooth interpolation of the rule
# table, with slope exactly 1 near zero. Min-max made the FLC up to 1.8 times stiffer near
# zero error than the linear PD it stands for (u(0.02, 0.02) = 0.073 instead of 0.04), and with the ROS loop
# delay that extra gain made the torque chatter (see verify/flc_study.py).
INFERENCE = "sum-product"


def fuzzy_inference(E, DE, method=None):
    """
    Fuzzy inference for n FLCs at once (normalised inputs, shape (n,)).
      1. fuzzify: membership of E and of DE in the 5 sets
      2. each of the 25 rules fires with strength  mu_E AND mu_DE
      3. each output set is weighted by the rules that point to it
      4. defuzzify with the centroid of area:  u = sum(mu(y) y) / sum(mu(y))  over the universe
    Returns u (n,) in [-1, 1].
    """
    method = method or INFERENCE
    muE, muD = membership(E), membership(DE)
    if method == "min-max":
        w = np.minimum(muE[:, :, None], muD[:, None, :])           # (n, 5, 5) rule strengths
        strength = np.zeros((len(E), 5))
        for k in range(5):
            strength[:, k] = np.where(_RULE_IDX == k, w, 0.0).max(axis=(1, 2))
        agg = np.max(np.minimum(strength[:, None, :], _OUT_MU[None, :, :]), axis=2)   # (n, 301)
    else:
        w = muE[:, :, None] * muD[:, None, :]
        strength = np.zeros((len(E), 5))
        for k in range(5):
            strength[:, k] = np.where(_RULE_IDX == k, w, 0.0).sum(axis=(1, 2))
        agg = strength @ _OUT_MU.T                                  # (n, 301): sum of scaled sets
    return (agg @ OUT_UNIVERSE) / np.maximum(agg.sum(axis=1), 1e-12)


# Scaling: how big an error counts as "large" (GE), how fast a change counts as "large" (GDE),
# and the largest output (GU), per joint.
#
# Near zero the normalised FLC is u = E + DE (slope exactly 1 with sum-product inference),
# so each joint's FLC acts like a PD with Kp = GU/GE and Kd = GU/GDE. We pick that equivalent
# stiffness and damping like a critically damped joint with natural frequency wn:
#     Kp = I wn^2,  Kd = 2 I wn   ->   GE = GU / (I wn^2),  GDE = GU / (2 I wn)
# where I is the inertia the FLC output acts on:
#   - MODEL_FLC: the FLC output is an ACCELERATION and the model turns it into torque,
#     tau = M(q) (q_ddot_des + u) + C q_dot + g. M(q) takes care of the coupling between joints,
#     so every joint looks like a plain double integrator: I = 1, the same scaling everywhere.
#   - GRAVITY_FLC / FLC_ONLY: the FLC output is a TORQUE and there is no M(q). A joint then
#     feels its "effective" inertia 1/(M^-1)_ii (the other joints are free to move, not held),
#     which along our motion is 3 to 14 times smaller than M_ii. We scale with the smallest
#     value on the path, so no joint is over-driven. First we used M_ii, and joint 1 oscillated
#     at 23 Hz between the torque limits: coupled to joint 3, it is much lighter than M_11 says.
I_EFF_MIN = np.array([0.25, 0.85, 0.17, 0.25, 0.13, 0.13, 0.11])   # min of 1/(M^-1)_ii on the path [kg m^2]


def flc_scaling(GU, wn, inertia=1.0):
    GU = np.asarray(GU, dtype=float)
    return dict(GE=GU / (inertia * wn**2), GDE=GU / (2 * inertia * wn), GU=GU)


# wn is limited by the loop rate. The template runs the loop at 100 Hz (TRAJ_DT = 0.01 s), and
# through ROS 2 the torque reaches the arm about one period late. A delay of d seconds eats phase,
# so wn has to stay well below 1/d. Our first tuning was for a 250 Hz loop (wn = 40 and 30);
# at 100 Hz those values oscillated (model + FLC: 10.7 deg error offline). We swept wn offline
# with 1 and 2 periods of delay and took the largest value that stays calm with 2 periods:
FLC_GAINS = {
    # Model + FLC: corrective acceleration up to 30 rad/s^2 on every joint, wn = 20 rad/s
    # (25 oscillated with 2 periods of delay).
    "MODEL_FLC":   flc_scaling(GU=np.full(7, 30.0), wn=20.0),
    # Team 5, g(q) + FLC: the FLC alone drives the motion, so it must also make the inertia and
    # Coriolis torques. Output in N m. wn = 15 rad/s (17 already started to chatter).
    "GRAVITY_FLC": flc_scaling(GU=[40, 40, 20, 30, 6, 6, 6], wn=15.0, inertia=I_EFF_MIN),
    # FLC with no model at all (for the debate): it must also hold the arm against gravity,
    # so it gets the full torque range.
    "FLC_ONLY":    flc_scaling(GU=[87, 87, 87, 87, 12, 12, 12], wn=15.0, inertia=I_EFF_MIN),
}

# Which law compute_torques() uses:
#   "MODEL_FLC"    tau = M q_ddot_des + C q_dot + g + FLC       (the main design, default)
#   "MODEL_ONLY"   tau = M q_ddot_des + C q_dot + g             (the FLC removed)
#   "GRAVITY_FLC"  tau = g(q) + FLC                              (Team 5 variant)
#   "FLC_ONLY"     tau = FLC                                     (no model, for comparison)
CONTROL_MODE = "MODEL_FLC"

_prev_error = None


def fuzzy_controller(error, delta_error, dt):
    """
    error:       (7,) q_des - q  [rad]
    delta_error: (7,) error now - error at the previous step [rad]
    dt:          time between the two steps [s]

    Returns (7,) control signals, one FLC per joint: a corrective acceleration [rad/s^2] in
    MODEL_FLC mode (the model turns it into torque), a correction torque [N m] otherwise.
    """
    gains = FLC_GAINS[CONTROL_MODE if CONTROL_MODE in FLC_GAINS else "MODEL_FLC"]
    E = np.clip(np.asarray(error) / gains["GE"], -1.0, 1.0)
    DE = np.clip(np.asarray(delta_error) / dt / gains["GDE"], -1.0, 1.0)
    return gains["GU"] * fuzzy_inference(E, DE)


def compute_torques(q, dq, q_des, dq_des, ddq_des):
    """
    Decide the 7 joint torques for this control step.

    The model does the heavy lifting (it knows the torque the planned motion needs);
    the FLC fills the gaps (friction, model errors, disturbances).
    """
    global _prev_error
    dt = TRAJ_DT          # the template calls us once per TRAJ_DT (0.01 s, 100 Hz)
    q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
    error = np.asarray(q_des) - q
    delta_error = np.zeros(7) if _prev_error is None else error - _prev_error
    _prev_error = error

    if CONTROL_MODE == "GRAVITY_FLC":
        tau = gravity_vector(q) + fuzzy_controller(error, delta_error, dt)
    elif CONTROL_MODE == "FLC_ONLY":
        tau = fuzzy_controller(error, delta_error, dt)
    else:
        M, h = compute_dynamics(q, dq)
        ddq = np.asarray(ddq_des, dtype=float)
        if CONTROL_MODE == "MODEL_FLC":
            ddq = ddq + fuzzy_controller(error, delta_error, dt)   # FLC: corrective acceleration
        tau = M @ ddq + h            # model torque + correction torque M u
    return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


# ---------------------------------------------------------------
# Trajectory: the Challenge 2 quintic (Team 5, the folded approach).
# ---------------------------------------------------------------

def _quintic_profile(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 * (1 - s)**2, 60 * s * (1 - s) * (1 - 2 * s)


def plan_trajectory(q_start, q_end, T, dt):
    """
    Quintic in joint space, all joints on one clock, from Challenge 2: zero velocity and
    acceleration at both ends. T is stretched if it would break LIMIT_FRACTION of a limit.
    Returns t (N,), q (N, 7), dq (N, 7), ddq (N, 7).
    """
    q_start, q_end = np.asarray(q_start, dtype=float), np.asarray(q_end, dtype=float)
    for name, q in (("q_start", q_start), ("q_end", q_end)):
        if np.any(q < Q_MIN) or np.any(q > Q_MAX):
            raise ValueError(f"{name} is outside the joint limits: {q}")
    D = np.abs(q_end - q_start)
    T = max(T, np.max(1.875 * D / (LIMIT_FRACTION * QD_MAX)),
            np.max(np.sqrt(5.7735 * D / (LIMIT_FRACTION * QDD_MAX))))
    n = int(np.ceil(T / dt - 1e-9))
    T = n * dt
    t = np.arange(n + 1) * dt
    h, hd, hdd = _quintic_profile(t / T)
    delta = q_end - q_start
    return t, q_start + np.outer(h, delta), np.outer(hd / T, delta), np.outer(hdd / T**2, delta)


# ==========================================================
# DO NOT MODIFY BELOW
# ==========================================================

CMD_SIZE = 9
GRIPPER_CTRL = 255.0
TRAJ_DURATION = 4.0
TRAJ_DT = 0.01

# Team-specific start and end configurations.
# Team 5, the folded approach (Challenge 2): stretched out in front -> folded back.
Q_START = np.array([0.3,  0.9, 0.0, -0.5,  0.0, 1.4,  0.785])
Q_END   = np.array([0.0, -1.0, 0.0, -2.95, 0.0, 1.95, 0.785])

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        self.q_current = np.zeros(7)
        self.dq_current = np.zeros(7)
        
        self.q_traj = None
        self.dq_traj = None
        self.ddq_traj = None
        self.step_idx = 0

        self.create_subscription(JointState, 'joint_state', self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, 'joint_commands', 10)

        self.create_timer(0.5, self.start_trajectory_once)

    def on_joint_state(self, msg):
        self.q_current = np.array(msg.position[:7])
        if hasattr(msg, 'velocity') and len(msg.velocity) >= 7:
            self.dq_current = np.array(msg.velocity[:7])

    def start_trajectory_once(self):
        if self.q_traj is not None:
            return

        result = plan_trajectory(Q_START, Q_END, TRAJ_DURATION, TRAJ_DT)
        if result is None:
            return

        _, self.q_traj, self.dq_traj, self.ddq_traj = result
        self.get_logger().info("Trajectory generated. Starting Controller.")
        
        # Switch to the control loop timer
        # Fix to the template (the only change below the line): in ROS 2 Humble, Node.timers is a
        # generator, so self.timers[0] raises "TypeError: 'generator' object is not subscriptable"
        # and the node dies when the trajectory starts. list(...) makes it a list first.
        self.destroy_timer(list(self.timers)[0])
        self.create_timer(TRAJ_DT, self.control_loop)

    def control_loop(self):
        if self.step_idx >= len(self.q_traj):
            # Hold final position by sending zero velocity and holding torque
            cmd = Float64MultiArray()
            M, Cg = compute_dynamics(self.q_current, np.zeros(7), np.zeros(7))
            cmd.data = Cg.tolist() + [GRIPPER_CTRL, 0.0]
            self.cmd_pub.publish(cmd)
            return

        q_des = self.q_traj[self.step_idx]
        dq_des = self.dq_traj[self.step_idx]
        ddq_des = self.ddq_traj[self.step_idx]

        torques = compute_torques(
            self.q_current, self.dq_current, 
            q_des, dq_des, ddq_des
        )

        cmd = Float64MultiArray()
        cmd.data = torques.tolist() + [GRIPPER_CTRL, 0.0]
        self.cmd_pub.publish(cmd)

        self.step_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
