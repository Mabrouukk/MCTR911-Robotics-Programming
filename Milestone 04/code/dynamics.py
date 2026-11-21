"""
Dynamics of the UR5e with the Robotiq 2F-85 gripper, by the Lagrangian method (Milestone 4).

    M(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau

How we get there:
  1. Kinetic energy. Every link i has a mass m_i, a centre of mass c_i and an
     inertia I_i. Its CoM moves with v_ci = Jv_i(q) q_dot and it spins with
     w_i = Jw_i(q) q_dot (Jv_i, Jw_i: the Jacobians of that link, built like the
     Milestone 3 Jacobian but for the point c_i). So
         K = 1/2 sum_i ( m_i v_ci^T v_ci + w_i^T (R_i I_i R_i^T) w_i ) + 1/2 sum_j Ia_j q_dot_j^2
           = 1/2 q_dot^T M(q) q_dot
         M(q) = sum_i ( m_i Jv_i^T Jv_i + Jw_i^T R_i I_i R_i^T Jw_i ) + diag(Ia)
     Ia is the motor's rotor inertia seen at the joint ("armature" in the XML).
  2. Potential energy. P = sum_i m_i g0 z_ci (g0 = 9.81 m/s^2, z_ci = CoM height).
     Its gradient is the gravity vector:
         g_k(q) = dP/dq_k = sum_i m_i g0 dz_ci/dq_k = sum_i m_i g0 (Jv_i)[z, k]
  3. Euler-Lagrange, d/dt dL/dq_dot - dL/dq = tau with L = K - P, gives
         M q_ddot + C q_dot + g = tau
     with C from the Christoffel symbols of M:
         C_kj = sum_i 1/2 ( dM_kj/dq_i + dM_ki/dq_j - dM_ij/dq_k ) q_dot_i
     dM/dq_i is computed in closed form below (no finite differences).

The masses, CoMs and inertias are the numbers in ur5e.xml and 2f85.xml. The
XML gives them in MuJoCo's link frames; each MuJoCo link frame is fixed to the
same link as our DH frame, so it is a constant rotation + shift away. We move
the numbers into our DH frames once, at import (LINKS below).
The gripper is treated as one rigid body bolted to link 6 (fingers open). Its
13 parts weigh 1.05 kg together; their combined CoM and inertia (parallel-axis
theorem) are GRIPPER_* below.

validate_dynamics.py checks M, C q_dot + g and the full torque against MuJoCo.
"""
import numpy as np

from kinematics import forward_kinematics

G0 = 9.81
GRAVITY = np.array([0.0, 0.0, -G0])        # base frame z points up
ARMATURE = np.full(6, 0.1)                 # rotor inertia at each joint [kg m^2], ur5e.xml
TORQUE_LIMIT = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])   # [N m], ur5e.xml forcerange (= UR5e datasheet)
FLANGE_TO_PINCH = 0.156                    # flange to the point between the gripper pads [m]


def _quat_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Link parameters, straight from ur5e.xml (<inertial> of each body)
#   mass [kg], CoM in the MuJoCo link frame [m], principal inertias [kg m^2],
#   orientation of the principal axes (quat w x y z) in the MuJoCo link frame.
# R_DH, P_DH: the MuJoCo link frame seen from our DH frame of the same link
#   (constant, read off the XML body chain and our DH table).
# ---------------------------------------------------------------------------
_XML_LINKS = [
    # shoulder_link: DH frame 1
    dict(mass=3.7, com=[0, 0, 0], diag=[0.0102675, 0.0102675, 0.00666], quat=[1, 0, 0, 0],
         R_DH=[[-1, 0, 0], [0, 0, 1], [0, 1, 0]], P_DH=[0, 0, 0]),
    # upper_arm_link: DH frame 2 (the DH origin is at the elbow, 0.425 m along the arm)
    dict(mass=8.393, com=[0, 0, 0.2125], diag=[0.133886, 0.133886, 0.0151074], quat=[1, 0, 0, 0],
         R_DH=[[0, 0, -1], [-1, 0, 0], [0, 1, 0]], P_DH=[0.425, 0, 0.138]),
    # forearm_link: DH frame 3
    dict(mass=2.275, com=[0, 0, 0.196], diag=[0.0311796, 0.0311796, 0.004095], quat=[1, 0, 0, 0],
         R_DH=[[0, 0, -1], [-1, 0, 0], [0, 1, 0]], P_DH=[0.392, 0, 0.007]),
    # wrist_1_link: DH frame 4
    dict(mass=1.219, com=[0, 0.127, 0], diag=[0.0025599, 0.0025599, 0.0021942], quat=[1, 0, 0, 0],
         R_DH=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], P_DH=[0, -0.127, 0]),
    # wrist_2_link: DH frame 5
    dict(mass=1.219, com=[0, 0, 0.1], diag=[0.0025599, 0.0025599, 0.0021942], quat=[1, 0, 0, 0],
         R_DH=[[1, 0, 0], [0, 0, -1], [0, 1, 0]], P_DH=[0, 0.1, 0]),
    # wrist_3_link: DH frame 6 (the flange frame)
    dict(mass=0.1889, com=[0, 0.0771683, 0], diag=[0.000132134, 9.90863e-05, 9.90863e-05], quat=[1, 0, 0, 1],
         R_DH=[[1, 0, 0], [0, 0, -1], [0, 1, 0]], P_DH=[0, 0, -0.1]),
]

# The whole 2F-85 gripper as one body, in the wrist_3_link (MuJoCo) frame, fingers open.
GRIPPER_MASS = 1.0526083388427392
GRIPPER_COM = np.array([-7.1356177129e-05, 1.4667383286e-01, -1.2164691678e-05])
GRIPPER_INERTIA = np.array([                       # about its own CoM
    [9.1189734659e-04, -2.5225810022e-06, -1.5792936981e-09],
    [-2.5225810022e-06, 5.8000106063e-04, -5.0876179374e-07],
    [-1.5792936981e-09, -5.0876179374e-07, 1.2584823745e-03],
])


def _combine(m1, c1, I1, m2, c2, I2):
    """Two rigid bodies bolted together -> one body (parallel-axis theorem)."""
    m = m1 + m2
    c = (m1 * c1 + m2 * c2) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in ((m1, c1, I1), (m2, c2, I2)):
        r = ci - c
        I += Ii + mi * (r @ r * np.eye(3) - np.outer(r, r))
    return m, c, I


def build_links(payload_mass=0.0):
    """
    (mass, CoM, inertia about the CoM) of the 6 moving links, all in our DH frames.

    payload_mass: a part held between the gripper pads (a point mass at the
    pinch point, FLANGE_TO_PINCH along the tool axis). 0 = empty gripper.
    """
    links = []
    for i, L in enumerate(_XML_LINKS):
        R_dh = np.array(L["R_DH"], dtype=float)
        p_dh = np.array(L["P_DH"], dtype=float)
        R_principal = _quat_to_R(*L["quat"])
        m = L["mass"]
        c = np.array(L["com"], dtype=float)                       # MuJoCo link frame
        I = R_principal @ np.diag(L["diag"]) @ R_principal.T
        if i == 5:
            m, c, I = _combine(m, c, I, GRIPPER_MASS, GRIPPER_COM, GRIPPER_INERTIA)
            if payload_mass > 0:
                pinch = np.array([0.0, 0.1 + FLANGE_TO_PINCH, 0.0])  # flange is 0.1 m along y
                m, c, I = _combine(m, c, I, payload_mass, pinch, np.zeros((3, 3)))
        links.append((m, R_dh @ c + p_dh, R_dh @ I @ R_dh.T))     # -> DH frame
    return links


LINKS = build_links()


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Model:
    """The rigid-body model. Model() is the nominal arm; Model(payload_mass=2.0) carries 2 kg."""

    def __init__(self, payload_mass=0.0, armature=ARMATURE):
        self.links = build_links(payload_mass) if payload_mass else LINKS
        self.armature = np.asarray(armature, dtype=float)
        self._cache_key = None

    # -- geometry of one instant ---------------------------------------------
    def _geometry(self, q):
        """Joint axes z_j and points o_j (j = 0..5), and per link: CoM c_i, world inertia R_i I_i R_i^T."""
        q = np.asarray(q, dtype=float)
        key = q.tobytes()
        if key == self._cache_key:              # M, C and g of the same q share one FK
            return self._cache
        F = forward_kinematics(q, return_all=True)
        Z = np.array([F[j][:3, 2] for j in range(6)])
        O = np.array([F[j][:3, 3] for j in range(6)])
        out = []
        for i, (m, c, I) in enumerate(self.links):
            R = F[i + 1][:3, :3]                  # link i+1 is fixed to DH frame i+1
            out.append((m, R @ c + F[i + 1][:3, 3], R @ I @ R.T))
        self._cache_key, self._cache = key, (Z, O, out)
        return Z, O, out

    @staticmethod
    def _link_jacobians(Z, O, i, c):
        """Jv, Jw (3x6) of link i's CoM c. Only joints 1..i+1 move link i."""
        Jv = np.zeros((3, 6))
        Jw = np.zeros((3, 6))
        for j in range(i + 1):
            Jv[:, j] = np.cross(Z[j], c - O[j])
            Jw[:, j] = Z[j]
        return Jv, Jw

    # -- M, g, C ----------------------------------------------------------------
    def mass_matrix(self, q):
        """M(q), 6x6, from the kinetic energy: sum of m Jv^T Jv + Jw^T I Jw, plus the rotor inertias."""
        Z, O, links = self._geometry(q)
        M = np.diag(self.armature)
        for i, (m, c, Iw) in enumerate(links):
            Jv, Jw = self._link_jacobians(Z, O, i, c)
            M += m * Jv.T @ Jv + Jw.T @ Iw @ Jw
        return M

    def gravity(self, q):
        """g(q) = dP/dq, the joint torques that just hold the arm still against gravity."""
        Z, O, links = self._geometry(q)
        g = np.zeros(6)
        for i, (m, c, _) in enumerate(links):
            Jv, _ = self._link_jacobians(Z, O, i, c)
            g -= m * Jv.T @ GRAVITY
        return g

    def mass_matrix_derivatives(self, q):
        """
        dM/dq_k for k = 0..5, as an array D[k] (6x6 each), in closed form.

        Turning joint k (axis z_k through o_k) rigidly rotates everything after it:
            d z_j / dq_k = z_k x z_j              (j > k)
            d o_j / dq_k = z_k x (o_j - o_k)      (j > k)
            d c_i / dq_k = z_k x (c_i - o_k)      (link i after joint k)
            d (R I R^T)/dq_k = S I_w - I_w S,     S = [z_k]x
        and everything before joint k does not move. Differentiate
        M = sum m Jv^T Jv + Jw^T I_w Jw with the product rule.
        """
        Z, O, links = self._geometry(q)
        D = np.zeros((6, 6, 6))
        later = np.triu(np.ones((6, 6)), 1)[:, :, None]        # later[k, j] = 1 if j > k
        dz_all = np.cross(Z[:, None, :], Z[None, :, :]) * later                 # dz_j/dq_k
        do_all = np.cross(Z[:, None, :], O[None, :, :] - O[:, None, :]) * later  # do_j/dq_k
        for i, (m, c, Iw) in enumerate(links):
            n = i + 1                                   # joints 0..i move link i, the others don't
            Jv, Jw = self._link_jacobians(Z, O, i, c)
            Zn, On = Z[:n], O[:n]
            dz, do = dz_all[:n, :n], do_all[:n, :n]     # [k, j, xyz]
            dc = np.cross(Zn, c - On)                   # dc/dq_k, [k, xyz]
            # dJv[k, :, j] = dz x (c - o_j) + z_j x (dc - do)
            dJv = np.cross(dz, c - On[None, :, :]) + np.cross(Zn[None, :, :], dc[:, None, :] - do)
            dJv = np.transpose(dJv, (0, 2, 1))          # [k, xyz, j]
            dJw = np.transpose(dz, (0, 2, 1))
            S = np.array([_skew(z) for z in Zn])        # [k, 3, 3]
            dIw = S @ Iw - Iw @ S
            Jvn, Jwn = Jv[:, :n], Jw[:, :n]
            A = m * np.transpose(dJv, (0, 2, 1)) @ Jvn + np.transpose(dJw, (0, 2, 1)) @ Iw @ Jwn
            D[:n, :n, :n] += A + np.transpose(A, (0, 2, 1)) + Jwn.T @ dIw @ Jwn
        return D

    def coriolis_matrix(self, q, dq):
        """C(q, q_dot) from the Christoffel symbols: C_kj = 1/2 sum_i (dM_kj/dq_i + dM_ki/dq_j - dM_ij/dq_k) q_dot_i."""
        D = self.mass_matrix_derivatives(q)
        dq = np.asarray(dq, dtype=float)
        term1 = np.einsum("ikj,i->kj", D, dq)      # dM_kj/dq_i q_dot_i
        term2 = np.einsum("jki,i->kj", D, dq)      # dM_ki/dq_j q_dot_i
        term3 = np.einsum("kij,i->kj", D, dq)      # dM_ij/dq_k q_dot_i
        return 0.5 * (term1 + term2 - term3)

    def bias(self, q, dq):
        """C(q, q_dot) q_dot + g(q): what the motors must give at q_ddot = 0."""
        return self.coriolis_matrix(q, dq) @ np.asarray(dq, dtype=float) + self.gravity(q)

    def inverse_dynamics(self, q, dq, ddq):
        """tau = M q_ddot + C q_dot + g."""
        return self.mass_matrix(q) @ np.asarray(ddq, dtype=float) + self.bias(q, dq)

    def forward_dynamics(self, q, dq, tau):
        """q_ddot = M^-1 (tau - C q_dot - g)."""
        return np.linalg.solve(self.mass_matrix(q), np.asarray(tau, dtype=float) - self.bias(q, dq))

    def energy(self, q, dq):
        """Kinetic and potential energy [J] (P measured from the base height)."""
        _, _, links = self._geometry(q)
        dq = np.asarray(dq, dtype=float)
        K = 0.5 * dq @ self.mass_matrix(q) @ dq
        P = sum(m * G0 * c[2] for m, c, _ in links)
        return K, P


NOMINAL = Model()
