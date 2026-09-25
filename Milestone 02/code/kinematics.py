import numpy as np

# Where the robot base sits in the world (on top of the 0.4 m pedestal in environment.xml).
# FK/IK below work in the robot base frame; add this to get world coordinates.
ROBOT_BASE_POS = np.array([0.0, 0.0, 0.4])

# Standard UR5e "ready" pose (elbow up, tool pointing down), in radians
HOME_Q = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

# Standard DH parameters of the UR5e, taken from the mujoco_menagerie model geometry.
# Official UR5e values: d1=0.1625, a2=-0.425, a3=-0.3922, d4=0.1333, d5=0.0997, d6=0.0996
D = np.array([0.163, 0.0, 0.0, 0.134, 0.1, 0.1])
A = np.array([0.0, -0.425, -0.392, 0.0, 0.0, 0.0])
ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])


def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def forward_kinematics(q, return_all=False):
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        T = T @ dh_transform(q[i], D[i], A[i], ALPHA[i])
        frames.append(T.copy())
    return frames if return_all else T


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def inverse_kinematics(T, eps=1e-9):
    """Analytical IK for the UR5e. Returns an (N, 6) array of solutions, N <= 8."""
    d1, d4, d5, d6 = D[0], D[3], D[4], D[5]
    a2, a3 = A[1], A[2]
    solutions = []

    # Wrist center (origin of frame 5), found by stepping back d6 along the tool z-axis
    p05 = T[:3, 3] - d6 * T[:3, 2]
    r = np.hypot(p05[0], p05[1])
    if r < abs(d4):
        return np.empty((0, 6))
    phi = np.arctan2(p05[1], p05[0])
    psi = np.arccos(np.clip(d4 / r, -1.0, 1.0))

    for q1 in (phi + psi + np.pi / 2, phi - psi + np.pi / 2):
        s1, c1 = np.sin(q1), np.cos(q1)

        c5 = (T[0, 3] * s1 - T[1, 3] * c1 - d4) / d6
        if abs(c5) > 1.0 + eps:
            continue
        acos5 = np.arccos(np.clip(c5, -1.0, 1.0))

        for q5 in (acos5, -acos5):
            s5 = np.sin(q5)
            if abs(s5) < eps:
                q6 = 0.0  # wrist singularity: q4 and q6 are coupled, pick q6 = 0
            else:
                q6 = np.arctan2(
                    (-T[0, 1] * s1 + T[1, 1] * c1) / s5,
                    (T[0, 0] * s1 - T[1, 0] * c1) / s5,
                )

            # Remove joints 1, 5, 6 to leave the planar 3-link chain (joints 2, 3, 4)
            T01 = dh_transform(q1, D[0], A[0], ALPHA[0])
            T45 = dh_transform(q5, D[4], A[4], ALPHA[4])
            T56 = dh_transform(q6, D[5], A[5], ALPHA[5])
            T14 = np.linalg.inv(T01) @ T @ np.linalg.inv(T45 @ T56)
            px, py = T14[0, 3], T14[1, 3]

            c3 = (px ** 2 + py ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
            if abs(c3) > 1.0 + eps:
                continue
            acos3 = np.arccos(np.clip(c3, -1.0, 1.0))

            for q3 in (acos3, -acos3):
                q2 = np.arctan2(py, px) - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))
                q4 = np.arctan2(T14[1, 0], T14[0, 0]) - q2 - q3
                solutions.append(wrap_angle(np.array([q1, q2, q3, q4, q5, q6])))

    return np.array(solutions) if solutions else np.empty((0, 6))
