import numpy as np

ROBOT_BASE_POS = np.array([0.0, 0.0, 0.4])

# EL INDEXES HENA BASE SHOULDER ELBOW WRIST1 WRIST2 WRIST3
HOME_Q = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

# DOL HEBTOHOM MEN EL DATA SHEET ELLY FE EL UR5E MANUAL ZAY MA2ASATO KDA ( MA2ASAT GESM EL ROBOT W EL JOINTS W EL MASAFA BEN EL JOINTS WEL TANYA)
D = np.array([0.163, 0.0, 0.0, 0.134, 0.1, 0.1]) #MASAFA MEN FO2 AW EL GANB 
A = np.array([0.0, -0.425, -0.392, 0.0, 0.0, 0.0]) #EL MASAFA BEN EL JOINTS WEL TANYA 
ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]) # EL JOINT EL GAY LAFEF AD EH 3AN EL ABLO YAANY LW MASLN 0 YEB2A PARALLEL


def dh_transform(theta, d, a, alpha): #law el aplha be 0 yeb2a parallel lw el alpha be 90 yeb2a perpindicular
    ct, st = np.cos(theta), np.sin(theta)#benlef hawlen el z, de betetghayr zay kda lama tefth bab ala mafsal el mafsal sabt da alpha enama el theta bttghyr 
    ca, sa = np.cos(alpha), np.sin(alpha) #benlef hawlen el x de msh betetghyr bethaded mara wahda
    return np.array([
        [ct, -st * ca, st * sa, a * ct], # awel 3 columns rotation w akher column position
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])

#function el forward kinematics betedeha el information beta3et el joints w el angles we heya t2olak el gripper feen dlwa2ty law enta f3lan fel positions de aw hata hagat enta betfkr feha w 3ayz teshif el gripper haykoun feen?
def forward_kinematics(q, return_all=False):
    T = np.eye(4) #e3mel identity matrix 4x4 (1000,0100,0010,0001) ka2ny ba2ol ana wa2ef and eel base w lsa mathrktsh w mengher ay rotation
    frames = [T.copy()] #hena ehna bensave el frames elly 3amelna fehom rotation w translation 3ashan n3rf el position w orientation bta3 kol joint w initially be (base)
    for i in range(6): #run el hwar da 6 marat 3ashan el robot 6 joints
        T = T @ dh_transform(q[i], D[i], A[i], ALPHA[i]) # hanmshy step by step men awel el base lehad el gripper w kol marra han3ml el rotation w translation bta3 el joint da 3ashan n3rf el position w orientation bta3 el joint da
        frames.append(T.copy()) # baad kol khatwa ekteb fe frames fa tege fel akher teb2a 3aref kol joint ba2a feen bzbt
    return frames if return_all else T


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def inverse_kinematics(T, eps=1e-9):
    """Analytical IK for the UR5e. Returns an (N, 6) array of solutions, N <= 8."""
    d1, d4, d5, d6 = D[0], D[3], D[4], D[5]
    a2, a3 = A[1], A[2]
    solutions = []


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
                q6 = 0.0 
            else:
                q6 = np.arctan2(
                    (-T[0, 1] * s1 + T[1, 1] * c1) / s5,
                    (T[0, 0] * s1 - T[1, 0] * c1) / s5,
                )

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
