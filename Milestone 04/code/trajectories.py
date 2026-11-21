"""
Trajectory planning for the UR5e (Milestone 4).

Every segment answers the same question: at time t, what should the joints be
doing? sample(t) -> (q_d, q_dot_d, q_ddot_d). The controller (control.py) then
makes the arm follow that.

Joint space (JointQuintic, JointLSPB)
    All 6 joints move from q0 to q1 on one clock, so they start and stop together.
        quintic: q(t) = q0 + h(s) (q1 - q0),  s = t/T,  h = 10 s^3 - 15 s^4 + 6 s^5
                 6 conditions fix the 6 coefficients: q, q_dot, q_ddot given at both ends
                 (start and end at rest, with zero acceleration -> no jerk at the ends).
        LSPB:    "linear segment with parabolic blends", a trapezoid in velocity:
                 accelerate at a constant rate, cruise at constant speed, brake.
                 Fastest for given speed and acceleration limits, but the acceleration
                 jumps (infinite jerk) at the corners of the trapezoid.
    The flange's path between the two poses is whatever the joints make it: a curve.

Task space (TaskLine)
    The flange moves on a straight line from p0 to p1, gripper orientation turning
    from R0 to R1 about one fixed axis, with the same quintic time law s(t):
        p(t) = p0 + h(s) (p1 - p0)                   v = h' (p1 - p0) / T     a = h'' (p1 - p0) / T^2
        R(t) = Rot(k, h(s) theta) R0                 w = h' theta k / T       alpha = h'' theta k / T^2
    and the joints follow from the Milestone 3 inverse kinematics:
        q       = IK(p, R)                   (inverse position, solution nearest the last one)
        q_dot   = J^-1 [v; w]                (inverse velocity)
        q_ddot  = J^-1 ([a; alpha] - J_dot q_dot)   (inverse acceleration)

Why both, in our cell: pick and place need the gripper to go straight down onto the
part and straight up again (a curved joint-space path could knock the part or hit the
bin wall), so those moves are TaskLines. The long transfers between the conveyor and the
bins only need to get there fast and safely, so they are joint-space quintics: no IK on
the way and no risk of passing near a singularity.
"""
import numpy as np

import velocity_kinematics as vk
from kinematics import forward_kinematics, inverse_kinematics, wrap_angle

# UR5e joint speed limit is 180 deg/s on every joint (datasheet). The planner uses only
# a fraction of the speed and of the acceleration we allow, to keep torque in reserve
# for the feedback.
QD_MAX = np.full(6, np.pi)
QDD_MAX = np.array([6.0, 6.0, 8.0, 12.0, 12.0, 12.0])
LIMIT_FRACTION = 0.5

QUINTIC_PEAK_VEL = 15 / 8       # max of h'(s)
QUINTIC_PEAK_ACC = 10 / np.sqrt(3)   # max of |h''(s)|


def quintic_profile(s):
    """h(s), h'(s), h''(s) of the rest-to-rest quintic, for s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return (10 * s**3 - 15 * s**4 + 6 * s**5,
            30 * s**2 * (1 - s)**2,
            60 * s * (1 - s) * (1 - 2 * s))


def quintic_coefficients(q0, q1, T, v0=0.0, v1=0.0, a0=0.0, a1=0.0):
    """
    The 6 coefficients a0..a5 of q(t) = sum a_k t^k from the 6 boundary conditions
    (position, velocity, acceleration at t = 0 and t = T), by solving the 6x6 system.
    With zero end velocities and accelerations it gives the same curve as quintic_profile.
    """
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
        [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
    ], dtype=float)
    return np.linalg.solve(A, np.array([q0, v0, a0, q1, v1, a1], dtype=float))


class Segment:
    """One piece of a plan. Subclasses give T and sample(t)."""
    kind = "segment"
    T = 0.0

    def sample(self, t):
        raise NotImplementedError

    @property
    def q_end(self):
        return self.sample(self.T)[0]


class Hold(Segment):
    """Stay still at q for T seconds (e.g. while the gripper closes)."""
    kind = "hold"

    def __init__(self, q, T):
        self.q, self.T = np.asarray(q, dtype=float), float(T)

    def sample(self, t):
        return self.q.copy(), np.zeros(6), np.zeros(6)


class JointQuintic(Segment):
    """Joint-space quintic from q0 to q1, all joints on one clock."""
    kind = "joint"

    def __init__(self, q0, q1, T=None):
        self.q0, self.q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
        self.dq = self.q1 - self.q0
        self.T = float(T) if T is not None else self.minimum_time(self.dq)

    @staticmethod
    def minimum_time(dq, fraction=LIMIT_FRACTION):
        """Shortest T that keeps every joint under fraction x its speed and acceleration limits.

        Peak speed is (15/8) |dq| / T and peak acceleration (10/sqrt 3) |dq| / T^2."""
        dq = np.abs(np.asarray(dq, dtype=float))
        T_vel = QUINTIC_PEAK_VEL * dq / (fraction * QD_MAX)
        T_acc = np.sqrt(QUINTIC_PEAK_ACC * dq / (fraction * QDD_MAX))
        return float(max(T_vel.max(), T_acc.max(), 0.5))

    def sample(self, t):
        h, hd, hdd = quintic_profile(t / self.T)
        return self.q0 + h * self.dq, hd / self.T * self.dq, hdd / self.T**2 * self.dq


class JointLSPB(Segment):
    """
    Joint-space trapezoidal velocity profile (LSPB), all joints on one clock.

    Normalised: s goes 0 -> 1 in T. Blend time tb = T / 3 (a common choice): accelerate
    for T/3, cruise for T/3, brake for T/3. Cruise rate s_dot = 1.5 / T, blend
    acceleration s_ddot = 4.5 / T^2.
    """
    kind = "joint-lspb"

    def __init__(self, q0, q1, T=None):
        self.q0, self.q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
        self.dq = self.q1 - self.q0
        if T is None:
            d = np.abs(self.dq)
            T = max((1.5 * d / (LIMIT_FRACTION * QD_MAX)).max(),
                    np.sqrt(4.5 * d / (LIMIT_FRACTION * QDD_MAX)).max(), 0.5)
        self.T = float(T)

    def sample(self, t):
        T = self.T
        tb, vc, ac = T / 3, 1.5 / T, 4.5 / T**2
        t = min(max(t, 0.0), T)
        if t < tb:
            s, sd, sdd = 0.5 * ac * t**2, ac * t, ac
        elif t <= T - tb:
            s, sd, sdd = 0.5 * ac * tb**2 + vc * (t - tb), vc, 0.0
        else:
            r = T - t
            s, sd, sdd = 1.0 - 0.5 * ac * r**2, ac * r, -ac
        return self.q0 + s * self.dq, sd * self.dq, sdd * self.dq


def _axis_angle(R):
    """Rotation matrix -> (unit axis, angle)."""
    angle = vk.rotation_error(R, np.eye(3))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.linalg.norm(w) < 1e-6:                     # angle ~ pi
        B = (R + np.eye(3)) / 2
        axis = B[:, np.argmax(np.diag(B))]
    else:
        axis = w
    return axis / np.linalg.norm(axis), angle


def _rotation(axis, angle):
    """Rodrigues: rotation by angle about the unit axis."""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def ik_nearest(T, q_ref):
    """The IK solution closest to q_ref, unwrapped so it does not jump by 2 pi."""
    solutions = inverse_kinematics(T)
    if len(solutions) == 0:
        raise ValueError(f"pose {np.round(T[:3, 3], 3)} is out of reach")
    deltas = wrap_angle(solutions - q_ref)
    return q_ref + deltas[np.argmin(np.abs(deltas).sum(axis=1))]


class TaskSegment(Segment):
    """A flange motion in task space. Subclasses give flange(t) = (p, R, v, w, a, alpha)."""
    kind = "task"

    def __init__(self, q0):
        self.q0 = np.asarray(q0, dtype=float)
        T0 = forward_kinematics(self.q0)
        self.p0, self.R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
        self._q_last = self.q0.copy()

    def flange(self, t):
        raise NotImplementedError

    def sample(self, t):
        p, R, v, w, a, alpha = self.flange(t)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        q = ik_nearest(T, self._q_last)                          # inverse position
        dq = vk.inverse_velocity(q, v, w)                        # inverse velocity
        ddq = vk.inverse_acceleration(q, dq, a, alpha)           # inverse acceleration
        self._q_last = q
        return q, dq, ddq


class TaskLine(TaskSegment):
    """Flange on a straight line p0 -> p1 (base frame), orientation R0 -> R1, quintic time law."""

    def __init__(self, q0, p1, R1=None, T=None, speed=0.15):
        """
        q0: joint angles at the start (gives p0, R0 and the IK branch).
        p1, R1: flange target in the base frame (R1 = keep the start orientation).
        T: duration; if None, chosen so the peak flange speed is `speed` m/s.
        """
        super().__init__(q0)
        self.p1 = np.asarray(p1, dtype=float)
        self.R1 = self.R0 if R1 is None else np.asarray(R1, dtype=float)
        self.d = self.p1 - self.p0
        self.axis, self.theta = _axis_angle(self.R1 @ self.R0.T)
        if T is None:
            T = max(QUINTIC_PEAK_VEL * np.linalg.norm(self.d) / speed, QUINTIC_PEAK_VEL * self.theta / 1.0, 0.5)
        self.T = float(T)

    def flange(self, t):
        """Planned flange pose, twist and acceleration at time t."""
        h, hd, hdd = quintic_profile(t / self.T)
        p = self.p0 + h * self.d
        R = _rotation(self.axis, h * self.theta) @ self.R0
        v, a = hd / self.T * self.d, hdd / self.T**2 * self.d
        w = hd / self.T * self.theta * self.axis
        alpha = hdd / self.T**2 * self.theta * self.axis
        return p, R, v, w, a, alpha


class TaskCircle(TaskSegment):
    """
    Flange on a horizontal circle through its start point, orientation fixed.
    The angle follows the quintic: phi(t) = 2 pi turns h(s), so it starts and stops at rest.
        p = c + r [cos phi, sin phi, 0]
        v = phi_dot r [-sin phi, cos phi, 0]
        a = phi_ddot r [-sin phi, cos phi, 0] - phi_dot^2 r [cos phi, sin phi, 0]   (tangential + centripetal)
    """

    def __init__(self, q0, radius, T, turns=1.0):
        super().__init__(q0)
        self.r, self.T, self.turns = float(radius), float(T), float(turns)
        self.centre = self.p0 - np.array([self.r, 0.0, 0.0])

    def flange(self, t):
        h, hd, hdd = quintic_profile(t / self.T)
        span = 2 * np.pi * self.turns
        phi, phid, phidd = span * h, span * hd / self.T, span * hdd / self.T**2
        radial = self.r * np.array([np.cos(phi), np.sin(phi), 0.0])
        tangent = self.r * np.array([-np.sin(phi), np.cos(phi), 0.0])
        zero = np.zeros(3)
        return self.centre + radial, self.R0, phid * tangent, zero, phidd * tangent - phid**2 * radial, zero


class Plan:
    """
    A list of segments played one after the other, each with a gripper command and a label.
    sample(t) -> (q_d, q_dot_d, q_ddot_d, gripper, label, index of the segment).
    """

    def __init__(self, q_start):
        self.steps = []
        self.q_end = np.asarray(q_start, dtype=float)

    def add(self, segment, gripper, label):
        self.steps.append((segment, gripper, label))
        self.q_end = segment.q_end
        return self

    @property
    def T(self):
        return sum(seg.T for seg, _, _ in self.steps)

    def sample(self, t):
        t0 = 0.0
        for i, (seg, gripper, label) in enumerate(self.steps):
            if t < t0 + seg.T or i == len(self.steps) - 1:
                q, dq, ddq = seg.sample(min(t - t0, seg.T))
                if t > t0 + seg.T:                    # past the end: hold the last point
                    dq, ddq = np.zeros(6), np.zeros(6)
                return q, dq, ddq, gripper, label, i
            t0 += seg.T
        raise ValueError("empty plan")
