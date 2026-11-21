"""
Motor-level control of the UR5e (Milestone 4): joint torques from the tracking error.

Every controller gets the sensed state (q, q_dot) and the planned state
(q_d, q_dot_d, q_ddot_d) and returns the 6 motor torques. Errors:
    e = q_d - q          e_dot = q_dot_d - q_dot

1. PD (no model)
       tau = Kp e + Kd e_dot
   Each joint is a spring-damper pulling towards the plan. Gravity is a constant load
   it can only fight by letting e grow: the arm sags until Kp e equals the weight.

2. PD + gravity compensation
       tau = Kp e + Kd e_dot + g(q)
   g(q) from our Lagrangian model holds the arm up, so at rest the error goes to zero
   (stable for any positive Kp, Kd, Lyapunov argument with the energy). While moving,
   the inertia and Coriolis torques are still left to the PD, so it lags.

3. Computed torque (inverse dynamics, feedback linearisation)
       tau = M(q) (q_ddot_d + Kd e_dot + Kp e) + C(q, q_dot) q_dot + g(q)
   Put it into M q_ddot + C q_dot + g = tau: the model terms cancel and what is left is
       e_ddot + Kd e_dot + Kp e = 0
   for every joint, separately: a linear, decoupled 2nd-order system with the poles we
   choose. Kp = wn^2, Kd = 2 zeta wn; zeta = 1 (critically damped, no overshoot).
   The model does the heavy lifting (it already knows the torque the plan needs) and
   the PD only fixes what the model gets wrong.

All torques are clipped to the UR5e's limits (150 N m big joints, 28 N m wrist).
"""
import numpy as np

from dynamics import NOMINAL, TORQUE_LIMIT

# Computed torque: every joint error behaves like e_ddot + 2 wn e_dot + wn^2 e = 0.
CT_WN = 20.0                  # rad/s -> settles in about 4 / wn = 0.2 s
CT_ZETA = 1.0

# PD and PD+g gains [N m/rad, N m s/rad]: picked for the same wn = 20 rad/s as the computed
# torque, using each joint's own inertia M_ii in our sorting poses, Kp = M_ii wn^2, Kd = 2 M_ii wn.
# M_ii = 1.3-1.8, 2.4-2.5, 1.0-1.1, 0.155, 0.127, 0.101 kg m^2 -> rounded up to the values below.
PD_M = np.array([2.0, 2.5, 1.1, 0.16, 0.13, 0.10])
PD_KP = PD_M * CT_WN**2
PD_KD = PD_M * 2 * CT_ZETA * CT_WN


class Controller:
    name = "controller"
    uses_model = False

    def __init__(self, model=NOMINAL):
        self.model = model
        self.last_parts = {}

    def torque(self, q, dq, q_d, dq_d, ddq_d):
        raise NotImplementedError

    def __call__(self, q, dq, q_d, dq_d, ddq_d):
        tau = self.torque(np.asarray(q, dtype=float), np.asarray(dq, dtype=float),
                          np.asarray(q_d, dtype=float), np.asarray(dq_d, dtype=float),
                          np.asarray(ddq_d, dtype=float))
        return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


class PD(Controller):
    name = "PD" 

    def __init__(self, model=NOMINAL, kp=PD_KP, kd=PD_KD):
        super().__init__(model)
        self.kp, self.kd = kp, kd

    def torque(self, q, dq, q_d, dq_d, ddq_d):
        fb = self.kp * (q_d - q) + self.kd * (dq_d - dq)
        self.last_parts = {"feedback": fb}
        return fb


class PDGravity(PD):
    name = "PD + gravity compensation"
    uses_model = True

    def torque(self, q, dq, q_d, dq_d, ddq_d):
        fb = self.kp * (q_d - q) + self.kd * (dq_d - dq)
        g = self.model.gravity(q)
        self.last_parts = {"feedback": fb, "gravity": g}
        return fb + g


class ComputedTorque(Controller):
    name = "computed torque"
    uses_model = True

    def __init__(self, model=NOMINAL, wn=CT_WN, zeta=CT_ZETA):
        super().__init__(model)
        self.kp, self.kd = wn**2, 2 * zeta * wn

    def torque(self, q, dq, q_d, dq_d, ddq_d):
        e, e_dot = q_d - q, dq_d - dq
        v = ddq_d + self.kd * e_dot + self.kp * e          # the acceleration we want
        M = self.model.mass_matrix(q)
        bias = self.model.bias(q, dq)                     # C q_dot + g
        self.last_parts = {"feedforward": M @ ddq_d + bias, "feedback": M @ (self.kd * e_dot + self.kp * e)}
        return M @ v + bias


class ComputedTorqueIntegral(ComputedTorque):
    """
    Computed torque with an integral term in the outer loop:
        v = q_ddot_d + Kd e_dot + Kp e + Ki integral(e dt)
    -> e_dddot + Kd e_ddot + Kp e_dot + Ki e = 0 (differentiated once). Stable while
    Kd Kp > Ki (Routh), here 40 * 400 = 16000 > 2000. The integral slowly builds up the
    torque that a constant unknown load needs (a payload, a wrong link mass), so the
    steady error goes to zero instead of staying at load / (M Kp).
    """
    name = "computed torque + integral"

    def __init__(self, model=NOMINAL, wn=CT_WN, zeta=CT_ZETA, ki=2000.0, dt=0.002, limit=0.02):
        super().__init__(model, wn, zeta)
        self.ki, self.dt, self.limit = ki, dt, limit     # limit: anti-windup clamp on the integral [rad s]
        self.integral = np.zeros(6)

    def torque(self, q, dq, q_d, dq_d, ddq_d):
        self.integral = np.clip(self.integral + (q_d - q) * self.dt, -self.limit, self.limit)
        return super().torque(q, dq, q_d, dq_d, ddq_d + self.ki * self.integral)


CONTROLLERS = {"pd": PD, "pdg": PDGravity, "ctc": ComputedTorque, "ctci": ComputedTorqueIntegral}


def make(name, model=NOMINAL):
    return CONTROLLERS[name](model)
