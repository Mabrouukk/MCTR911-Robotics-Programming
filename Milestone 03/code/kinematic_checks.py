"""
Our kinematic equations vs the simulator's sensors, at one instant (Milestone 3).

Pure numpy, so the same check runs in the MuJoCo simulator on the Mac
(live panel in the viewer), in the offline validation script, and in the
ROS 2 validator node inside Docker.
"""
import numpy as np

import velocity_kinematics as vk
from kinematics import forward_kinematics, inverse_kinematics, wrap_angle


def kinematics_check(q, dq, ddq, ee):
    """
    Our equations vs the sensors, at one instant. Returns a dict of errors.

    forward: FK(q), J q_dot, J q_ddot + J_dot q_dot   vs   sensed pose, twist, acceleration
    inverse: IK(sensed pose), J^-1 x_dot, J^-1 (x_ddot - J_dot q_dot)   vs   sensed q, q_dot, q_ddot
    """
    T = forward_kinematics(q)
    v, w = vk.forward_velocity(q, dq)
    a, alpha = vk.forward_acceleration(q, dq, ddq)
    out = {
        "fk_pos": np.linalg.norm(T[:3, 3] - ee["p"]),
        "fk_rot": vk.rotation_error(T[:3, :3], ee["R"]),
        "fv_lin": np.linalg.norm(v - ee["v"]),
        "fv_ang": np.linalg.norm(w - ee["w"]),
        "fa_lin": np.linalg.norm(a - ee["a"]),
        "fa_ang": np.linalg.norm(alpha - ee["alpha"]),
        "pred": (T[:3, 3], v, w, a, alpha),
    }
    T_sensed = np.eye(4)
    T_sensed[:3, :3], T_sensed[:3, 3] = ee["R"], ee["p"]
    solutions = inverse_kinematics(T_sensed)
    out["ik"] = np.abs(wrap_angle(solutions - q)).max(axis=1).min() if len(solutions) else np.inf
    out["iv"] = np.abs(vk.inverse_velocity(q, ee["v"], ee["w"]) - dq).max()
    out["ia"] = np.abs(vk.inverse_acceleration(q, dq, ee["a"], ee["alpha"]) - ddq).max()
    out["sigma_min"] = vk.singular_values(q)[-1]
    return out


class Panel:
    """
    The small live panel in the viewer: one line per check, OK / FAIL, and the
    worst error seen so far (so the numbers stay still instead of flickering).
    """

    TOL = {"pos": 1e-6, "vel": 1e-6, "acc": 1e-5, "inv": 1e-5}

    def __init__(self):
        self.worst = dict.fromkeys(self.TOL, 0.0)

    def update(self, ee, check):
        w = self.worst
        w["pos"] = max(w["pos"], check["fk_pos"])
        w["vel"] = max(w["vel"], check["fv_lin"], check["fv_ang"])
        w["acc"] = max(w["acc"], check["fa_lin"], check["fa_ang"])
        if check["sigma_min"] >= vk.SINGULAR_EPS:          # inverse is damped right at a singularity
            w["inv"] = max(w["inv"], check["ik"], check["iv"], check["ia"])

        def status(key):
            return "OK" if w[key] < self.TOL[key] else "FAIL"

        near = check["sigma_min"] < vk.SINGULAR_EPS
        left = "\n".join([
            "EQUATIONS vs SENSORS",
            "position   FK(q)",
            "velocity   J q_dot",
            "accel.     J q_ddot + J_dot q_dot",
            "inverse    IK, J^-1",
            "flange speed",
            "singularity",
        ])
        right = "\n".join([
            "worst error so far",
            f"{status('pos')}   {w['pos']:.0e} m",
            f"{status('vel')}   {w['vel']:.0e} m/s",
            f"{status('acc')}   {w['acc']:.0e} m/s2",
            f"{status('inv')}   {w['inv']:.0e} rad",
            f"{np.linalg.norm(ee['v']):.2f} m/s",
            ("NEAR  " if near else "far   ") + f"(sigma_min {check['sigma_min']:.2f})",
        ])
        return left, right
