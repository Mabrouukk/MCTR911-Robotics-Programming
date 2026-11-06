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


def panel_text(ee, check):
    """Two text columns for the viewer overlay."""
    p, v, w, a, alpha = check["pred"]
    f = lambda x: "[" + " ".join(f"{c:+.3f}" for c in x) + "]"  # noqa: E731
    left = "\n".join([
        "KINEMATICS CHECK (live)", "",
        "flange position  sensed", "            FK(q)",
        "linear velocity  sensed", "            J q_dot",
        "angular velocity sensed", "            J q_dot",
        "linear accel.    sensed", "            J q_ddot + J_dot q_dot",
        "", "forward errors  pos / vel / acc",
        "inverse errors  IK / J^-1 v / J^-1 a",
        "sigma_min(J)",
    ])
    right = "\n".join([
        "", "",
        f"{f(ee['p'])} m", f"{f(p)} m",
        f"{f(ee['v'])} m/s", f"{f(v)} m/s",
        f"{f(ee['w'])} rad/s", f"{f(w)} rad/s",
        f"{f(ee['a'])} m/s2", f"{f(a)} m/s2",
        "",
        f"{check['fk_pos']:.0e} m / {check['fv_lin']:.0e} m/s / {check['fa_lin']:.0e} m/s2",
        f"{check['ik']:.0e} / {check['iv']:.0e} / {check['ia']:.0e} rad(/s)(/s2)",
        f"{check['sigma_min']:.3f} m/rad" + ("   NEAR SINGULAR" if check["sigma_min"] < vk.SINGULAR_EPS else ""),
    ])
    return left, right
