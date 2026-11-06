"""
The task-space test path for the inverse kinematics (Milestone 3).

The flange moves from above the pick station to above the red bin in a
straight line, then draws one horizontal circle, with the gripper pointing
straight down the whole time (so its angular velocity and acceleration are
zero). Quintic timing: every segment starts and ends at rest.

plan_joint_path() turns it into joint motion with all three inverse problems:
    q     = IK(p(t), tool down)                       inverse position
    q_dot = J^-1 [v(t); 0]                            inverse velocity
    q_ddot = J^-1 ([a(t); 0] - J_dot q_dot)           inverse acceleration
"""
import numpy as np

import velocity_kinematics as vk
from kinematics import HOME_Q, ROBOT_BASE_POS
from sorting_task import BIN_XY, CLEAR_Z, STATION_XY, ik_tool_down

P_A = np.array([STATION_XY[0], STATION_XY[1], CLEAR_Z]) - ROBOT_BASE_POS   # above the pick station (base frame)
P_B = np.array([*BIN_XY["red"], CLEAR_Z]) - ROBOT_BASE_POS                 # above the red bin
T_LINE = 3.0
T_CIRCLE = 4.0
RADIUS = 0.08
T_TOTAL = T_LINE + T_CIRCLE


def quintic(s):
    """h(s) and its first two derivatives; h goes 0 -> 1 with zero rate and acceleration at both ends."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 * (1 - s)**2, 60 * s * (1 - s) * (1 - 2 * s)


def flange_target(t):
    """Planned flange position, velocity and acceleration in the base frame at time t."""
    t = min(max(t, 0.0), T_TOTAL)
    if t < T_LINE:
        h, hd, hdd = quintic(t / T_LINE)
        d = P_B - P_A
        return P_A + h * d, hd / T_LINE * d, hdd / T_LINE**2 * d
    h, hd, hdd = quintic((t - T_LINE) / T_CIRCLE)
    th, thd, thdd = 2 * np.pi * h, 2 * np.pi * hd / T_CIRCLE, 2 * np.pi * hdd / T_CIRCLE**2
    centre = P_B - np.array([RADIUS, 0.0, 0.0])
    r = RADIUS * np.array([np.cos(th), np.sin(th), 0.0])
    dr = RADIUS * np.array([-np.sin(th), np.cos(th), 0.0])
    return centre + r, thd * dr, thdd * dr - thd**2 * r


def plan_joint_path(dt):
    """Joint angles, rates and accelerations along the path, every dt seconds."""
    t_path = np.arange(0.0, T_TOTAL + 1e-9, dt)
    out = {k: [] for k in ("q", "dq", "ddq", "p", "v", "a")}
    q_prev = ik_tool_down(P_A + ROBOT_BASE_POS, HOME_Q)
    for t in t_path:
        p, v, a = flange_target(t)
        q = ik_tool_down(p + ROBOT_BASE_POS, q_prev)              # inverse position, nearest solution
        dq = vk.inverse_velocity(q, v, np.zeros(3))                # inverse velocity
        ddq = vk.inverse_acceleration(q, dq, a, np.zeros(3))       # inverse acceleration
        for key, val in zip(("q", "dq", "ddq", "p", "v", "a"), (q, dq, ddq, p, v, a)):
            out[key].append(val)
        q_prev = q
    return {"t": t_path} | {k: np.array(v) for k, v in out.items()}
