#!/usr/bin/env python3
"""
Milestone 3 validation, without ROS 2: our kinematic equations vs the MuJoCo simulator.

The brief: "Test the equations of the Forward and Inverse Position and Velocity
Kinematics on the chosen simulator by inputting joint angles / angular
velocities / angular accelerations and sensing the position / velocities /
accelerations of the end effector, and the inverse process, to be the same as
that obtained by the kinematic equations with simple tolerance."

  1. Input joint values straight into the simulator (1000 random states):
     sensed flange pose / twist / acceleration vs FK, J q_dot, J q_ddot + J_dot q_dot,
     and the inverse: IK, J^-1 x_dot, J^-1 (x_ddot - J_dot q_dot) from the sensed
     flange values vs the joint values we put in.
  2. Same, while the physics runs: the servos move the arm along joint-space
     sine waves, and we compare at every 10 ms.
  3. Inverse kinematics driving the simulator: a flange path in task space
     (straight line, then a circle, gripper pointing down) is turned into joint
     angles / rates / accelerations with IK / J^-1 / J^-1(...), sent to the
     servos, and the sensed flange motion is compared with the planned one.
  4. Near a singularity: what the inverse velocity does as the wrist lines up.
  5. Figures and a results file for the report.

    .venv/bin/python "Milestone 03/code/validate_kinematics.py"      (about 20 s)
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import velocity_kinematics as vk  # noqa: E402
from ee_sensors import read_ee  # noqa: E402
from kinematics import HOME_Q  # noqa: E402
from kinematic_checks import kinematics_check  # noqa: E402
from sim_bridge import Arm, build_model  # noqa: E402
from sorting_task import TOOL_DOWN  # noqa: E402
from task_path import T_TOTAL, flange_target, plan_joint_path  # noqa: E402

REPORT = os.path.join(HERE, "..", "report")
FIGURES = os.path.join(REPORT, "figures")

# "Simple tolerance" for equations vs sensors (the simulator computes the same
# quantities its own way, so anything above round-off would mean a mistake).
TOL = {"pos": 1e-6, "rot": 1e-6, "vel": 1e-6, "acc": 1e-5, "joint": 1e-6}
# Tolerance for the flange actually following a task-space path (servo tracking).
TRACK_TOL = 0.005    # 5 mm

failures = []
results = {}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


model = build_model()
data = mujoco.MjData(model)
arm = Arm(model)
mujoco.mj_resetData(model, data)
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
print("\n1. Joint values put straight into the simulator (1000 random states)")
print("   q in [-pi, pi], q_dot in [-pi, pi] rad/s (UR5e max 180 deg/s), q_ddot in [-10, 10] rad/s^2")
keys = ["fk_pos", "fk_rot", "fv_lin", "fv_ang", "fa_lin", "fa_ang", "ik", "iv", "ia"]
worst = dict.fromkeys(keys, 0.0)
near_singular = 0
for _ in range(1000):
    q = rng.uniform(-np.pi, np.pi, 6)
    dq = rng.uniform(-np.pi, np.pi, 6)
    ddq = rng.uniform(-10, 10, 6)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.qpos[arm.qpos], data.qvel[arm.dof], data.qacc[arm.dof] = q, dq, ddq
    mujoco.mj_inverse(model, data)      # kinematics + sensors for exactly these q, q_dot, q_ddot
    c = kinematics_check(q, dq, ddq, read_ee(model, data))
    for k in keys[:6]:
        worst[k] = max(worst[k], c[k])
    if c["sigma_min"] < vk.SINGULAR_EPS:
        near_singular += 1          # inverse not unique / damped there: checked in part 4
        continue
    for k in keys[6:]:
        worst[k] = max(worst[k], c[k])
check("forward position: FK(q) = sensed flange position", worst["fk_pos"] < TOL["pos"], f"worst {worst['fk_pos']:.1e} m")
check("forward position: FK(q) = sensed flange orientation", worst["fk_rot"] < TOL["rot"], f"worst {worst['fk_rot']:.1e} rad")
check("forward velocity: J q_dot = sensed linear velocity", worst["fv_lin"] < TOL["vel"], f"worst {worst['fv_lin']:.1e} m/s")
check("forward velocity: J q_dot = sensed angular velocity", worst["fv_ang"] < TOL["vel"], f"worst {worst['fv_ang']:.1e} rad/s")
check("forward accel.: J q_ddot + J_dot q_dot = sensed linear acc.", worst["fa_lin"] < TOL["acc"], f"worst {worst['fa_lin']:.1e} m/s^2")
check("forward accel.: J q_ddot + J_dot q_dot = sensed angular acc.", worst["fa_ang"] < TOL["acc"], f"worst {worst['fa_ang']:.1e} rad/s^2")
check("inverse position: IK(sensed pose) gives back q", worst["ik"] < TOL["joint"], f"worst {worst['ik']:.1e} rad")
check("inverse velocity: J^-1 (sensed twist) gives back q_dot", worst["iv"] < TOL["joint"], f"worst {worst['iv']:.1e} rad/s")
check("inverse accel.: J^-1 (sensed acc. - J_dot q_dot) gives back q_ddot", worst["ia"] < 10 * TOL["joint"],
      f"worst {worst['ia']:.1e} rad/s^2")
print(f"         {near_singular} of 1000 states were within sigma_min < {vk.SINGULAR_EPS} of a singularity "
      f"(left out of the inverse checks, see part 4)")
results["random_states"] = {k: float(v) for k, v in worst.items()} | {"near_singular": near_singular, "n": 1000}


# ---------------------------------------------------------------------------
print("\n2. While the physics runs: servos follow joint sine waves, compared every 10 ms")
mujoco.mj_resetData(model, data)
data.qpos[arm.qpos] = HOME_Q
data.ctrl[arm.act] = HOME_Q
mujoco.mj_forward(model, data)
AMP = np.radians([40, 25, 30, 40, 40, 60])
FREQ = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
log = {k: [] for k in ("t", "v_s", "v_e", "w_s", "w_e", "a_s", "a_e", "dq_s", "dq_i", "err")}
worst = dict.fromkeys(keys, 0.0)
steps_per_sample = round(0.01 / model.opt.timestep)
for k in range(int(8.0 / model.opt.timestep)):
    t = data.time
    data.ctrl[arm.act] = HOME_Q + AMP * np.sin(2 * np.pi * FREQ * t)
    mujoco.mj_step(model, data)
    if k % steps_per_sample == 0:
        mujoco.mj_forward(model, data)          # joints and sensors from the same instant
        q, dq, ddq = arm.state(data)
        ee = read_ee(model, data)
        c = kinematics_check(q, dq, ddq, ee)
        for key in keys:
            worst[key] = max(worst[key], c[key])
        _, v, w, a, _ = c["pred"]
        for key, val in (("t", data.time), ("v_s", ee["v"]), ("v_e", v), ("w_s", ee["w"]), ("w_e", w),
                         ("a_s", ee["a"]), ("a_e", a), ("dq_s", dq),
                         ("dq_i", vk.inverse_velocity(q, ee["v"], ee["w"])),
                         ("err", [c["fv_lin"], c["fa_lin"], c["iv"], c["ia"]])):
            log[key].append(val)
log = {k: np.array(v) for k, v in log.items()}
check("forward position / velocity / acceleration while moving",
      worst["fk_pos"] < TOL["pos"] and worst["fv_lin"] < TOL["vel"] and worst["fv_ang"] < TOL["vel"]
      and worst["fa_lin"] < TOL["acc"] and worst["fa_ang"] < TOL["acc"],
      f"worst {worst['fk_pos']:.0e} m, {max(worst['fv_lin'], worst['fv_ang']):.0e} m/s, "
      f"{max(worst['fa_lin'], worst['fa_ang']):.0e} m/s^2")
check("inverse position / velocity / acceleration while moving",
      worst["ik"] < TOL["joint"] and worst["iv"] < TOL["joint"] and worst["ia"] < 10 * TOL["joint"],
      f"worst {worst['ik']:.0e} rad, {worst['iv']:.0e} rad/s, {worst['ia']:.0e} rad/s^2")
print(f"         peak flange speed {np.linalg.norm(log['v_s'], axis=1).max():.2f} m/s, "
      f"peak flange acceleration {np.linalg.norm(log['a_s'], axis=1).max():.2f} m/s^2")
results["physics_run"] = {k: float(v) for k, v in worst.items()}


# ---------------------------------------------------------------------------
print("\n3. Inverse kinematics driving the simulator: a task-space path for the flange")


# Inverse position / velocity / acceleration along the path, every 2 ms (task_path.py).
path = plan_joint_path(model.opt.timestep)
t_path, q_path, dq_path, ddq_path = path["t"], path["q"], path["dq"], path["ddq"]
p_path = path["p"]

# The three inverse results must be consistent with each other: the rates from
# J^-1 are the time derivative of the IK angles, and so on.
dq_num = np.gradient(q_path, t_path, axis=0)
ddq_num = np.gradient(dq_path, t_path, axis=0)
inner = slice(2, -2)
check("d/dt IK(p(t)) = J^-1 v(t)  (inverse velocity matches inverse position)",
      np.abs(dq_num - dq_path)[inner].max() < 1e-4, f"max diff {np.abs(dq_num - dq_path)[inner].max():.1e} rad/s")
check("d/dt J^-1 v(t) = J^-1 (a - J_dot q_dot)  (inverse accel. matches inverse velocity)",
      np.abs(ddq_num - ddq_path)[inner].max() < 1e-2, f"max diff {np.abs(ddq_num - ddq_path)[inner].max():.1e} rad/s^2")
sig = np.array([vk.singular_values(q)[-1] for q in q_path[::50]])
print(f"         path stays away from singularities: smallest sigma_min {sig.min():.3f} m/rad")

KV_OVER_KP = model.actuator_biasprm[arm.act, 2] / model.actuator_biasprm[arm.act, 1]


def track(feedforward):
    """Send the joint path to the servos; return sensed flange p, v, a and orientation error."""
    mujoco.mj_resetData(model, data)
    data.qpos[arm.qpos] = q_path[0]
    data.ctrl[arm.act] = q_path[0]
    mujoco.mj_forward(model, data)
    out = {k: [] for k in ("t", "p", "v", "a", "rot")}
    for k in range(len(t_path) + int(1.0 / model.opt.timestep)):
        i = min(k, len(t_path) - 1)
        moving = k < len(t_path)
        # Position servo: torque = kp (ctrl - q) - kd q_dot. Commanding ctrl = q_d
        # leaves a lag of (kd/kp) q_dot_d. Adding (kd/kp) q_dot_d cancels it
        # (velocity feed-forward); q_dot_d comes from our inverse velocity.
        ff = KV_OVER_KP * dq_path[i] if (feedforward and moving) else 0.0
        data.ctrl[arm.act] = q_path[i] + ff
        mujoco.mj_step(model, data)
        if k % 5 == 0:
            mujoco.mj_forward(model, data)
            ee = read_ee(model, data)
            out["t"].append(data.time)
            out["p"].append(ee["p"]), out["v"].append(ee["v"]), out["a"].append(ee["a"])
            out["rot"].append(vk.rotation_error(TOOL_DOWN, ee["R"]))
    out = {k: np.array(v) for k, v in out.items()}
    # Planned values at the same instants (held at the end after T_TOTAL).
    planned = [flange_target(t) for t in out["t"]]
    out["p_ref"] = np.array([x[0] for x in planned])
    out["v_ref"] = np.array([x[1] if t <= T_TOTAL else np.zeros(3) for x, t in zip(planned, out["t"])])
    out["a_ref"] = np.array([x[2] if t <= T_TOTAL else np.zeros(3) for x, t in zip(planned, out["t"])])
    out["p_err"] = np.linalg.norm(out["p"] - out["p_ref"], axis=1)
    out["v_err"] = np.linalg.norm(out["v"] - out["v_ref"], axis=1)
    return out


plain = track(feedforward=False)
ff = track(feedforward=True)
for name, run in (("angles only", plain), ("angles + velocity feed-forward", ff)):
    print(f"         {name:32s} worst position error {run['p_err'].max() * 1000:6.2f} mm, "
          f"worst velocity error {run['v_err'].max() * 1000:6.1f} mm/s, "
          f"worst tilt {np.degrees(run['rot'].max()):.2f} deg, final error {run['p_err'][-1] * 1000:.2f} mm")
check(f"flange follows the planned path within {TRACK_TOL * 1000:.0f} mm (with velocity feed-forward)",
      ff["p_err"].max() < TRACK_TOL, f"worst {ff['p_err'].max() * 1000:.2f} mm")
check("flange ends on the planned point (< 1 mm)", ff["p_err"][-1] < 0.001, f"{ff['p_err'][-1] * 1000:.3f} mm")
results["task_space"] = {
    name: {"max_pos_mm": float(r["p_err"].max() * 1000), "max_vel_mm_s": float(r["v_err"].max() * 1000),
           "max_tilt_deg": float(np.degrees(r["rot"].max())), "final_mm": float(r["p_err"][-1] * 1000)}
    for name, r in (("plain", plain), ("feedforward", ff))}
results["task_space"]["min_sigma"] = float(sig.min())


# ---------------------------------------------------------------------------
print("\n4. Inverse velocity near the wrist singularity (q5 -> 0)")
q5_values = np.radians(np.logspace(np.log10(30), np.log10(0.05), 60))
x_dot = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])     # 5 cm/s along x, no rotation
sing = {k: [] for k in ("q5", "sigma", "plain", "dls", "dls_err")}
for q5 in q5_values:
    q = HOME_Q.copy()
    q[4] = -q5
    J = vk.jacobian(q)
    plain_dq = np.linalg.solve(J, x_dot)
    dls_dq = vk.inverse_velocity(q, x_dot[:3], x_dot[3:])
    sing["q5"].append(np.degrees(q5))
    sing["sigma"].append(vk.singular_values(q)[-1])
    sing["plain"].append(np.abs(plain_dq).max())
    sing["dls"].append(np.abs(dls_dq).max())
    sing["dls_err"].append(np.linalg.norm(J @ dls_dq - x_dot))
sing = {k: np.array(v) for k, v in sing.items()}
print(f"         at q5 = {sing['q5'][-1]:.2f} deg: plain J^-1 asks {sing['plain'][-1]:.0f} rad/s, "
      f"damped asks {sing['dls'][-1]:.2f} rad/s (UR5e limit pi rad/s)")
check("damped inverse keeps the joint rates bounded at the singularity", sing["dls"].max() < np.pi,
      f"max {sing['dls'].max():.2f} rad/s")
check("away from the singularity the damped inverse is exact", sing["dls_err"][sing["sigma"] > vk.SINGULAR_EPS].max() < 1e-12)
results["singularity"] = {"q5_deg": float(sing["q5"][-1]), "plain": float(sing["plain"][-1]),
                          "dls": float(sing["dls"][-1]), "dls_twist_err": float(sing["dls_err"][-1])}


# ---------------------------------------------------------------------------
if "--no-plots" not in sys.argv:
    print("\n5. Figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference palette (dataviz skill), validated; fixed order.
    S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0, "figure.facecolor": "white",
    })
    os.makedirs(FIGURES, exist_ok=True)

    # Forward: sensed (lines) vs equations (markers), three components.
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    every = 15
    for ax, (s, e, unit, title) in zip(axes, (
            ("v_s", "v_e", "m/s", "Flange linear velocity: sensed (line) vs J q̇ (dots)"),
            ("w_s", "w_e", "rad/s", "Flange angular velocity: sensed (line) vs J q̇ (dots)"),
            ("a_s", "a_e", "m/s²", "Flange linear acceleration: sensed (line) vs J q̈ + J̇ q̇ (dots)"))):
        for j, comp in enumerate("xyz"):
            ax.plot(log["t"], log[s][:, j], color=S[j], label=comp)
            ax.plot(log["t"][::every], log[e][::every, j], "o", color=S[j], markersize=4,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left")
    axes[0].legend(ncol=3, loc="upper right")
    axes[2].set_xlabel("time (s)")
    fig.text(0.01, 0.005, f"Servos follow joint sine waves. Worst difference: velocity "
             f"{results['physics_run']['fv_lin']:.0e} m/s, acceleration {results['physics_run']['fa_lin']:.0e} m/s² "
             "(floating-point round-off).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(FIGURES, "validation_forward.png"), dpi=150)
    plt.close(fig)

    # Inverse: sensed joint rates vs J^-1 (sensed twist).
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j in range(6):
        ax.plot(log["t"], log["dq_s"][:, j], color=S[j], label=f"q̇{j + 1}")
        ax.plot(log["t"][::every], log["dq_i"][::every, j], "o", color=S[j], markersize=4,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rad/s")
    ax.set_title("Joint rates: sensed (line) vs J⁻¹ × sensed flange twist (dots)", loc="left")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "validation_inverse.png"), dpi=150)
    plt.close(fig)

    # Task-space tracking: top view + error over time.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(p_path[:, 0], p_path[:, 1], color=MUTED, linewidth=6, alpha=0.35, label="planned")
    ax.plot(plain["p"][:, 0], plain["p"][:, 1], color=S[1], linewidth=1.5, label="angles only")
    ax.plot(ff["p"][:, 0], ff["p"][:, 1], color=S[0], linewidth=1.5, label="+ velocity feed-forward")
    ax.set_aspect("equal")
    ax.set_xlabel("x in base frame (m)")
    ax.set_ylabel("y in base frame (m)")
    ax.set_title("Flange path, top view", loc="left")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.plot(plain["t"], plain["p_err"] * 1000, color=S[1], label="angles only")
    ax.plot(ff["t"], ff["p_err"] * 1000, color=S[0], label="+ velocity feed-forward")
    ax.axhline(TRACK_TOL * 1000, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"{TRACK_TOL * 1000:.0f} mm tolerance", (0.1, TRACK_TOL * 1000), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color=MUTED)
    ax.axvline(T_TOTAL, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mm")
    ax.set_ylim(0, None)
    ax.set_title("Distance between sensed and planned flange", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "validation_task_space.png"), dpi=150)
    plt.close(fig)

    # Singularity.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.loglog(sing["q5"], sing["plain"], color=S[1], label="plain J⁻¹")
    ax.loglog(sing["q5"], sing["dls"], color=S[0], label="damped least squares")
    ax.axhline(np.pi, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.02, np.pi * 1.15, "UR5e joint speed limit (π rad/s)", transform=ax.get_yaxis_transform(),
            fontsize=8, color=MUTED, va="bottom")
    ax.invert_xaxis()
    ax.set_xlabel("|q5| (deg), wrist lining up →")
    ax.set_ylabel("largest joint rate (rad/s)")
    ax.set_title("Joint rates for a 5 cm/s flange motion", loc="left")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.semilogx(sing["q5"], sing["sigma"], color=S[0])
    ax.axhline(vk.SINGULAR_EPS, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.98, vk.SINGULAR_EPS + 0.003, "damping switches on below this", transform=ax.get_yaxis_transform(),
            fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.invert_xaxis()
    ax.set_xlabel("|q5| (deg)")
    ax.set_ylabel("m/rad")
    ax.set_title("Smallest singular value of J", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "validation_singularity.png"), dpi=150)
    plt.close(fig)
    print("  saved validation_forward.png, validation_inverse.png, validation_task_space.png, "
          "validation_singularity.png")

with open(os.path.join(REPORT, "validation_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
print("All checks passed.")
