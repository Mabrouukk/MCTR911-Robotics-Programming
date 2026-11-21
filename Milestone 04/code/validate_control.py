"""
Milestone 4 offline validation (no ROS 2): dynamics, trajectories and control, plus the report figures.

    .venv/bin/python "Milestone 04/code/validate_control.py"      (about 3 minutes)

Test 1  Lagrangian dynamics vs MuJoCo, 1000 random states: M(q), C q_dot + g, M q_ddot + C q_dot + g,
        dM/dq (closed form vs finite differences), the property M_dot - 2C skew-symmetric.
Test 2  Trajectories: quintic boundary conditions, speed / acceleration limits, task-space segments
        (FK, J q_dot, J q_ddot + J_dot q_dot of the planned joints give back the planned flange motion).
Test 3  Closed loop in MuJoCo (torque mode, gravity on): PD, PD + gravity, computed torque, on the
        joint-space and the task-space test trajectories.
Test 4  Robustness of the model-based controllers: a hidden 2 kg payload, the upper arm and forearm
        20 % heavier than the model, and a ROS-like loop (250 Hz, one period of delay).
Test 5  The colour-sorting application with computed torque: all six parts, tracking while moving
        and during the grasps.
Results -> report/validation_results.json, figures -> report/figures/.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control  # noqa: E402
import dynamics as dyn  # noqa: E402
import test_paths  # noqa: E402
import trajectories as tr  # noqa: E402
import velocity_kinematics as vk  # noqa: E402
from kinematics import HOME_Q, forward_kinematics  # noqa: E402
from sim_bridge import ARM_JOINTS, Arm, Conveyor, build_model  # noqa: E402
from sorting_task import SortingApp  # noqa: E402

REPORT = os.path.join(HERE, "..", "report")
FIGS = os.path.join(REPORT, "figures")
CAD = os.path.join(HERE, "..", "..", "Milestone 01", "cad_models")

# One colour per controller, the same in every figure; the plan is grey and dashed.
COLORS = {"ctc": "#2a78d6", "pdg": "#eb6834", "pd": "#1baf7a", "ctci": "#eda100"}
NAMES = {"ctc": "computed torque", "pdg": "PD + gravity comp.", "pd": "PD only", "ctci": "computed torque + integral"}
PLAN_COLOR, INK, MUTED, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.8, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False,
})
RESULTS = {}


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 1: dynamics
# ---------------------------------------------------------------------------
def clean_model():
    """UR5e + 2F-85 only (no cell, no contacts): the reference for the dynamics check."""
    arm = mujoco.MjSpec.from_file(os.path.join(CAD, "ur5e", "ur5e.xml"))
    gripper = mujoco.MjSpec.from_file(os.path.join(CAD, "robotiq_2f85", "2f85.xml"))
    arm.option.cone, arm.option.impratio = gripper.option.cone, gripper.option.impratio
    arm.attach(gripper, prefix="gripper_", site=arm.site("attachment_site"))
    model = arm.compile()
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return model


def test_dynamics():
    model = clean_model()
    data = mujoco.MjData(model)
    dof = [model.joint(j).dofadr[0] for j in ARM_JOINTS]
    qadr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
    rng = np.random.default_rng(4)
    worst = dict.fromkeys(("M", "bias", "tau", "gravity", "dMdq", "skew"), 0.0)
    biggest_tau = 0.0
    t_ours = []
    for _ in range(1000):
        q, dq, ddq = rng.uniform(-np.pi, np.pi, 6), rng.uniform(-np.pi, np.pi, 6), rng.uniform(-10, 10, 6)
        data.qpos[:], data.qvel[:], data.qacc[:] = model.qpos0, 0, 0
        data.qpos[qadr], data.qvel[dof], data.qacc[dof] = q, dq, ddq
        mujoco.mj_inverse(model, data)
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, full)
        c0 = time.perf_counter()
        tau = dyn.NOMINAL.inverse_dynamics(q, dq, ddq)
        t_ours.append(time.perf_counter() - c0)
        worst["M"] = max(worst["M"], np.abs(full[np.ix_(dof, dof)] - dyn.NOMINAL.mass_matrix(q)).max())
        worst["bias"] = max(worst["bias"], np.abs(data.qfrc_bias[dof] - dyn.NOMINAL.bias(q, dq)).max())
        worst["tau"] = max(worst["tau"], np.abs(data.qfrc_inverse[dof] - tau).max())
        biggest_tau = max(biggest_tau, np.abs(tau).max())
        # gravity alone: MuJoCo's bias force at zero velocity
        data.qvel[dof] = 0
        mujoco.mj_inverse(model, data)
        worst["gravity"] = max(worst["gravity"], np.abs(data.qfrc_bias[dof] - dyn.NOMINAL.gravity(q)).max())
    for _ in range(100):
        q, dq = rng.uniform(-np.pi, np.pi, 6), rng.uniform(-np.pi, np.pi, 6)
        D = dyn.NOMINAL.mass_matrix_derivatives(q)
        for k in range(6):
            e = np.zeros(6)
            e[k] = 1e-6
            fd = (dyn.NOMINAL.mass_matrix(q + e) - dyn.NOMINAL.mass_matrix(q - e)) / 2e-6
            worst["dMdq"] = max(worst["dMdq"], np.abs(fd - D[k]).max())
        N = np.einsum("kij,k->ij", D, dq) - 2 * dyn.NOMINAL.coriolis_matrix(q, dq)
        worst["skew"] = max(worst["skew"], np.abs(N + N.T).max())
    RESULTS["dynamics"] = {k: float(v) for k, v in worst.items()} | {
        "largest_torque_tested_Nm": float(biggest_tau), "ms_per_inverse_dynamics": 1000 * float(np.median(t_ours)),
        "arm_mass_kg": float(sum(m for m, _, _ in dyn.LINKS)),
    }
    print("Test 1  dynamics vs MuJoCo (1000 random states):")
    print(f"        M(q) {worst['M']:.1e}   C q_dot + g {worst['bias']:.1e}   g(q) {worst['gravity']:.1e}   "
          f"tau {worst['tau']:.1e} N m (torques up to {biggest_tau:.0f} N m)")
    print(f"        dM/dq vs finite differences {worst['dMdq']:.1e}   M_dot - 2C skew-symmetric to {worst['skew']:.1e}")

    # Figure: gravity torque of the shoulder and elbow as the shoulder turns, ours vs MuJoCo.
    q2s = np.linspace(-np.pi, 0, 60)
    q_base = np.array(test_paths.WAIT_Q)
    ours, theirs = [], []
    for q2 in q2s:
        q = q_base.copy()
        q[1] = q2
        data.qpos[:], data.qvel[:], data.qacc[:] = model.qpos0, 0, 0
        data.qpos[qadr] = q
        mujoco.mj_inverse(model, data)
        ours.append(dyn.NOMINAL.gravity(q))
        theirs.append(data.qfrc_bias[dof].copy())
    ours, theirs = np.array(ours), np.array(theirs)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for j, color in ((1, COLORS["ctc"]), (2, COLORS["pdg"])):
        ax.plot(np.degrees(q2s), ours[:, j], color=color, label=f"g{j + 1}(q), our model")
        ax.plot(np.degrees(q2s[::4]), theirs[::4, j], "o", ms=5, mfc="none", mec=INK, mew=1.0)
    ax.plot([], [], "o", ms=5, mfc="none", mec=INK, label="MuJoCo")
    ax.set_xlabel("shoulder angle q2 [deg] (others at the waiting pose)")
    ax.set_ylabel("torque to hold the arm [N m]")
    ax.set_title("Gravity vector g(q): shoulder and elbow")
    ax.legend()
    save(fig, "dynamics_gravity.png")


# ---------------------------------------------------------------------------
# Test 2: trajectories
# ---------------------------------------------------------------------------
def test_trajectories():
    q0, q1 = test_paths.WAIT_Q, test_paths.SWING_Q
    quintic, lspb = tr.JointQuintic(q0, q1), tr.JointLSPB(q0, q1)
    bc = max(np.abs(quintic.sample(0)[0] - q0).max(), np.abs(quintic.sample(quintic.T)[0] - q1).max(),
             np.abs(quintic.sample(0)[1]).max(), np.abs(quintic.sample(quintic.T)[1]).max(),
             np.abs(quintic.sample(0)[2]).max(), np.abs(quintic.sample(quintic.T)[2]).max())
    # The 6x6 system for the coefficients gives the same curve.
    coeff_err = 0.0
    for j in range(6):
        c = tr.quintic_coefficients(q0[j], q1[j], quintic.T)
        for t in np.linspace(0, quintic.T, 50):
            coeff_err = max(coeff_err, abs(sum(c[k] * t**k for k in range(6)) - quintic.sample(t)[0][j]))
    peaks = {}
    for name, seg in (("quintic", quintic), ("lspb", lspb)):
        ts = np.linspace(0, seg.T, 2000)
        s = np.array([seg.sample(t) for t in ts])
        peaks[name] = {"T": seg.T, "vel_ratio": float((np.abs(s[:, 1]).max(axis=0) / tr.QD_MAX).max()),
                       "acc_ratio": float((np.abs(s[:, 2]).max(axis=0) / tr.QDD_MAX).max())}
    # Task-space segments: our forward kinematics of the planned joints = the planned flange motion.
    task_err = 0.0
    for seg, _, _ in test_paths.task_test_plan().steps:
        if not isinstance(seg, tr.TaskSegment):
            continue
        for t in np.linspace(0, seg.T, 200):
            q, dq, ddq = seg.sample(t)
            p, R, v, w, a, alpha = seg.flange(t)
            T = forward_kinematics(q)
            vv, ww = vk.forward_velocity(q, dq)
            aa, al = vk.forward_acceleration(q, dq, ddq)
            task_err = max(task_err, np.linalg.norm(T[:3, 3] - p), vk.rotation_error(R, T[:3, :3]),
                           np.linalg.norm(vv - v), np.linalg.norm(ww - w), np.linalg.norm(aa - a), np.linalg.norm(al - alpha))
    RESULTS["trajectories"] = {"quintic_boundary_error": float(bc), "coefficients_vs_profile": float(coeff_err),
                               "task_space_consistency": float(task_err), "swing_move": peaks}
    print("Test 2  trajectories:")
    print(f"        quintic boundary conditions {bc:.1e}, 6x6 coefficient solve vs closed form {coeff_err:.1e}")
    print(f"        same move: quintic T = {quintic.T:.2f} s, LSPB T = {lspb.T:.2f} s; "
          f"peak speed {peaks['quintic']['vel_ratio']:.0%} / {peaks['lspb']['vel_ratio']:.0%} of the limit, "
          f"peak accel. {peaks['quintic']['acc_ratio']:.0%} / {peaks['lspb']['acc_ratio']:.0%}")
    print(f"        task space: FK / J q_dot / J q_ddot + J_dot q_dot of the planned joints vs planned flange {task_err:.1e}")

    # Figure: quintic vs LSPB, same move, same duration, normalised.
    T = 2.0
    qq, ql = tr.JointQuintic(np.zeros(6), np.ones(6), T), tr.JointLSPB(np.zeros(6), np.ones(6), T)
    ts = np.linspace(0, T, 400)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for k, (ax, title) in enumerate(zip(axes, ("position  s(t)", "velocity  s'(t) [1/s]", "acceleration  s''(t) [1/s²]"))):
        ax.plot(ts, [qq.sample(t)[k][0] for t in ts], color=COLORS["ctc"], label="quintic")
        ax.plot(ts, [ql.sample(t)[k][0] for t in ts], color=COLORS["pdg"], label="LSPB (trapezoid)")
        ax.set_title(title)
        ax.set_xlabel("t [s]")
    axes[0].legend()
    fig.suptitle("Joint-space time laws for a move of 1 rad in 2 s", y=1.03)
    save(fig, "joint_time_laws.png")

    # Figure: flange paths of the two tests (3D).
    fig = plt.figure(figsize=(7.5, 5))
    ax = fig.add_subplot(projection="3d")
    for name, color, label in (("joint", COLORS["pdg"], "joint-space test (flange path = curves)"),
                               ("task", COLORS["ctc"], "task-space test (lines + circle)")):
        plan = test_paths.PLANS[name]()
        p = np.array([forward_kinematics(plan.sample(t)[0])[:3, 3] for t in np.linspace(0, plan.T, 900)])
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, label=label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Planned flange paths (robot base frame)")
    ax.legend(loc="upper left")
    ax.view_init(elev=28, azim=-60)
    save(fig, "flange_paths.png")


# ---------------------------------------------------------------------------
# Closed-loop runs in MuJoCo
# ---------------------------------------------------------------------------
def run_plan(plan, ctrl_name, payload=0.0, mass_scale=None, every=1, delay_steps=0, environment=True):
    """Play a plan with a controller in the torque-mode cell. Returns a log dict."""
    model = build_model(environment=environment, payload_mass=payload, mass_scale=mass_scale)
    data = mujoco.MjData(model)
    arm = Arm(model)
    data.qpos[arm.qpos] = plan.sample(0)[0]
    mujoco.mj_forward(model, data)
    ctrl = control.make(ctrl_name)
    if hasattr(ctrl, "dt"):
        ctrl.dt = model.opt.timestep * every          # integral term: one control period
    pending = [np.zeros(6)] * delay_steps
    tau = ctrl(*arm.state(data)[:2], *plan.sample(0)[:3])
    log = {k: [] for k in ("t", "q", "q_d", "tau", "ff", "seg")}
    n = 0
    while data.time <= plan.T:
        q, dq, _ = arm.state(data)
        q_d, dq_d, ddq_d, gripper, _, seg = plan.sample(data.time)
        if n % every == 0:
            new = ctrl(q, dq, q_d, dq_d, ddq_d)
            pending.append(new)
            tau = pending.pop(0)
        data.ctrl[arm.act] = tau
        data.ctrl[arm.grip] = gripper
        mujoco.mj_step(model, data)
        n += 1
        for k, v in (("t", data.time), ("q", q), ("q_d", q_d), ("tau", tau),
                     ("ff", ctrl.last_parts.get("feedforward", np.zeros(6))), ("seg", seg)):
            log[k].append(v)
    return {k: np.array(v) for k, v in log.items()}


def errors(log):
    e = np.degrees(np.abs(log["q_d"] - log["q"]))
    f = 1000 * np.array([np.linalg.norm(forward_kinematics(a)[:3, 3] - forward_kinematics(b)[:3, 3])
                         for a, b in zip(log["q_d"], log["q"])])
    return e, f


def summary(log):
    e, f = errors(log)
    return {"joint_err_max_deg": float(e.max()), "joint_err_rms_deg": float(np.sqrt(np.mean(e**2))),
            "flange_err_max_mm": float(f.max()), "flange_err_rms_mm": float(np.sqrt(np.mean(f**2))),
            "final_joint_err_deg": float(e[-1].max()), "final_flange_err_mm": float(f[-1]),
            "peak_torque_Nm": np.abs(log["tau"]).max(axis=0).round(2).tolist()}


def test_controllers():
    print("Test 3  closed loop in MuJoCo, torque mode, gravity on (control every 2 ms):")
    RESULTS["tracking"] = {}
    logs = {}
    for which in ("joint", "task"):
        RESULTS["tracking"][which] = {}
        for name in ("pd", "pdg", "ctc"):
            log = run_plan(test_paths.PLANS[which](), name)
            logs[(which, name)] = log
            s = summary(log)
            RESULTS["tracking"][which][name] = s
            print(f"        {which:5s} {NAMES[name]:20s} joint error max {s['joint_err_max_deg']:.3f} deg, "
                  f"flange error max {s['flange_err_max_mm']:6.2f} mm, at the end {s['final_flange_err_mm']:.3f} mm")

    # Figure: flange error vs time, both tests, log scale (the three controllers differ by 10x steps).
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharey=True)
    for ax, which in zip(axes, ("joint", "task")):
        for name in ("pd", "pdg", "ctc"):
            log = logs[(which, name)]
            _, f = errors(log)
            ax.plot(log["t"], np.maximum(f, 1e-3), color=COLORS[name], lw=1.4, label=NAMES[name])
        ax.set_yscale("log")
        ax.set_ylabel("flange error [mm]")
        ax.set_title(f"{which}-space test trajectory")
    axes[1].set_xlabel("t [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, "tracking_error.png")

    # Figure: desired vs actual, shoulder lift joint, the three controllers (joint test).
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    log = logs[("joint", "ctc")]
    axes[0].plot(log["t"], np.degrees(log["q_d"][:, 1]), "--", color=PLAN_COLOR, lw=2.2, label="planned q2")
    for name in ("pd", "pdg", "ctc"):
        lg = logs[("joint", name)]
        axes[0].plot(lg["t"], np.degrees(lg["q"][:, 1]), color=COLORS[name], lw=1.3, label=NAMES[name])
        axes[1].plot(lg["t"], np.degrees(lg["q_d"][:, 1] - lg["q"][:, 1]), color=COLORS[name], lw=1.3)
    axes[0].set_title("shoulder lift q2: planned vs actual")
    axes[0].set_ylabel("angle [deg]")
    axes[1].set_title("q2 error  (q2 planned - q2 actual)")
    axes[1].set_ylabel("error [deg]")
    for ax in axes:
        ax.set_xlabel("t [s]")
    axes[0].legend(fontsize=8)
    save(fig, "desired_vs_actual_q2.png")

    # Figure: computed torque on the shoulder, split into model (feedforward) and feedback.
    log = logs[("joint", "ctc")]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(log["t"], log["tau"][:, 1], color=INK, lw=2.4, label="total torque, joint 2")
    ax.plot(log["t"], log["ff"][:, 1], "--", color=COLORS["ctc"], lw=1.6, label="model part  M q̈_d + C q̇ + g")
    ax.set_ylabel("torque [N m]")
    ax.set_title("Computed torque on the shoulder: the model does the heavy lifting")
    ax.legend(loc="lower right", fontsize=8)
    ax2.plot(log["t"], log["tau"][:, 1] - log["ff"][:, 1], color=COLORS["pdg"], lw=1.4)
    ax2.set_title("feedback part  M (Kd ė + Kp e)  (note the scale)", fontsize=10)
    ax2.set_ylabel("torque [N m]")
    ax2.set_xlabel("t [s]")
    save(fig, "ctc_torque_split.png")

    # Figure: torque terms along the joint test (what M q_ddot, C q_dot, g each contribute).
    log = logs[("joint", "ctc")]
    plan = test_paths.joint_test_plan()
    terms = []
    for t in log["t"][::5]:
        q, dq, ddq = plan.sample(min(t, plan.T))[:3]
        terms.append((dyn.NOMINAL.mass_matrix(q) @ ddq, dyn.NOMINAL.coriolis_matrix(q, dq) @ dq, dyn.NOMINAL.gravity(q)))
    terms = np.array(terms)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=False)
    for ax, j in zip(axes, (1, 0)):
        ax.plot(log["t"][::5], terms[:, 0, j], color=COLORS["ctc"], label="M(q) q̈  (inertia)")
        ax.plot(log["t"][::5], terms[:, 1, j], color=COLORS["pdg"], label="C(q, q̇) q̇  (Coriolis / centrifugal)")
        ax.plot(log["t"][::5], terms[:, 2, j], color=COLORS["pd"], label="g(q)  (gravity)")
        ax.set_title(f"joint {j + 1} ({'shoulder lift' if j == 1 else 'base'})")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("torque [N m]")
    axes[0].legend(fontsize=8)
    fig.suptitle("The three terms of the equation of motion along the joint-space test", y=1.03)
    save(fig, "dynamics_terms.png")
    RESULTS["torque_terms_joint_test"] = {
        f"joint{j + 1}": {"inertia_max": float(np.abs(terms[:, 0, j]).max()),
                          "coriolis_max": float(np.abs(terms[:, 1, j]).max()),
                          "gravity_max": float(np.abs(terms[:, 2, j]).max())} for j in range(6)}


def test_robustness():
    print("Test 4  robustness (joint-space test):")
    cases = [
        ("nominal", {}),
        ("hidden 2 kg payload", {"payload": 2.0}),
        ("upper arm + forearm 20 % heavier", {"mass_scale": {"upper_arm_link": 1.2, "forearm_link": 1.2}}),
        ("250 Hz loop + 4 ms delay (like ROS 2)", {"every": 2, "delay_steps": 1}),
    ]
    RESULTS["robustness"] = {}
    rows = []
    for label, kw in cases:
        RESULTS["robustness"][label] = {}
        for name in ("pdg", "ctc", "ctci"):
            s = summary(run_plan(test_paths.joint_test_plan(), name, environment=False, **kw))
            RESULTS["robustness"][label][name] = s
            rows.append((label, name, s["flange_err_max_mm"], s["final_flange_err_mm"]))
            print(f"        {label:38s} {NAMES[name]:27s} flange error max {s['flange_err_max_mm']:6.2f} mm, "
                  f"at the end {s['final_flange_err_mm']:.3f} mm")
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    labels = [c[0] for c in cases]
    x = np.arange(len(labels))
    for k, name in enumerate(("pdg", "ctc", "ctci")):
        vals = [r[3] for r in rows if r[1] == name]
        vals = [max(v, 1e-3) for v in vals]
        bars = ax.bar(x + (k - 1) * 0.27, vals, width=0.25, color=COLORS[name], label=NAMES[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.2, "0" if v <= 1e-3 else f"{v:.2g}", ha="center",
                    va="bottom", fontsize=7.5, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x, [lbl.replace(" (", "\n(").replace("+ forearm", "+\nforearm") for lbl in labels], fontsize=8)
    ax.set_ylabel("flange error left at the end [mm]")
    ax.set_title("Model-based controllers when the model is wrong: the error that stays")
    ax.set_ylim(5e-4, 300)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    save(fig, "robustness.png")


def test_sorting(duration=95.0):
    print("Test 5  colour sorting with computed torque (all six parts):")
    model = build_model()
    data = mujoco.MjData(model)
    arm, conveyor = Arm(model), Conveyor(model)
    data.qpos[arm.qpos] = HOME_Q
    mujoco.mj_forward(model, data)
    ctrl, app = control.make("ctc"), SortingApp(HOME_Q)
    rows, events = [], []
    while data.time < duration and sum(app.sorted.values()) < 6:
        q, dq, _ = arm.state(data)
        q_d, dq_d, ddq_d, gripper, event = app.update(data.time, conveyor.station_color(data))
        if event:
            events.append((round(data.time, 2), event))
        data.ctrl[arm.act] = ctrl(q, dq, q_d, dq_d, ddq_d)
        data.ctrl[arm.grip] = gripper
        conveyor.step(data)
        mujoco.mj_step(model, data)
        grasp = any(app.label.startswith(g) for g in ("close the gripper", "lift", "open the gripper"))
        rows.append((data.time, np.degrees(np.abs(q_d - q)).max(),
                     1000 * np.linalg.norm(forward_kinematics(q_d)[:3, 3] - forward_kinematics(q)[:3, 3]), grasp))
    a = np.array(rows, dtype=float)
    moving = (a[:, 3] == 0) & (a[:, 0] > 0.3)
    in_bins = {}
    for b in range(model.nbody):
        name = model.body(b).name
        if name.startswith("env_part_"):
            in_bins[name] = data.xpos[b].round(2).tolist()
    RESULTS["sorting"] = {"sorted": app.sorted, "time_s": float(data.time),
                          "moving_joint_err_max_deg": float(a[moving, 1].max()),
                          "moving_flange_err_max_mm": float(a[moving, 2].max()),
                          "grasp_joint_err_max_deg": float(a[a[:, 3] == 1, 1].max()),
                          "grasp_flange_err_max_mm": float(a[a[:, 3] == 1, 2].max()),
                          "part_positions": in_bins}
    s = RESULTS["sorting"]
    print(f"        sorted {s['sorted']} in {s['time_s']:.0f} s; while moving: flange error max "
          f"{s['moving_flange_err_max_mm']:.2f} mm; during the grasps: {s['grasp_flange_err_max_mm']:.2f} mm")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(a[:, 0], np.maximum(a[:, 2], 1e-3), color=COLORS["ctc"], lw=1.0, label="flange error")
    on = a[:, 3] == 1
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = [a[0, 0]] if on[0] else []
    spans = []
    for i in edges:
        if not on[i]:
            starts.append(a[i + 1, 0])
        elif starts:
            spans.append((starts.pop(), a[i, 0]))
    for k, (s0, s1) in enumerate(spans):
        ax.axvspan(s0, s1, color=PLAN_COLOR, alpha=0.18, lw=0, label="gripper on the part" if k == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("flange error [mm]")
    ax.set_title("Colour sorting with computed torque: tracking error over the whole run", pad=26)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    save(fig, "sorting_error.png")


def main():
    t0 = time.time()
    test_dynamics()
    test_trajectories()
    test_controllers()
    test_robustness()
    test_sorting()
    with open(os.path.join(REPORT, "validation_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"Done in {time.time() - t0:.0f} s. Results: report/validation_results.json, figures: report/figures/")


if __name__ == "__main__":
    main()
