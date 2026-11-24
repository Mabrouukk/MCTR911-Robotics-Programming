"""
Challenge 3 (b): the fuzzy controller, studied and plotted.

    .venv/bin/python "Challenge 03/verify/flc_study.py"        (about 1 minute)

Uses the logs of the live ROS 2 runs (results/sim_ros_<mode>.npz, written by panda_sim.py when you
run scripts/run_ch3.sh <mode>) for desired vs actual, and an offline loop (same MuJoCo model,
100 Hz like the template, one control period of delay like the ROS 2 loop) for the design studies.

Figures (results/):
    flc_membership.png      the five sets on the inputs and the output
    flc_surface.png         the control surface: sum-product vs min-max inference
    tracking_desired_vs_actual.png   planned vs actual joints, live through ROS 2
    tracking_error_modes.png         hand error: model + FLC, g + FLC, FLC only, model only
    mass_plus20.png         every link 20 % heavier than the model (Team 5 debate)
    error_scaling.png       how big the errors are that each FLC sees (why the sets were narrowed)
    chatter.png             the first design (M_ii scaling, min-max) vs the final one
and rule_walkthrough.txt: one control step of joint 2, rule by rule.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import panda_sim as ps  # noqa: E402

cn = ps.cn
RESULTS = os.path.join(HERE, "..", "results")
COL = {"MODEL_FLC": "#2a78d6", "GRAVITY_FLC": "#eb6834", "FLC_ONLY": "#1baf7a", "MODEL_ONLY": "#eda100"}
NAME = {"MODEL_FLC": "model + FLC", "GRAVITY_FLC": "g(q) + FLC (Team 5)", "FLC_ONLY": "FLC only", "MODEL_ONLY": "model only (no FLC)"}
PLAN, INK, MUTED, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
SET_COLORS = ["#2a78d6", "#1baf7a", "#8a8985", "#eda100", "#eb6834"]
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.8, "font.size": 10, "legend.frameon": False,
})
SUMMARY = {}


def save(fig, name):
    fig.savefig(os.path.join(RESULTS, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def hand_error(a):
    return 1000 * np.linalg.norm(a["hand"] - a["hand_ref"], axis=1)


# ---------------------------------------------------------------------------
# Offline loop: the controller of controller_node.py on the same MuJoCo model, 100 Hz, delay.
# ---------------------------------------------------------------------------
def run_offline(mode, mass_scale=None, delay=1, hold=1.0):
    cn.CONTROL_MODE = mode
    cn._prev_error = None
    model = ps.build_model(mass_scale, show_plan=False)
    data = mujoco.MjData(model)
    ps.reset_to_start(model, data)
    t, Q, dQ, ddQ = ps.plan()
    every = int(round(cn.TRAJ_DT / model.opt.timestep))
    first = cn.compute_torques(data.qpos[:7].copy(), data.qvel[:7].copy(), Q[0], dQ[0], ddQ[0])
    cn._prev_error = None
    pending = [first] * delay                      # brakes hold the arm until the first command
    rows = []
    for k in range(len(t) + int(hold / cn.TRAJ_DT)):
        q, dq = data.qpos[:7].copy(), data.qvel[:7].copy()
        i = min(k, len(t) - 1)
        if k < len(t):
            qd, dqd, ddqd = Q[i], dQ[i], ddQ[i]
            pending.append(cn.compute_torques(q, dq, qd, dqd, ddqd))
        else:
            # like the template's control_loop after the plan: hold with C q_dot + g at q_dot = 0, i.e. g(q)
            qd = cn.Q_END
            pending.append(cn.compute_dynamics(q, np.zeros(7), np.zeros(7))[1])
        tau = pending.pop(0)
        rows.append(np.concatenate([[k * cn.TRAJ_DT], q, qd, tau]))
        for _ in range(every):
            data.ctrl[:7] = tau
            mujoco.mj_step(model, data)
    a = np.array(rows)
    return {"t": a[:, 0], "q": a[:, 1:8], "q_ref": a[:, 8:15], "tau": a[:, 15:22]}


# ---------------------------------------------------------------------------
def fig_membership():
    x = np.linspace(-1.5, 1.5, 601)
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    for ax, shoulders, title in ((axes[0], True, "inputs E = e / GE and DE = (Δe/dt) / GDE"),
                                 (axes[1], False, "output u (torque or acceleration = GU · u)")):
        mu = cn.membership(np.clip(x, -1, 1) if shoulders else x, shoulders=shoulders)
        if not shoulders:
            mu = cn.membership(x, shoulders=False)
        for k, name in enumerate(cn.SETS):
            ax.plot(x, mu[:, k], color=SET_COLORS[k], label=name)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.12)
        ax.set_ylabel("membership")
    axes[0].set_xlabel("normalised input (clipped to ±1)")
    axes[1].set_xlabel("normalised output universe (±1.5 so the outer sets are whole)")
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.1, 1.38), fontsize=9)
    save(fig, "flc_membership.png")


def fig_surface():
    x = np.linspace(-1, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for method, color, label in (("sum-product", COL["MODEL_FLC"], "sum-product (used)"),
                                 ("min-max", COL["GRAVITY_FLC"], "min-max (first design)")):
        axes[0].plot(x, cn.fuzzy_inference(x, np.zeros_like(x), method), color=color, label=label)
        axes[1].plot(x, cn.fuzzy_inference(x, x, method), color=color, label=label)
    for ax in axes:
        ax.plot(x, np.clip(x if ax is axes[0] else 2 * x, -1, 1), "--", color=PLAN, lw=1.2, label="linear PD, same gains")
        ax.set_xlabel("E")
        ax.set_ylabel("u")
    axes[0].set_title("slice DE = 0: u(E, 0)", fontsize=10)
    axes[1].set_title("slice DE = E: u(E, E)", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")
    inset = axes[1].inset_axes([0.08, 0.55, 0.32, 0.38])
    xs = np.linspace(-0.1, 0.1, 81)
    for method, color in (("sum-product", COL["MODEL_FLC"]), ("min-max", COL["GRAVITY_FLC"])):
        inset.plot(xs, cn.fuzzy_inference(xs, xs, method), color=color, lw=1.4)
    inset.plot(xs, 2 * xs, "--", color=PLAN, lw=1.0)
    inset.set_title("near zero", fontsize=8)
    inset.tick_params(labelsize=7)
    save(fig, "flc_surface.png")
    SUMMARY["slope_near_zero"] = {m: float(cn.fuzzy_inference(np.array([0.02]), np.array([0.02]), m)[0] / 0.04)
                                  for m in ("sum-product", "min-max")}


def load_live():
    runs = {}
    for mode in ("model_flc", "gravity_flc", "flc_only", "model_only", "model_flc_mass20", "gravity_flc_mass20"):
        path = os.path.join(RESULTS, f"sim_ros_{mode}.npz")
        if os.path.exists(path):
            runs[mode] = dict(np.load(path))
    return runs


def fig_tracking(runs):
    if "model_flc" not in runs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.4), sharex=True)
    for col, j in enumerate((1, 3)):
        a = runs["model_flc"]
        axes[0, col].plot(a["t"], np.degrees(a["q_ref"][:, j]), "--", color=PLAN, lw=2.2, label="planned")
        for mode in ("model_flc", "gravity_flc", "model_only"):
            if mode in runs:
                b = runs[mode]
                axes[0, col].plot(b["t"], np.degrees(b["q"][:, j]), color=COL[mode.upper()], lw=1.3, label=NAME[mode.upper()])
                axes[1, col].plot(b["t"], np.degrees(b["q_ref"][:, j] - b["q"][:, j]), color=COL[mode.upper()], lw=1.3)
        axes[0, col].set_title(f"joint {j + 1}: planned vs actual (live, ROS 2)", fontsize=10)
        axes[1, col].set_title(f"joint {j + 1} error", fontsize=10)
        axes[0, col].set_ylabel("angle [deg]")
        axes[1, col].set_ylabel("error [deg]")
        axes[1, col].set_xlabel("t [s]")
        axes[1, col].set_ylim(-1.5, 1.5)
    lo = min(np.degrees(runs["model_flc"]["q_ref"][:, 1]).min(), -70)
    axes[0, 0].set_ylim(lo - 10, 70)
    axes[0, 1].set_ylim(-200, 0)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.text(0.5, -0.01, "Model only leaves the plotted range within a second (it ends 147° off); error axes clipped at ±1.5°.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "tracking_desired_vs_actual.png")


def fig_error_modes(runs):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for mode in ("model_only", "flc_only", "gravity_flc", "model_flc"):
        if mode in runs:
            a = runs[mode]
            ax.plot(a["t"], np.maximum(hand_error(a), 1e-3), color=COL[mode.upper()], lw=1.4, label=NAME[mode.upper()])
    ax.axvline(4.0, color=PLAN, lw=1, ls=":")
    ax.text(4.05, 2e-3, "plan ends", color=MUTED, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("hand error [mm]")
    ax.set_title("Hand position error, live through ROS 2 (log scale)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.07))
    save(fig, "tracking_error_modes.png")
    SUMMARY["live"] = {}
    for mode, a in runs.items():
        h = hand_error(a)
        e = np.degrees(np.abs(a["q_ref"] - a["q"]))
        mv = a["t"] <= float(a["T"])
        SUMMARY["live"][mode] = {"joint_max_deg": float(e[mv].max()), "hand_max_mm": float(h[mv].max()),
                                 "hand_rms_mm": float(np.sqrt(np.mean(h[mv] ** 2))), "hand_final_mm": float(h[-10:].mean()),
                                 "joint_final_deg": float(e[-10:].max())}


def fig_mass(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, mode in zip(axes, ("gravity_flc", "model_flc")):
        for tag, ls, label in ((mode, "-", "model = real arm"), (mode + "_mass20", "--", "real links 20 % heavier")):
            if tag in runs:
                a = runs[tag]
                ax.plot(a["t"], np.maximum(hand_error(a), 1e-3), ls, color=COL[mode.upper()], lw=1.5, label=label)
        ax.set_yscale("log")
        ax.set_title(NAME[mode.upper()], fontsize=10)
        ax.set_xlabel("t [s]")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("hand error [mm]")
    save(fig, "mass_plus20.png")


def fig_error_scaling(runs):
    """How big are the errors each FLC works with? (Team 5: the sets were narrowed for g + FLC.)"""
    if "gravity_flc" not in runs or "flc_only" not in runs:
        return
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    j = 1
    bins = np.logspace(-4, 0.5, 60)
    for mode in ("flc_only", "gravity_flc", "model_flc"):
        a = runs[mode]
        mv = a["t"] <= float(a["T"])
        ax.hist(np.degrees(np.abs(a["q_ref"][mv, j] - a["q"][mv, j])), bins=bins, histtype="step", lw=1.8,
                color=COL[mode.upper()], label=NAME[mode.upper()])
    for mode, height in (("FLC_ONLY", 0.55), ("GRAVITY_FLC", 0.85)):
        ge = np.degrees(cn.FLC_GAINS[mode]["GE"][j])
        ax.axvline(ge, color=COL[mode], ls=":", lw=1.3)
        ax.text(ge * 0.93, ax.get_ylim()[1] * height, f"GE, {NAME[mode].split(' (')[0]}\n= {ge:.2f}°", fontsize=7.5,
                color=INK, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("|error| of joint 2 during the motion [deg]")
    ax.set_ylabel("control steps")
    ax.set_title("The errors each FLC has to work with (live runs) and where its 'large' set starts", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "error_scaling.png")


def fig_chatter():
    """The first design (scaled by M_ii, min-max) against the final one, offline with 2 steps of delay."""
    saved_gains, saved_inf = dict(cn.FLC_GAINS), cn.INFERENCE
    m_ii = np.array([3.2, 3.6, 0.5, 1.05, 0.15, 0.16, 0.11])
    gu = np.array([15, 15, 8, 10, 3, 3, 3.0])
    cn.FLC_GAINS = dict(saved_gains)
    cn.FLC_GAINS["FIRST"] = dict(GE=gu / (m_ii * 400), GDE=gu / (2 * m_ii * 20), GU=gu)
    cn.INFERENCE = "min-max"

    def first_design(q, dq, q_des, dq_des, ddq_des):
        dt = cn.TRAJ_DT
        # tau = M q_ddot_des + h + FLC torque, FLC scaled with M_ii, min-max inference
        global_prev = cn._prev_error
        error = np.asarray(q_des) - q
        de = np.zeros(7) if global_prev is None else error - global_prev
        cn._prev_error = error
        g = cn.FLC_GAINS["FIRST"]
        u = g["GU"] * cn.fuzzy_inference(np.clip(error / g["GE"], -1, 1), np.clip(de / dt / g["GDE"], -1, 1), "min-max")
        M, h = cn.compute_dynamics(q, dq)
        return np.clip(M @ ddq_des + h + u, -cn.TORQUE_LIMIT, cn.TORQUE_LIMIT)

    original = cn.compute_torques
    cn.compute_torques = first_design
    first = run_offline("MODEL_FLC")
    cn.compute_torques = original
    cn.FLC_GAINS, cn.INFERENCE = saved_gains, saved_inf
    final = run_offline("MODEL_FLC")
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for ax, j in zip(axes, (0, 3)):
        ax.plot(first["t"], first["tau"][:, j], color=COL["GRAVITY_FLC"], lw=0.9, label="first design: FLC torque scaled by M_ii, min-max")
        ax.plot(final["t"], final["tau"][:, j], color=COL["MODEL_FLC"], lw=1.6, label="final: FLC acceleration × M(q), sum-product")
        ax.set_ylabel(f"torque joint {j + 1} [N m]")
    axes[1].set_xlabel("t [s]")
    axes[0].set_title("Joint torques, model + FLC, offline with the ROS 2 delay", fontsize=10, pad=30)
    axes[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    save(fig, "chatter.png")
    SUMMARY["chatter_mean_abs_dtau_Nm"] = {
        "first_design": np.abs(np.diff(first["tau"], axis=0)).mean(axis=0).round(3).tolist(),
        "final": np.abs(np.diff(final["tau"], axis=0)).mean(axis=0).round(3).tolist()}


def rule_walkthrough():
    """One control step of joint 2 in the Team 5 (g + FLC) controller, rule by rule."""
    mode = "GRAVITY_FLC"
    g = cn.FLC_GAINS[mode]
    j = 1
    e, e_rate = 0.6 * g["GE"][j], -0.25 * g["GDE"][j]           # error big-ish, already shrinking a bit
    E, DE = e / g["GE"][j], e_rate / g["GDE"][j]
    muE, muD = cn.membership(np.array([E]))[0], cn.membership(np.array([DE]))[0]
    lines = [f"Team 5 controller (g(q) + FLC), joint 2: GE = {g['GE'][j]:.4f} rad, GDE = {g['GDE'][j]:.4f} rad/s, GU = {g['GU'][j]:.0f} N m",
             f"error e = {e:.5f} rad ({np.degrees(e):.3f} deg), error rate = {e_rate:.4f} rad/s",
             f"1. normalise:  E = e/GE = {E:+.2f},  DE = rate/GDE = {DE:+.2f}",
             "2. fuzzify:    E  -> " + ", ".join(f"{s} {m:.2f}" for s, m in zip(cn.SETS, muE) if m > 0),
             "               DE -> " + ", ".join(f"{s} {m:.2f}" for s, m in zip(cn.SETS, muD) if m > 0),
             "3. rules that fire (strength = mu_E x mu_DE, sum-product):"]
    strength = np.zeros(5)
    for i, si in enumerate(cn.SETS):
        for k, sk in enumerate(cn.SETS):
            w = muE[i] * muD[k]
            if w > 0:
                out = cn.RULES[i][k]
                strength[cn.SETS.index(out)] += w
                lines.append(f"     IF error is {si:2s} AND rate is {sk:2s} THEN output is {out:2s}   strength {w:.2f}")
    lines.append("4. output sets weighted: " + ", ".join(f"{s} {w:.2f}" for s, w in zip(cn.SETS, strength) if w > 0))
    u = cn.fuzzy_inference(np.array([E]), np.array([DE]))[0]
    lines.append(f"5. centroid of the summed, scaled triangles: u = {u:+.3f}  (= sum(w_k c_k)/sum(w_k) = "
                 f"{(strength @ cn.CENTERS) / strength.sum():+.3f})")
    lines.append(f"6. torque correction = GU x u = {g['GU'][j] * u:+.2f} N m, added to g2(q)")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, "rule_walkthrough.txt"), "w") as f:
        f.write(text + "\n")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    fig_membership()
    fig_surface()
    runs = load_live()
    fig_tracking(runs)
    fig_error_modes(runs)
    fig_mass(runs)
    fig_error_scaling(runs)
    fig_chatter()
    rule_walkthrough()
    with open(os.path.join(RESULTS, "flc_summary.json"), "w") as f:
        json.dump(SUMMARY, f, indent=2)
    print("\nLive runs through ROS 2 (during the 4 s motion / 2 s after it):")
    for mode, s in SUMMARY.get("live", {}).items():
        print(f"  {mode:20s} joint max {s['joint_max_deg']:7.3f} deg   hand max {s['hand_max_mm']:7.2f} mm   "
              f"after: {s['hand_final_mm']:7.3f} mm")
    print(f"FLC slope near zero (E = DE = 0.02, relative to linear): {SUMMARY['slope_near_zero']}")
    print(f"mean |torque change| per step, first design vs final: {SUMMARY['chatter_mean_abs_dtau_Nm']}")
    print("Figures and summaries in Challenge 03/results/")


if __name__ == "__main__":
    main()
