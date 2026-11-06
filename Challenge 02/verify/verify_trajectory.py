#!/usr/bin/env python3
"""
Checks everything in Challenge 2 without ROS 2:

  1. FK copied from Challenge 1 still matches MuJoCo
  2. compute_jacobian      vs finite differences and vs MuJoCo's own Jacobian
  3. compute_jacobian_dot  vs finite differences
  4. acceleration kinematics along the plan: p_ddot = J ddq + J_dot dq
  5. plan_trajectory: output format and boundary conditions
  6. joint position / velocity / acceleration limits (and no overshoot)
  7. self-collision along the whole motion, with the MuJoCo collision meshes
  8. singularities along the path
  9. both control modes simulated (POSITION and VELOCITY), commands at 20 Hz
     exactly like the node, with and without gravity compensation and timer jitter
 10. plots in results/

    .venv/bin/python "Challenge 02/verify/verify_trajectory.py"

Exits non-zero if any check fails.
"""
import itertools
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
from ros_free_import import load_trajectory_node  # noqa: E402

tn = load_trajectory_node()
import panda_sim as sim  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
rng = np.random.default_rng(42)
failures = []

SELF_MARGIN = 0.015    # every pair of non-neighbouring links at least 1.5 cm apart
FLOOR_MARGIN = 0.05    # every moving link at least 5 cm above the floor
COLLISION_STEP = 2     # check every 2nd 1 ms sample, i.e. every 2 ms
ARM_SIZE = 1.25        # m, upper bound on the distance between any two points of the arm
                       # (0.333 m base to shoulder + 0.858 m max reach + fingers)
# Two wrist pairs sit about 2-3 cm apart in EVERY pose (link5's housing wraps
# around link7, link6 around the hand). They are checked against SELF_MARGIN
# like the rest, but left out of the "fold clearance", which measures how
# close the fold brings the arm to itself.
WRIST_PAIRS = [{"link5", "link7"}, {"link6", "hand"}]
# Same fold but the wrist tilted back: the fingers go into link1. Used to
# show that the collision check really catches a bad configuration.
Q_BAD_FOLD = np.array([0.0, -1.0, 0.0, -3.05, 0.0, 0.8, 0.785])


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def random_q():
    return rng.uniform(tn.Q_MIN, tn.Q_MAX)


def hand_position(q):
    return tn.forward_kinematics(q)[:3, 3]


model = sim.build_model("position", show_plan=False)
data = mujoco.MjData(model)
ARM_QPOS = [model.joint(j).qposadr[0] for j in sim.ARM_JOINTS]
ARM_DOF = [model.joint(j).dofadr[0] for j in sim.ARM_JOINTS]
FINGER_QPOS = [model.joint(j).qposadr[0] for j in sim.FINGER_JOINTS]
HAND = model.body("hand").id
BODY_NAMES = [model.body(i).name for i in range(model.nbody)]


def set_arm(q, qd=None):
    data.qpos[ARM_QPOS] = q
    data.qpos[FINGER_QPOS] = 0.04
    data.qvel[:] = 0.0
    if qd is not None:
        data.qvel[ARM_DOF] = qd
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
print("\n1. FK copied from Challenge 1 vs MuJoCo (500 random configurations)")
worst_pos = 0.0
for _ in range(500):
    q = random_q()
    set_arm(q)
    worst_pos = max(worst_pos, np.linalg.norm(hand_position(q) - data.xpos[HAND]))
check("hand position matches MuJoCo", worst_pos < 1e-9, f"worst {worst_pos * 1000:.1e} mm")

# ---------------------------------------------------------------------------
print("\n2. compute_jacobian (3x7) vs finite differences and vs MuJoCo mj_jac")
J = tn.compute_jacobian(tn.Q_START)
check("shape is 3x7", J.shape == (3, 7), str(J.shape))
worst_fd = worst_mj = 0.0
eps = 1e-6
jacp = np.zeros((3, model.nv))
for _ in range(200):
    q = random_q()
    J = tn.compute_jacobian(q)
    J_fd = np.column_stack([(hand_position(q + eps * e) - hand_position(q - eps * e)) / (2 * eps)
                            for e in np.eye(7)])
    worst_fd = max(worst_fd, np.abs(J - J_fd).max())
    set_arm(q)
    mujoco.mj_jac(model, data, jacp, None, data.xpos[HAND], HAND)
    worst_mj = max(worst_mj, np.abs(J - jacp[:, ARM_DOF]).max())
check("matches finite differences of FK", worst_fd < 1e-7, f"max diff {worst_fd:.1e}")
check("matches MuJoCo's own Jacobian", worst_mj < 1e-9, f"max diff {worst_mj:.1e}")

# ---------------------------------------------------------------------------
print("\n3. compute_jacobian_dot (analytical) vs finite differences of compute_jacobian")
worst = 0.0
h = 1e-6
for _ in range(200):
    q = random_q()
    qd = rng.uniform(-tn.QD_MAX, tn.QD_MAX)
    Jd = tn.compute_jacobian_dot(q, qd)
    Jd_fd = (tn.compute_jacobian(q + h * qd) - tn.compute_jacobian(q - h * qd)) / (2 * h)
    worst = max(worst, np.abs(Jd - Jd_fd).max())
check("shape is 3x7", Jd.shape == (3, 7))
check("matches dJ/dt = (J(q + h qd) - J(q - h qd)) / 2h", worst < 1e-6, f"max diff {worst:.1e}")
check("zero when the arm is still", np.allclose(tn.compute_jacobian_dot(tn.Q_START, np.zeros(7)), 0))

# ---------------------------------------------------------------------------
print("\n4. Velocity and acceleration kinematics along the planned motion (1 ms samples)")
t_f, q_f, dq_f, ddq_f = tn.plan_trajectory(tn.Q_START, tn.Q_END, tn.TRAJ_DURATION, 0.001)
p_f = np.array([hand_position(q) for q in q_f])
p_dot_num = np.gradient(p_f, t_f, axis=0)
p_ddot_num = np.gradient(p_dot_num, t_f, axis=0)
p_dot_J = np.array([tn.compute_jacobian(q) @ dq for q, dq in zip(q_f, dq_f)])
p_ddot_J = np.array([tn.compute_jacobian(q) @ ddq + tn.compute_jacobian_dot(q, dq) @ dq
                     for q, dq, ddq in zip(q_f, dq_f, ddq_f)])
inner = slice(5, -5)    # np.gradient is one-sided at the ends
v_err = np.abs(p_dot_J - p_dot_num)[inner].max()
a_err = np.abs(p_ddot_J - p_ddot_num)[inner].max()
a_err_noJdot = np.abs(np.array([tn.compute_jacobian(q) @ ddq for q, ddq in zip(q_f, ddq_f)])
                      - p_ddot_num)[inner].max()
check("p_dot = J dq", v_err < 1e-5, f"max diff {v_err:.1e} m/s (peak hand speed "
      f"{np.linalg.norm(p_dot_J, axis=1).max():.3f} m/s)")
check("p_ddot = J ddq + J_dot dq", a_err < 1e-3, f"max diff {a_err:.1e} m/s^2")
print(f"         without the J_dot dq term the error would be {a_err_noJdot:.3f} m/s^2 "
      f"(peak hand acceleration {np.linalg.norm(p_ddot_J, axis=1).max():.3f} m/s^2)")

# ---------------------------------------------------------------------------
print("\n5. plan_trajectory: output format and boundary conditions")
t, q, dq, ddq = tn.plan_trajectory(tn.Q_START, tn.Q_END, tn.TRAJ_DURATION, tn.TRAJ_DT)
N = len(t)
check("four arrays of equal length", all(len(a) == N for a in (q, dq, ddq)), f"N = {N}")
check("shapes (N, 7)", q.shape == dq.shape == ddq.shape == (N, 7))
check("t from 0 to T in steps of dt", t[0] == 0 and np.isclose(t[-1], tn.TRAJ_DURATION)
      and np.allclose(np.diff(t), tn.TRAJ_DT), f"t = 0 ... {t[-1]:.2f} s, dt = {tn.TRAJ_DT}")
check("q(0) = Q_START and q(T) = Q_END", np.allclose(q[0], tn.Q_START) and np.allclose(q[-1], tn.Q_END))
check("zero velocity at both ends", np.allclose(dq[[0, -1]], 0))
check("zero acceleration at both ends", np.allclose(ddq[[0, -1]], 0))
check("dq is the derivative of q", np.abs(np.gradient(q_f, t_f, axis=0) - dq_f)[1:-1].max() < 1e-5)
check("ddq is the derivative of dq", np.abs(np.gradient(dq_f, t_f, axis=0) - ddq_f)[1:-1].max() < 1e-4)

q_far = np.array([2.5, 1.5, 2.5, -0.2, 2.5, 3.5, 2.5])
q_near = np.array([-2.5, -1.5, -2.5, -3.0, -2.5, 0.0, -2.5])
t_s, _, dq_s, ddq_s = tn.plan_trajectory(q_near, q_far, 1.0, tn.TRAJ_DT)
check("too-short T is stretched to respect the limits",
      t_s[-1] > 1.0 and np.all(np.abs(dq_s) <= tn.QD_MAX + 1e-9) and np.all(np.abs(ddq_s) <= tn.QDD_MAX + 1e-9),
      f"asked 1.0 s for a 5 rad move, got {t_s[-1]:.2f} s")

# ---------------------------------------------------------------------------
print("\n6. Joint limits during the motion (1 ms samples)")
margin_lo = (q_f - tn.Q_MIN).min(axis=0)
margin_hi = (tn.Q_MAX - q_f).min(axis=0)
margin = np.minimum(margin_lo, margin_hi)
j = int(np.argmin(margin))
check("every joint inside its position limits at every sample", margin.min() > 0,
      f"tightest: joint {j + 1}, {margin[j]:.3f} rad ({np.degrees(margin[j]):.1f} deg) from its limit")
steps = np.diff(q_f, axis=0) * np.sign(tn.Q_END - tn.Q_START)
check("no overshoot: each joint only moves towards its end value", np.all(steps >= -1e-12))
vel_use = np.abs(dq_f).max(axis=0) / tn.QD_MAX
acc_use = np.abs(ddq_f).max(axis=0) / tn.QDD_MAX
check(f"peak velocity <= {tn.LIMIT_FRACTION:.0%} of the limit", vel_use.max() <= tn.LIMIT_FRACTION,
      f"worst joint {np.argmax(vel_use) + 1}: {np.abs(dq_f).max():.3f} rad/s = {vel_use.max():.0%} of limit")
check(f"peak acceleration <= {tn.LIMIT_FRACTION:.0%} of the limit", acc_use.max() <= tn.LIMIT_FRACTION,
      f"worst joint {np.argmax(acc_use) + 1}: {acc_use.max():.0%} of limit")
print(f"         shortest safe duration for this move: {tn.minimum_duration(tn.Q_START, tn.Q_END):.2f} s "
      f"(we use {tn.TRAJ_DURATION:.1f} s)")

# ---------------------------------------------------------------------------
print("\n7. Self-collision along the motion (MuJoCo collision meshes, every 2 ms)")
robot_geoms = [g for g in range(model.ngeom)
               if (model.geom_contype[g] or model.geom_conaffinity[g]) and model.geom_bodyid[g] != 0]
floor = model.geom("floor").id
self_pairs, fold_pairs = [], []
for a, b in itertools.combinations(robot_geoms, 2):
    ba, bb = model.geom_bodyid[a], model.geom_bodyid[b]
    if ba == bb or model.body_parentid[ba] == bb or model.body_parentid[bb] == ba:
        continue    # same link, or two links joined by a joint: always touching
    self_pairs.append((a, b))
    if {BODY_NAMES[ba], BODY_NAMES[bb]} not in WRIST_PAIRS:
        fold_pairs.append((a, b))
floor_geoms = [g for g in robot_geoms if BODY_NAMES[model.geom_bodyid[g]] not in ("link0", "link1")]
fromto = np.zeros(6)


def closest(pairs):
    best, who = np.inf, None
    for a, b in pairs:
        dist = mujoco.mj_geomDistance(model, data, a, b, 0.3, fromto)
        if dist < best:
            best, who = dist, (BODY_NAMES[model.geom_bodyid[a]], BODY_NAMES[model.geom_bodyid[b]])
    return best, who


def robot_contacts():
    """Contacts MuJoCo's own collision detection finds between two robot links."""
    return sum(1 for c in data.contact[:data.ncon] if c.geom1 != floor and c.geom2 != floor)


t_c = t_f[::COLLISION_STEP]
clear_self, clear_fold, clear_floor, contacts = [], [], [], 0
worst_self = worst_fold = (np.inf, None, 0.0)
for tau, q in zip(t_c, q_f[::COLLISION_STEP]):
    set_arm(q)
    d_self, who_self = closest(self_pairs)
    d_fold, who_fold = closest(fold_pairs)
    d_floor, _ = closest([(floor, g) for g in floor_geoms])
    contacts += robot_contacts()
    clear_self.append(d_self)
    clear_fold.append(d_fold)
    clear_floor.append(d_floor)
    if d_self < worst_self[0]:
        worst_self = (d_self, who_self, tau)
    if d_fold < worst_fold[0]:
        worst_fold = (d_fold, who_fold, tau)
clear_self, clear_fold, clear_floor = map(np.array, (clear_self, clear_fold, clear_floor))
check("MuJoCo finds no link-link contact at any sample", contacts == 0, f"{len(t_c)} samples")
check(f"every non-neighbouring pair >= {SELF_MARGIN * 100:.1f} cm apart", worst_self[0] >= SELF_MARGIN,
      f"closest {worst_self[0] * 100:.1f} cm, {worst_self[1][0]}-{worst_self[1][1]} (a wrist pair, same in every pose)")
check(f"fold clearance (wrist pairs left out) >= {SELF_MARGIN * 100:.1f} cm", worst_fold[0] > SELF_MARGIN,
      f"closest {worst_fold[0] * 100:.1f} cm, {worst_fold[1][0]}-{worst_fold[1][1]} at t = {worst_fold[2]:.2f} s")
check(f"every moving link >= {FLOOR_MARGIN * 100:.0f} cm above the floor", clear_floor.min() >= FLOOR_MARGIN,
      f"lowest {clear_floor.min() * 100:.1f} cm")
# Between two samples, can two links get closer by more than the margin?
# A point x on the arm moves at  sum_i qd_i z_i x (x - o_i),  so its speed is at
# most sum_i |qd_i| * ARM_SIZE. Two points close in on each other at most twice
# that. If that times the sample step is below the margin, and every sample has
# at least the margin, no contact can happen between samples either.
dt_c = t_c[1] - t_c[0]
closing = 2 * ARM_SIZE * np.abs(dq_f).sum(axis=1).max() * dt_c
check("samples are dense enough to cover the time between them", closing < SELF_MARGIN,
      f"links close in by at most {closing * 100:.2f} cm per {dt_c * 1000:.0f} ms step, "
      f"< {SELF_MARGIN * 100:.1f} cm margin")
set_arm(Q_BAD_FOLD)
d_bad, who_bad = closest(fold_pairs)
check("the check catches a bad fold (same fold, wrist tilted back)", d_bad < 0 and robot_contacts() > 0,
      f"q6 = 0.8: {who_bad[0]}-{who_bad[1]} overlap by {-d_bad * 100:.1f} cm")

# ---------------------------------------------------------------------------
print("\n8. Singularities along the path")
sv = np.array([np.linalg.svd(tn.compute_jacobian(q), compute_uv=False) for q in q_f[::COLLISION_STEP]])
sigma_min = sv[:, -1]
manip = sv.prod(axis=1)                      # sqrt(det(J J^T)) = product of singular values
cond = sv[:, 0] / sv[:, -1]
i = int(np.argmin(sigma_min))
check("Jv keeps full rank (3) along the whole path", sigma_min.min() > 1e-3,
      f"smallest singular value {sigma_min.min():.3f} m/rad at t = {t_c[i]:.2f} s")
print(f"         manipulability: {manip[0]:.4f} at start, {manip.min():.4f} lowest, {manip[-1]:.4f} at end; "
      f"worst condition number {cond.max():.1f}")
print("         The planner never inverts J, so a small singular value cannot make dq blow up.")

# ---------------------------------------------------------------------------
print("\n9. Both control modes, simulated like the node (one command every 50 ms)")


def simulate(mode, gravcomp=True, jitter_std=0.0, seed=0):
    """Replays exactly what tick() publishes, holding each command for one timer period."""
    m = sim.build_model(mode, gravcomp=gravcomp, show_plan=False)
    d = mujoco.MjData(m)
    sim.reset_to_start(m, d, mode)
    hand = m.body("hand").id
    qadr = [m.joint(name).qposadr[0] for name in sim.ARM_JOINTS]
    log = sim.TrackingLog()
    _, q_plan, dq_plan, _ = sim.plan()
    commands = list(q_plan) if mode == "position" else list(dq_plan)
    hold = tn.Q_END if mode == "position" else np.zeros(7)
    r = np.random.default_rng(seed)
    periods = tn.TRAJ_DT + jitter_std * r.standard_normal(len(commands))
    switch_times = np.concatenate([[0.0], np.cumsum(periods)])
    k, last_log = 0, -1.0
    while d.time < log.T + sim.SETTLE_TIME:
        while k < len(switch_times) and d.time >= switch_times[k] - 1e-9:
            cmd = commands[k] if k < len(commands) else hold
            sim.apply_command(m, d, list(cmd) + [tn.GRIPPER_CTRL, 0.0])
            k += 1
        mujoco.mj_step(m, d)
        if d.time - last_log >= sim.LOG_PERIOD - 1e-9:
            last_log = d.time
            log.record(d.time, d.qpos[qadr].copy(), d.xpos[hand].copy())
    return log


runs = {
    "POSITION": simulate("position"),
    "VELOCITY": simulate("velocity"),
    "VELOCITY + 5 ms timer jitter": simulate("velocity", jitter_std=0.005),
    "POSITION, no gravity comp.": simulate("position", gravcomp=False),
    "VELOCITY, no gravity comp.": simulate("velocity", gravcomp=False),
}
summaries = {name: log.summary() for name, log in runs.items()}
print(f"         {'run':32s} {'worst joint err':>16s} {'worst hand err':>15s} {'hand err at T':>14s} {'hand err T+2s':>14s}")
for name, s in summaries.items():
    print(f"         {name:32s} {s['max_joint_err_deg']:12.2f} deg {s['max_hand_err_mm']:11.1f} mm "
          f"{s['at_T_hand_err_mm']:10.1f} mm {s['final_hand_err_mm']:10.2f} mm")
check("POSITION mode ends on Q_END (< 1 mm)", summaries["POSITION"]["final_hand_err_mm"] < 1.0)
check("VELOCITY mode tracks the plan (< 1 cm during the motion)",
      summaries["VELOCITY"]["max_hand_err_mm"] < 10.0)
print("         VELOCITY mode has no position feedback: whatever error builds up stays after the end.")

# ---------------------------------------------------------------------------
if "--no-plots" not in sys.argv:
    print("\n10. Plots")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference palette (dataviz skill), validated: fixed order, one per moving joint.
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0, "figure.facecolor": "white",
    })
    os.makedirs(RESULTS, exist_ok=True)
    moving = [i for i in range(7) if not np.isclose(tn.Q_START[i], tn.Q_END[i])]

    def label_line_ends(ax, x, ys, labels):
        for y, lab, c in zip(ys, labels, SERIES):
            ax.annotate(lab, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)

    # Figure 1: q, dq, ddq of the moving joints (others stay still).
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8), sharex=True)
    for ax, arr, unit, lim, title in (
            (axes[0], q_f, "rad", None, "Joint position q(t)"),
            (axes[1], dq_f, "rad/s", tn.QD_MAX, "Joint velocity dq(t)"),
            (axes[2], ddq_f, "rad/s²", tn.QDD_MAX, "Joint acceleration ddq(t)")):
        for c, i in zip(SERIES, moving):
            ax.plot(t_f, arr[:, i], color=c, label=f"q{i + 1}")
        if lim is None:     # all curves end at 0 in the dq and ddq panels; the legend names them there
            label_line_ends(ax, t_f, [arr[:, i] for i in moving], [f"q{i + 1}" for i in moving])
        ax.set_ylabel(unit)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        if lim is None:
            ax.set_title(title, loc="left")
        else:
            use = np.abs(arr).max(axis=0) / lim
            ax.set_title(f"{title}   (peak {use.max():.0%} of its limit, joint {np.argmax(use) + 1})", loc="left")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.32))
    axes[2].set_xlabel("time (s)")
    axes[2].set_xlim(0, t_f[-1] * 1.06)
    fig.text(0.01, 0.005, "Quintic in joint space, T = 4 s. Joints 3, 5 and 7 do not move. "
             "Velocity and acceleration are zero at t = 0 and t = T.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(RESULTS, "trajectory_profiles.png"), dpi=150)
    plt.close(fig)

    # Figure 2: hand path, clearance, singular value: three panels, one unit each.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    ax = axes[0]
    radial = np.hypot(p_f[:, 0], p_f[:, 1])
    ax.plot(radial, p_f[:, 2], color=SERIES[0])
    ax.plot(radial[0], p_f[0, 2], "o", color=SERIES[0], markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.plot(radial[-1], p_f[-1, 2], "s", color=SERIES[0], markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.annotate("start (stretched)", (radial[0], p_f[0, 2]), xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=8, color=INK)
    ax.annotate("end (folded)", (radial[-1], p_f[-1, 2]), xytext=(8, -4), textcoords="offset points",
                fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(0, 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("distance from the base axis (m)")
    ax.set_ylabel("height (m)")
    ax.set_title("Hand path, side view", loc="left")

    ax = axes[1]
    ax.plot(t_c, clear_fold * 100, color=SERIES[0], label="fold clearance")
    ax.plot(t_c, clear_floor * 100, color=SERIES[1], label="floor clearance")
    ax.axhline(SELF_MARGIN * 100, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"{SELF_MARGIN * 100:.1f} cm margin", (0.05, SELF_MARGIN * 100), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color=MUTED)
    label_line_ends(ax, t_c, [clear_fold * 100, clear_floor * 100], ["fold", "floor"])
    ax.set_xlim(0, t_c[-1] * 1.12)
    ax.set_ylim(0, clear_floor.max() * 100 * 1.25)
    ax.set_ylim(0, None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cm")
    ax.set_title("Closest distance between links", loc="left")
    ax.legend(loc="upper left", ncol=2)

    ax = axes[2]
    ax.plot(t_c, sigma_min, color=SERIES[0])
    ax.set_ylim(0, None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/rad")
    ax.set_title("Smallest singular value of Jv", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "path_clearance_singularity.png"), dpi=150)
    plt.close(fig)

    # Figure 3: tracking error of the hand in both control modes.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, names, title in (
            (axes[0], ["POSITION", "VELOCITY", "VELOCITY + 5 ms timer jitter"], "With gravity compensation"),
            (axes[1], ["POSITION, no gravity comp.", "VELOCITY, no gravity comp."], "Without gravity compensation")):
        for c, name in zip(SERIES, names):
            a = runs[name].arrays()
            err = 1000 * np.linalg.norm(a["hand"] - a["hand_ref"], axis=1)
            ax.plot(a["t"], err, color=c, label=name.split(",")[0])
        ax.axvline(runs[names[0]].T, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate("plan ends", (runs[names[0]].T, ax.get_ylim()[1]), xytext=(3, -12),
                    textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_xlabel("time since first command (s)")
        ax.set_ylabel("hand error vs plan (mm)")
        ax.set_ylim(0, None)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "tracking_position_vs_velocity.png"), dpi=150)
    plt.close(fig)
    print(f"  saved trajectory_profiles.png, path_clearance_singularity.png, "
          f"tracking_position_vs_velocity.png in {os.path.relpath(RESULTS, os.getcwd())}")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
print("All checks passed.")
