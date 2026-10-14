#!/usr/bin/env python3
"""
Checks our FK and IK against MuJoCo's own kinematics on panda.xml.

MuJoCo is the "true FK": it places every body from the pos/quat in the XML,
which is exactly what the simulator does. No ROS 2 needed.

    .venv/bin/python "Challenge 01/verify/verify_kinematics.py"

Exits non-zero if any check fails.
"""
import os
import re
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'panda_kinematics'))
from panda_kinematics.kinematics import (  # noqa: E402
    Q_HOME, Q_MAX, Q_MIN, TARGET_JOINTS, TARGET_POSE, forward_kinematics,
    inverse_kinematics, jacobian, pose_error, _rotation_error)

XML = os.path.join(HERE, '..', 'model', 'panda.xml')
rng = np.random.default_rng(42)
failures = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def load_model():
    # Meshes only matter for drawing and collisions; dropping them lets the
    # model load without the 35 MB assets folder. Kinematics are unchanged.
    xml = open(XML).read()
    xml = re.sub(r'<mesh [^>]*/>', '', xml)
    xml = re.sub(r'<geom [^>]*mesh=[^>]*/>', '', xml)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


model, data = load_model()
HAND = model.body('hand').id


def mujoco_hand_pose(q):
    data.qpos[:7] = q
    mujoco.mj_kinematics(model, data)
    T = np.eye(4)
    T[:3, :3] = data.xmat[HAND].reshape(3, 3)
    T[:3, 3] = data.xpos[HAND]
    return T


def errs(T_target, T):
    e = pose_error(T_target, T)
    return np.linalg.norm(e[:3]) * 1000.0, np.degrees(np.linalg.norm(e[3:]))


def random_q(n):
    return rng.uniform(Q_MIN, Q_MAX, size=(n, 7))


print("\n1. Model data matches panda.xml")
jnt_ranges = np.array([model.jnt_range[model.joint(f'joint{i}').id] for i in range(1, 8)])
check("Q_MIN / Q_MAX equal the <joint> ranges",
      np.allclose(jnt_ranges[:, 0], Q_MIN) and np.allclose(jnt_ranges[:, 1], Q_MAX))

print("\n2. Forward kinematics vs MuJoCo")
configs = np.vstack([Q_HOME, TARGET_JOINTS, random_q(5000)])
worst_p = worst_r = 0.0
for q in configs:
    p, r = errs(mujoco_hand_pose(q), forward_kinematics(q))
    worst_p, worst_r = max(worst_p, p), max(worst_r, r)
check("5002 configurations, worst position error < 1e-6 mm", worst_p < 1e-6, f"({worst_p:.2e} mm)")
# The hand quat in panda.xml is written to 7 digits (0.9238795 for cos 22.5 deg),
# which alone is ~2e-6 deg away from an exact -45 deg, hence the looser bound.
check("5002 configurations, worst orientation error < 1e-4 deg", worst_r < 1e-4, f"({worst_r:.2e} deg)")
T_home = forward_kinematics(Q_HOME)
print(f"  home keyframe hand pose:\n{np.round(T_home, 4)}")

print("\n3. Jacobian vs finite differences of FK")
worst = 0.0
for q in random_q(200):
    J, T = jacobian(q)
    J_num = np.zeros((6, 7))
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = 1e-6
        J_num[:, i] = pose_error(forward_kinematics(q + dq), T) / 1e-6
    worst = max(worst, np.abs(J - J_num).max())
check("200 configurations, max |J - J_numeric| < 1e-5", worst < 1e-5, f"({worst:.2e})")

print("\n4. Orientation error at 0 and 180 deg")
R = forward_kinematics(Q_HOME)[:3, :3]
check("identical rotations give zero error", np.linalg.norm(_rotation_error(R, R)) < 1e-12)
for axis in np.eye(3):
    Rflip = R @ (2 * np.outer(axis, axis) - np.eye(3))          # 180 deg about a body axis
    check(f"180 deg about body axis {axis.astype(int)} gives pi",
          abs(np.linalg.norm(_rotation_error(Rflip, R)) - np.pi) < 1e-9)


def ik_trials(name, q_true_list, seed_fn):
    n_ok, worst_p, worst_r, times, in_limits = 0, 0.0, 0.0, [], True
    for q_true in q_true_list:
        target = forward_kinematics(q_true)
        t0 = time.perf_counter()
        q_sol, ok = inverse_kinematics(target, seed_fn(q_true))
        times.append(time.perf_counter() - t0)
        if not np.all(np.isfinite(q_sol)):
            in_limits = False
            continue
        in_limits &= bool(np.all(q_sol >= Q_MIN - 1e-12) and np.all(q_sol <= Q_MAX + 1e-12))
        if ok:
            n_ok += 1
            # Independent check: put the solution into MuJoCo, not our FK.
            p, r = errs(target, mujoco_hand_pose(q_sol))
            worst_p, worst_r = max(worst_p, p), max(worst_r, r)
    n = len(q_true_list)
    times = np.array(times) * 1000
    print(f"  {name}: {n_ok}/{n} converged, worst error {worst_p:.4f} mm / {worst_r:.4f} deg "
          f"(checked in MuJoCo), time mean {times.mean():.1f} ms, max {times.max():.1f} ms")
    return n_ok, n, worst_p, worst_r, in_limits


print("\n5. Inverse kinematics round trip (target = FK(random q), solution checked in MuJoCo)")
qs = random_q(300)
near = lambda q: np.clip(q + rng.normal(0, 0.3, 7), Q_MIN, Q_MAX)
for label, seed_fn, min_rate in [
        ("seed near the answer (arm already close)", near, 0.99),
        ("seed = home keyframe", lambda q: Q_HOME, 0.97),
        ("seed = zeros (node start-up)", lambda q: np.zeros(7), 0.97)]:
    n_ok, n, p, r, lim = ik_trials(label, qs, seed_fn)
    check(f"{label}: success rate >= {min_rate:.0%}", n_ok / n >= min_rate)
    check(f"{label}: converged answers within 0.1 mm / 0.06 deg", p < 0.1 and r < 0.06)
    check(f"{label}: every returned q inside joint limits", lim)

print("\n6. Singularity, q5 = 0 (joints 4 and 6 collinear)")
qs_sing = random_q(100)
qs_sing[:, 4] = 0.0
n_ok, n, p, r, lim = ik_trials("target at q5 = 0, seed near", qs_sing, near)
check("targets at the singularity solve", n_ok / n >= 0.97)
check("no NaN, every q inside limits", lim)
n_ok, n, p, r, lim = ik_trials("seed exactly at q5 = 0", random_q(100),
                               lambda q: np.r_[q[:4], 0.0, q[5:]])
check("seeding from the singularity still solves", n_ok / n >= 0.97)
J, _ = jacobian(qs_sing[0])
print(f"  singular values of J at q5 = 0: {np.round(np.linalg.svd(J, compute_uv=False), 4)}")

print("\n7. Unreachable targets")
for label, p_target in [("1.5 m in front", [1.5, 0.0, 0.4]),
                        ("2 m above", [0.0, 0.0, 2.0])]:
    T = T_home.copy()
    T[:3, 3] = p_target
    t0 = time.perf_counter()
    q_sol, ok = inverse_kinematics(T, Q_HOME)
    dt = (time.perf_counter() - t0) * 1000
    check(f"{label}: flag is False and seed is returned", (not ok) and np.allclose(q_sol, Q_HOME),
          f"({dt:.2f} ms, rejected by the reach check)")
# Inside the reach sphere but the requested orientation cannot be made there.
T = np.eye(4)
T[:3, 3] = [0.0, 0.0, 1.15]                     # 0.817 m straight up, hand pointing up
T[:3, :3] = np.diag([1.0, -1.0, -1.0])           # ...but asked to point down
t0 = time.perf_counter()
q_sol, ok = inverse_kinematics(T, Q_HOME)
dt = (time.perf_counter() - t0) * 1000
check("in reach, impossible orientation: flag is False", not ok, f"({dt:.0f} ms, all restarts used)")

print("\n8. Redundancy: different seeds, different joints, same pose")
target = forward_kinematics(TARGET_JOINTS)
q_a, ok_a = inverse_kinematics(target, TARGET_JOINTS + np.array([0.4, 0, -0.4, 0, 0, 0, 0]))
q_b, ok_b = inverse_kinematics(target, TARGET_JOINTS + np.array([-0.4, 0, 0.4, 0, 0, 0, 0]))
pa, ra = errs(target, mujoco_hand_pose(q_a))
pb, rb = errs(target, mujoco_hand_pose(q_b))
print(f"  q_a = {np.round(q_a, 3)}\n  q_b = {np.round(q_b, 3)}")
check("both converge to the target pose", ok_a and ok_b and max(pa, pb) < 0.1 and max(ra, rb) < 0.06)
check("joint vectors differ (elbow swivel follows the seed)", np.linalg.norm(q_a - q_b) > 0.1,
      f"(|q_a - q_b| = {np.linalg.norm(q_a - q_b):.3f} rad)")

print("\n9. The node's TARGET_POSE")
print(np.round(TARGET_POSE, 4))
q_sol, ok = inverse_kinematics(TARGET_POSE, Q_HOME)
p, r = errs(TARGET_POSE, mujoco_hand_pose(q_sol))
check("IK from home reaches TARGET_POSE", ok and p < 0.1 and r < 0.06, f"({p:.4f} mm, {r:.4f} deg)")
check("hand is above the table (z > 0.1 m)", TARGET_POSE[2, 3] > 0.1)
print(f"  solution: {np.round(q_sol, 4).tolist()}")

print(f"\n{'All checks passed.' if not failures else f'{len(failures)} check(s) FAILED: {failures}'}")
sys.exit(1 if failures else 0)
