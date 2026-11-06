"""
Symbolic derivation of the UR5e velocity and acceleration kinematics (sympy).

Steps, the same ones the report walks through:
  1. Build T_0^6 from the DH table with symbolic joint angles.
  2. Velocity kinematics, two ways that must agree:
       a. geometric:     J_i = [z_(i-1) x (p - o_(i-1)); z_(i-1)]
       b. differentiate: J_v = d p / d q   (the linear rows only)
  3. Acceleration kinematics: J_dot = sum_k (dJ/dq_k) q_dot_k.
  4. Turn the symbolic results into numbers and compare them with the
     functions in velocity_kinematics.py, at random joint values.
  5. Print the short closed forms used in the report.

    .venv/bin/python "Milestone 03/code/derive_symbolic.py"     (about 5 s)

Writes the printed derivation to Milestone 03/report/derivation_output.txt.
"""
import io
import os
import sys
import time

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import velocity_kinematics as vk  # noqa: E402
from kinematics import A, ALPHA, D  # noqa: E402

OUT = os.path.join(HERE, "..", "report", "derivation_output.txt")
log = io.StringIO()


def say(*parts):
    text = " ".join(str(p) for p in parts)
    print(text)
    log.write(text + "\n")


q = sp.symbols("q1:7", real=True)
dq = sp.symbols("qd1:7", real=True)
d1, a2, a3, d4, d5, d6 = sp.symbols("d1 a2 a3 d4 d5 d6", real=True)
D_SYM = [d1, 0, 0, d4, d5, d6]
A_SYM = [0, a2, a3, 0, 0, 0]
ALPHA_SYM = [sp.pi / 2, 0, 0, sp.pi / 2, -sp.pi / 2, 0]
NUMBERS = {d1: D[0], a2: A[1], a3: A[2], d4: D[3], d5: D[4], d6: D[5]}


def dh(theta, d, a, alpha):
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ])


t_start = time.time()
say("=" * 72)
say("1. Forward position kinematics, T_0^6 = A1 A2 A3 A4 A5 A6")
say("=" * 72)
say("DH table (standard): i | theta | d | a | alpha")
for i in range(6):
    say(f"  {i + 1} | q{i + 1} | {D_SYM[i]} | {A_SYM[i]} | {ALPHA_SYM[i]}")
frames = [sp.eye(4)]
for i in range(6):
    frames.append(sp.simplify(frames[-1] * dh(q[i], D_SYM[i], A_SYM[i], ALPHA_SYM[i])))
p = frames[6][:3, 3]
say("\nFlange position p = (px, py, pz):")
for name, expr in zip(("px", "py", "pz"), p):
    say(f"  {name} = {sp.simplify(expr)}")

say("\n" + "=" * 72)
say("2. Velocity kinematics: the geometric Jacobian")
say("=" * 72)
J = sp.zeros(6, 6)
for i in range(6):
    z = frames[i][:3, 2]
    o = frames[i][:3, 3]
    J[:3, i] = z.cross(p - o)
    J[3:, i] = z
say("Column i: J_i = [ z_(i-1) x (p - o_(i-1)) ; z_(i-1) ]")
say("Joint axes z_(i-1) (third column of T_0^(i-1)):")
for i in range(6):
    say(f"  z{i} = {list(sp.simplify(frames[i][:3, 2]))}")
say("\nColumn 1 (base joint, z0 = (0,0,1) through the origin):")
say(f"  J_1 = {list(sp.simplify(J[:, 0]))}   i.e. (-py, px, 0, 0, 0, 1)")
say("Column 6 (last joint): its axis passes through the flange point, so")
say(f"  J_6 = {list(sp.simplify(J[:, 5]))}   (linear part is zero: joint 6 only spins the flange)")

say("\nCheck: the linear rows equal d p / d q (differentiating the FK directly):")
Jv_diff = p.jacobian(q)
f_geo = sp.lambdify((q,), J[:3, :].subs(NUMBERS), "numpy")
f_diff = sp.lambdify((q,), Jv_diff.subs(NUMBERS), "numpy")
rng = np.random.default_rng(3)
worst = max(np.abs(np.array(f_geo(x), float) - np.array(f_diff(x), float)).max()
            for x in rng.uniform(-np.pi, np.pi, (50, 6)))
say(f"  max |z x (p - o) - dp/dq| over 50 random q: {worst:.1e}")

say("\nForward velocity:  [v; w] = J(q) q_dot")
say("Inverse velocity:  q_dot = J(q)^-1 [v; w]   (J is 6x6, so it has an inverse away from singularities)")

say("\n" + "=" * 72)
say("3. Singularities: where det J = 0")
say("=" * 72)
say("det J = a2 a3 sin(q3) sin(q5) (a2 cos(q2) + a3 cos(q2+q3) + d5 sin(q2+q3+q4))")
f_J = sp.lambdify((q,), J.subs(NUMBERS), "numpy")
worst = max(abs(np.linalg.det(np.array(f_J(x), float)) - vk.determinant_closed_form(x))
            for x in rng.uniform(-np.pi, np.pi, (200, 6)))
say(f"  closed form vs det of the symbolic J at 200 random q: max difference {worst:.1e}")
say("  elbow singularity:    sin(q3) = 0  (arm stretched straight or folded flat)")
say("  wrist singularity:    sin(q5) = 0  (axes of joints 4 and 6 line up)")
say("  shoulder singularity: a2 cos(q2) + a3 cos(q2+q3) + d5 sin(q2+q3+q4) = 0 (wrist centre on the joint 1 axis)")

say("\n" + "=" * 72)
say("4. Acceleration kinematics")
say("=" * 72)
say("Differentiate x_dot = J q_dot in time:  x_ddot = J q_ddot + J_dot q_dot")
say("J_dot = sum over k of (dJ/dq_k) * qd_k   (chain rule, J depends on time only through q)")
J_dot = sp.zeros(6, 6)
for k in range(6):
    J_dot += J.diff(q[k]) * dq[k]
say("Column 1 of J_dot (base joint):")
say(f"  {[sp.simplify(e) for e in J_dot[:, 0]]}")
f_Jdot = sp.lambdify((q, dq), J_dot.subs(NUMBERS), "numpy")
worst = 0.0
for _ in range(50):
    x, xd = rng.uniform(-np.pi, np.pi, 6), rng.uniform(-1.5, 1.5, 6)
    worst = max(worst, np.abs(np.array(f_Jdot(x, xd), float) - vk.jacobian_dot(x, xd)).max())
say(f"  symbolic J_dot vs velocity_kinematics.jacobian_dot (recursive formula), 50 random (q, q_dot): {worst:.1e}")
say("Forward acceleration:  [a; alpha] = J q_ddot + J_dot q_dot")
say("Inverse acceleration:  q_ddot = J^-1 ([a; alpha] - J_dot q_dot)")

say("\n" + "=" * 72)
say("5. Symbolic J vs velocity_kinematics.jacobian (numbers)")
say("=" * 72)
worst = max(np.abs(np.array(f_J(x), float) - vk.jacobian(x)).max() for x in rng.uniform(-np.pi, np.pi, (50, 6)))
say(f"  max difference over 50 random q: {worst:.1e}")
say(f"\nDone in {time.time() - t_start:.0f} s.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(log.getvalue())
print(f"Saved {os.path.relpath(OUT)}")
