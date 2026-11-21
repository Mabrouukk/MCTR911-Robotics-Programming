"""
Challenge 3 (a): our analytical M q_ddot + C q_dot + g against MuJoCo's own dynamics.

    .venv/bin/python "Challenge 03/verify/verify_dynamics.py"

For random q, q_dot, q_ddot (inside the joint limits) we put the same state into MuJoCo and compare:
    M(q)                  vs  mj_fullM                       (the mass matrix)
    C(q, q_dot) q_dot + g vs  qfrc_bias                      (Coriolis / centrifugal + gravity)
    g(q)                  vs  qfrc_bias at q_dot = 0
    M q_ddot + C q_dot + g vs  mj_inverse (qfrc_inverse) minus the joint damping 1 N m s/rad x q_dot
                              (MuJoCo's inverse dynamics also pays for friction; our Lagrangian has none)
plus two checks of our own: closed-form dM/dq vs finite differences, and M_dot - 2C skew-symmetric.
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
from ros_free_import import load_controller_node  # noqa: E402

cn = load_controller_node()
MODEL = os.path.join(HERE, "..", "..", "Challenge 01", "model", "panda.xml")
RESULTS = os.path.join(HERE, "..", "results")
N = 1000


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)
    rng = np.random.default_rng(3)
    worst = dict.fromkeys(("M", "Cqd_g", "g", "tau", "dMdq", "skew"), 0.0)
    biggest = 0.0
    example = None
    for n in range(N):
        q = rng.uniform(cn.Q_MIN, cn.Q_MAX)
        dq = rng.uniform(-cn.QD_MAX, cn.QD_MAX)
        ddq = rng.uniform(-cn.QDD_MAX, cn.QDD_MAX)
        data.qpos[:], data.qvel[:], data.qacc[:] = 0, 0, 0
        data.qpos[7:9] = 0.04                       # fingers open, as in our lumped hand
        data.qpos[:7], data.qvel[:7], data.qacc[:7] = q, dq, ddq
        mujoco.mj_inverse(model, data)
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, full)
        M, h = cn.compute_dynamics(q, dq)
        tau = M @ ddq + h
        tau_mujoco = data.qfrc_inverse[:7] - cn.JOINT_DAMPING * dq
        worst["M"] = max(worst["M"], np.abs(full[:7, :7] - M).max())
        worst["Cqd_g"] = max(worst["Cqd_g"], np.abs(data.qfrc_bias[:7] - h).max())
        worst["tau"] = max(worst["tau"], np.abs(tau_mujoco - tau).max())
        biggest = max(biggest, np.abs(tau).max())
        if n == 0:
            example = {"q": q.round(3).tolist(), "dq": dq.round(3).tolist(), "ddq": ddq.round(3).tolist(),
                       "ours": tau.round(4).tolist(), "mujoco": tau_mujoco.round(4).tolist()}
        data.qvel[:7] = 0
        mujoco.mj_inverse(model, data)
        worst["g"] = max(worst["g"], np.abs(data.qfrc_bias[:7] - cn.gravity_vector(q)).max())
    for _ in range(100):
        q, dq = rng.uniform(cn.Q_MIN, cn.Q_MAX), rng.uniform(-2, 2, 7)
        D = cn.mass_matrix_derivatives(q)
        for k in range(7):
            e = np.zeros(7)
            e[k] = 1e-6
            fd = (cn.mass_matrix(q + e) - cn.mass_matrix(q - e)) / 2e-6
            worst["dMdq"] = max(worst["dMdq"], np.abs(fd - D[k]).max())
        Nm = np.einsum("kij,k->ij", D, dq) - 2 * cn.coriolis_matrix(q, dq)
        worst["skew"] = max(worst["skew"], np.abs(Nm + Nm.T).max())

    print(f"Challenge 3: our Lagrangian dynamics vs MuJoCo, {N} random states (q, q_dot, q_ddot inside the limits)")
    print(f"  M(q)                       vs mj_fullM            worst {worst['M']:.1e} kg m^2")
    print(f"  C(q,q_dot) q_dot + g(q)    vs qfrc_bias           worst {worst['Cqd_g']:.1e} N m")
    print(f"  g(q)                       vs qfrc_bias, q_dot=0  worst {worst['g']:.1e} N m")
    print(f"  M q_ddot + C q_dot + g     vs mj_inverse          worst {worst['tau']:.1e} N m  (torques up to {biggest:.0f} N m)")
    print(f"  dM/dq closed form vs finite differences            worst {worst['dMdq']:.1e}")
    print(f"  M_dot - 2C skew-symmetric                          worst {worst['skew']:.1e}")
    print("  (Why 1e-8 and not 1e-15: MuJoCo turns every <fullinertia> into principal axes with an iterative")
    print("   eigen-solver, and the inertia it keeps differs from the XML numbers by up to 5e-7 kg m^2, e.g.")
    print("   link 1's Iyz becomes 0.0191694570 instead of 0.019169. Our model uses the XML numbers exactly.)")
    print(f"  example state: ours {example['ours']}")
    print(f"                 MuJoCo {example['mujoco']}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "dynamics_check.json"), "w") as f:
        json.dump({k: float(v) for k, v in worst.items()} | {"largest_torque": float(biggest), "example": example}, f, indent=2)


if __name__ == "__main__":
    main()
