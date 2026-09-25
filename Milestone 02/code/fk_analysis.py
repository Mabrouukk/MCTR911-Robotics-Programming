import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from kinematics import A, ALPHA, D, HOME_Q, forward_kinematics, inverse_kinematics, wrap_angle

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_XML = os.path.join(HERE, "..", "..", "Milestone 01", "cad_models", "ur5e", "ur5e.xml")
FIGURES = os.path.join(HERE, "..", "report", "figures")


def print_dh_table():
    print("DH parameters (standard convention)")
    print(f"{'i':>2} {'theta_i':>8} {'d_i [m]':>8} {'a_i [m]':>8} {'alpha_i':>8}")
    for i in range(6):
        print(f"{i + 1:>2} {'q' + str(i + 1):>8} {D[i]:>8.3f} {A[i]:>8.3f} {np.degrees(ALPHA[i]):>7.0f}°")


def verify_against_mujoco():
    model = mujoco.MjModel.from_xml_path(ARM_XML)
    data = mujoco.MjData(model)
    site = model.site("attachment_site").id
    rng = np.random.default_rng(7)
    poses = [("zero", np.zeros(6)), ("home", HOME_Q)] + [(f"random {k}", rng.uniform(-np.pi, np.pi, 6)) for k in range(1, 4)]

    print("\nForward kinematics vs MuJoCo (end-effector = gripper mounting flange)")
    for name, q in poses:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        T = forward_kinematics(q)
        pos_err = np.linalg.norm(T[:3, 3] - data.site_xpos[site]) * 1000
        rot_err = np.abs(T[:3, :3] - data.site_xmat[site].reshape(3, 3)).max()
        print(f"{name:>9}: q = {np.round(np.degrees(q)).astype(int)} deg")
        print(f"{'':>11}FK = {np.round(T[:3, 3], 4)}  MuJoCo = {np.round(data.site_xpos[site], 4)}"
              f"  |pos err| = {pos_err:.2e} mm  max rot err = {rot_err:.1e}")


def verify_inverse_kinematics(n=2000):
    rng = np.random.default_rng(42)
    counts, worst, recovered = [], 0.0, 0
    for _ in range(n):
        q = rng.uniform(-np.pi, np.pi, 6)
        T = forward_kinematics(q)
        solutions = inverse_kinematics(T)
        counts.append(len(solutions))
        worst = max([worst] + [np.abs(forward_kinematics(s) - T).max() for s in solutions])
        recovered += any(np.abs(wrap_angle(s - q)).max() < 1e-6 for s in solutions)
    print(f"\nInverse kinematics round trip over {n} random poses:")
    print(f"  solutions per pose: {min(counts)}-{max(counts)} (mean {np.mean(counts):.2f})")
    print(f"  worst |FK(IK(T)) - T|: {worst:.1e}")
    print(f"  original joint angles recovered: {recovered}/{n}")


def _draw_frames(ax, frames, which, length, span_margin):
    origins = np.array([T[:3, 3] for T in frames])
    ax.plot(*origins.T, color="0.5", linewidth=6, alpha=0.5, solid_capstyle="round")
    for i in which:
        T = frames[i]
        o = T[:3, 3]
        for axis, color in zip(range(3), ("red", "green", "blue")):
            ax.quiver(*o, *(T[:3, axis] * length), color=color, linewidth=2, arrow_length_ratio=0.25)
        ax.scatter(*o, color="black", s=18, depthshade=False)
        ax.text(*(o + length * np.array([-0.2, -0.6, 0.4])), f"{{{i}}}", fontsize=12, weight="bold")
    shown = origins[list(which)]
    span = np.ptp(shown, axis=0).max() / 2 + span_margin
    center = (shown.max(axis=0) + shown.min(axis=0)) / 2
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x0 [m]")
    ax.set_ylabel("y0 [m]")
    ax.set_zlabel("z0 [m]")
    ax.view_init(elev=22, azim=-60)


def plot_frames(q, filename, title):
    frames = forward_kinematics(q, return_all=True)
    fig = plt.figure(figsize=(13, 6))
    full = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_frames(full, frames, range(7), length=0.07, span_margin=0.1)
    full.set_title("Whole arm, frames {0} to {6}")
    wrist = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_frames(wrist, frames, range(3, 7), length=0.04, span_margin=0.06)
    wrist.set_title("Wrist close-up, frames {3} to {6}")
    fig.suptitle(f"{title}   (x: red, y: green, z: blue)", fontsize=13)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {os.path.relpath(path, HERE)}")


def main():
    print_dh_table()
    verify_against_mujoco()
    verify_inverse_kinematics()
    print()
    plot_frames(np.zeros(6), "dh_frames_zero.png", "UR5e DH frames, zero configuration (all joints at 0)")
    plot_frames(HOME_Q, "dh_frames_home.png", "UR5e DH frames, home configuration")


if __name__ == "__main__":
    main()
