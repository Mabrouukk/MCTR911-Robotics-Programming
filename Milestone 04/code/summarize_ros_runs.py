"""
Collects the tracking numbers of the ROS 2 runs (results/ros_<mode>_<controller>.npz, written by
ur5e_controller_node.py) into report/ros_results.json, the same way the node's summary computes them.

    .venv/bin/python "Milestone 04/code/summarize_ros_runs.py"
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kinematics import HOME_Q, forward_kinematics  # noqa: E402
from sorting_task import plan_to_station  # noqa: E402
GRASP_LABELS = ("close the gripper", "lift", "open the gripper")   # same as ur5e_controller_node.py (which needs rclpy)

RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "report", "ros_results.json")


def errors(d):
    e = np.degrees(np.abs(d["q_d"] - d["q"]))
    f = 1000 * np.array([np.linalg.norm(forward_kinematics(a)[:3, 3] - forward_kinematics(b)[:3, 3])
                         for a, b in zip(d["q_d"], d["q"])])
    return e, f


def main():
    out = {}
    test_start = plan_to_station(HOME_Q).T
    for mode in ("joint", "task"):
        out[mode] = {}
        for ctrl in ("pd", "pdg", "ctc", "ctci"):
            path = os.path.join(RESULTS, f"ros_{mode}_{ctrl}.npz")
            if not os.path.exists(path):
                continue
            d = np.load(path)
            e, f = errors(d)
            w = d["t"] >= test_start
            out[mode][ctrl] = {"joint": float(e[w].max()), "flange": float(f[w].max()),
                               "final_flange": float(f[-1]), "ms": float(np.median(d["compute_ms"])),
                               "rate_hz": float(1 / np.median(np.diff(d["t"])))}
    path = os.path.join(RESULTS, "ros_sort_ctc.npz")
    if os.path.exists(path):
        d = np.load(path)
        e, f = errors(d)
        labels = [str(x) for x in d["labels"]]
        grasp = np.array([any(labels[i].startswith(g) for g in GRASP_LABELS) for i in d["label"]])
        moving = (d["t"] > 0.3) & ~grasp
        out["sort"] = {"time": float(d["t"][-1]), "moving_joint": float(e[moving].max()),
                       "moving_flange": float(f[moving].max()), "grasp_joint": float(e[grasp].max()),
                       "grasp_flange": float(f[grasp].max()), "ms": float(np.median(d["compute_ms"]))}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
