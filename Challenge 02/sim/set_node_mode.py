#!/usr/bin/env python3
"""
Switch trajectory_node.py between POSITION and VELOCITY publishing.

Does exactly what the brief asks you to do by hand in tick(): comment one of
the two lines in each "UNCOMMENT ONE BELOW" block and uncomment the other.
Nothing else in the file changes.

    python3 set_node_mode.py velocity path/to/trajectory_node.py
    python3 set_node_mode.py position path/to/trajectory_node.py

run_ch2.sh runs this on the copy it builds inside Docker, so the file in the
repo always stays in the default POSITION mode.
"""
import sys

POSITION_LINES = [
    "            cmd.data = Q_END.tolist() + [GRIPPER_CTRL, 0.0]",
    "        q_target = self.q_traj[self.step_idx]",
]
VELOCITY_LINES = [
    "            cmd.data = [0.0] * 7 + [GRIPPER_CTRL, 0.0]",
    "        q_target = self.dq_traj[self.step_idx]",
]


def comment(line):
    indent = len(line) - len(line.lstrip())
    return line[:indent] + "# " + line[indent:]


def set_mode(text, mode):
    on, off = (POSITION_LINES, VELOCITY_LINES) if mode == "position" else (VELOCITY_LINES, POSITION_LINES)
    lines = text.split("\n")
    for target in on + off:
        hits = [i for i, l in enumerate(lines) if l in (target, comment(target))]
        if len(hits) != 1:
            raise SystemExit(f"Expected the line once, found it {len(hits)} times: {target.strip()}")
        lines[hits[0]] = target if target in on else comment(target)
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("position", "velocity"):
        raise SystemExit(__doc__)
    path = sys.argv[2]
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(set_mode(text, sys.argv[1]))
    print(f"{path}: now publishing {'q_traj' if sys.argv[1] == 'position' else 'dq_traj'} "
          f"({sys.argv[1].upper()} mode)")
