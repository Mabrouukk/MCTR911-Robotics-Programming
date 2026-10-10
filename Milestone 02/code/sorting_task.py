import numpy as np

from kinematics import HOME_Q, ROBOT_BASE_POS, inverse_kinematics, wrap_angle

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0

# Gripper pointing straight down, fingers closing along world y (across the belt).
TOOL_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
FLANGE_TO_PINCH = 0.156  # flange to the point between the gripper pads [m]

STATION_XY = (0.105, 0.6)  # where parts stop against the conveyor's end stop
PART_CENTER_Z = 0.325
BIN_XY = {"red": (0.55, 0.25), "green": (0.55, -0.05), "blue": (0.55, -0.35)}

CLEAR_Z = 0.70  # flange height for moving between the conveyor and the bins
GRASP_Z = PART_CENTER_Z + FLANGE_TO_PINCH + 0.005  # pads close just above the part's centre, clear of the belt
DROP_Z = 0.52


def ik_tool_down(xyz_world, reference_q):
    """Joint angles that put the wrist flange at xyz_world with the gripper pointing down.

    Picks the IK solution nearest to reference_q and unwraps it so the arm takes the short way round.
    """
    T = np.eye(4)
    T[:3, :3] = TOOL_DOWN
    T[:3, 3] = np.asarray(xyz_world, dtype=float) - ROBOT_BASE_POS
    solutions = inverse_kinematics(T)
    if len(solutions) == 0:
        raise ValueError(f"Target {tuple(xyz_world)} is out of the UR5e's reach")
    deltas = wrap_angle(solutions - reference_q)
    return reference_q + deltas[np.argmin(np.abs(deltas).sum(axis=1))]


def sorting_cycle(color, start_q):
    """Waypoints that move one part from the pick station into the bin of its colour.

    Returns a list of (label, q, gripper, duration [s]); the arm blends to each q over its duration.
    """
    sx, sy = STATION_XY
    bx, by = BIN_XY[color]
    plan = [
        ("above station", (sx, sy, CLEAR_Z), GRIPPER_OPEN, 2.0),
        ("down to part", (sx, sy, GRASP_Z), GRIPPER_OPEN, 1.5),
        ("close gripper", (sx, sy, GRASP_Z), GRIPPER_CLOSED, 1.0),
        ("lift", (sx, sy, CLEAR_Z), GRIPPER_CLOSED, 1.5),
        (f"above {color} bin", (bx, by, CLEAR_Z), GRIPPER_CLOSED, 2.5),
        (f"down into {color} bin", (bx, by, DROP_Z), GRIPPER_CLOSED, 1.0),
        ("open gripper", (bx, by, DROP_Z), GRIPPER_OPEN, 0.8),
        ("back above station", (sx, sy, CLEAR_Z), GRIPPER_OPEN, 2.5),
    ]
    steps, q = [], np.asarray(start_q, dtype=float)
    for label, xyz, gripper, duration in plan:
        q = ik_tool_down(xyz, q)
        steps.append((label, q, gripper, duration))
    return steps


class SortingController:
    """Waits at the pick station, and when the colour sensor reports a part, runs one sorting cycle.

    Call update() at a fixed rate with the time and the colour currently seen ("" if none);
    it returns the joint command, the gripper command and the label of a step that just started (or None).
    """

    def __init__(self, start_q=HOME_Q):
        self.q = np.asarray(start_q, dtype=float)
        self.gripper = GRIPPER_OPEN
        self.sorted = {color: 0 for color in BIN_XY}
        self._steps = None
        self._color = None

    def update(self, t, seen_color):
        started = None
        if self._steps is None:
            if seen_color in BIN_XY:
                self._color = seen_color
                self._steps = sorting_cycle(seen_color, self.q)
                self._from_q, self._step, self._step_t0 = self.q, 0, t
                started = f"{seen_color} part detected -> {self._steps[0][0]}"
            else:
                return self.q, self.gripper, None

        label, target, gripper, duration = self._steps[self._step]
        s = min((t - self._step_t0) / duration, 1.0)
        blend = 0.5 - 0.5 * np.cos(np.pi * s)  # smooth start and stop
        self.q = self._from_q + blend * (target - self._from_q)
        self.gripper = gripper

        if s >= 1.0:
            self._step += 1
            self._from_q, self._step_t0 = target, t
            if self._step == len(self._steps):
                self.sorted[self._color] += 1
                self._steps = None
                started = f"{self._color} part sorted (sorted so far: {self.sorted})"
            else:
                started = self._steps[self._step][0]
        return self.q, self.gripper, started
