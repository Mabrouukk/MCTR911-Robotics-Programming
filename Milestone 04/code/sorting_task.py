"""
The colour-sorting application (Milestone 4): where the parts and bins are, and the
trajectory plan of one sorting cycle (task-space lines for picking and placing,
joint-space quintics for the transfers, see trajectories.py).
"""
import numpy as np

from kinematics import ROBOT_BASE_POS, inverse_kinematics, wrap_angle
from trajectories import Hold, JointQuintic, Plan, TaskLine

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0

# Gripper pointing straight down, fingers closing along world y (across the belt).
TOOL_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
FLANGE_TO_PINCH = 0.156  # flange to the point between the gripper pads [m]

STATION_XY = (0.105, 0.6)  # el makan el el object wa2f feh ala x,y
PART_CENTER_Z = 0.325
BIN_XY = {"red": (0.55, 0.25), "green": (0.55, -0.05), "blue": (0.55, -0.35)}

CLEAR_Z = 0.70  # flange height for moving between the conveyor and the bins
GRASP_Z = PART_CENTER_Z + FLANGE_TO_PINCH + 0.005  # pads close just above the part's centre, clear of the belt
DROP_Z = 0.52
LINE_SPEED = 0.25  # peak flange speed on the straight up / down moves [m/s]


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


def _base(xyz_world):
    return np.asarray(xyz_world, dtype=float) - ROBOT_BASE_POS


def plan_to_station(q_now):
    """From wherever the arm is to the waiting pose above the pick station (joint space)."""
    q_wait = ik_tool_down((*STATION_XY, CLEAR_Z), q_now)
    return Plan(q_now).add(JointQuintic(q_now, q_wait), GRIPPER_OPEN, "to the pick station (joint space)")


def plan_sorting_cycle(color, q_wait):
    """
    One part, from the waiting pose above the station into the bin of its colour, and back.

        straight down to the part        task space   (gripper must come straight down)
        close the gripper                hold
        straight up                      task space
        over to the bin                  joint space  (fast, no IK on the way)
        straight down into the bin       task space
        open the gripper                 hold
        straight up                      task space
        back over the station            joint space
    """
    sx, sy = STATION_XY
    bx, by = BIN_XY[color]
    plan = Plan(q_wait)
    plan.add(TaskLine(plan.q_end, _base((sx, sy, GRASP_Z)), speed=LINE_SPEED), GRIPPER_OPEN, "down to the part (task space)")
    plan.add(Hold(plan.q_end, 0.8), GRIPPER_CLOSED, "close the gripper")
    plan.add(TaskLine(plan.q_end, _base((sx, sy, CLEAR_Z)), speed=LINE_SPEED), GRIPPER_CLOSED, "lift (task space)")
    q_bin = ik_tool_down((bx, by, CLEAR_Z), plan.q_end)
    plan.add(JointQuintic(plan.q_end, q_bin), GRIPPER_CLOSED, f"over the {color} bin (joint space)")
    plan.add(TaskLine(plan.q_end, _base((bx, by, DROP_Z)), speed=LINE_SPEED), GRIPPER_CLOSED, f"down into the {color} bin (task space)")
    plan.add(Hold(plan.q_end, 0.6), GRIPPER_OPEN, "open the gripper")
    plan.add(TaskLine(plan.q_end, _base((bx, by, CLEAR_Z)), speed=LINE_SPEED), GRIPPER_OPEN, "up from the bin (task space)")
    plan.add(JointQuintic(plan.q_end, q_wait), GRIPPER_OPEN, "back to the pick station (joint space)")
    return plan


class SortingApp:
    """
    The colour-sorting application: wait above the pick station; when the colour sensor
    reports a part, plan and play one sorting cycle; repeat.

    update(t, seen_color) -> (q_d, q_dot_d, q_ddot_d, gripper, event or None)
    is called by the controller every control period; the controller turns the planned
    state into torques. Plans always start from the end of the previous plan, so the
    reference never jumps.
    """

    def __init__(self, q_start):
        self.plan = plan_to_station(np.asarray(q_start, dtype=float))
        self.q_wait = self.plan.q_end
        self.t0 = 0.0
        self.color = None                    # colour of the part being sorted, None = going to / waiting at the station
        self.sorted = {color: 0 for color in BIN_XY}
        self._label = None

    def update(self, t, seen_color):
        event = None
        tp = t - self.t0
        if tp >= self.plan.T:                                 # the current plan is done
            if self.color is not None:
                self.sorted[self.color] += 1
                event = f"{self.color} part sorted (so far: {self.sorted})"
                self.color = None
                self.plan = Plan(self.q_wait).add(Hold(self.q_wait, 0.0), GRIPPER_OPEN, "waiting for a part")
            if seen_color in BIN_XY:
                self.color = seen_color
                self.plan = plan_sorting_cycle(seen_color, self.q_wait)
                self.t0, tp = t, 0.0
                detected = f"{seen_color} part detected: cycle of {self.plan.T:.1f} s planned"
                event = f"{event}; {detected}" if event else detected
        q, dq, ddq, gripper, label, _ = self.plan.sample(tp)
        if label != self._label:
            self._label = label
            event = event or label
        return q, dq, ddq, gripper, event

    @property
    def label(self):
        return self._label
