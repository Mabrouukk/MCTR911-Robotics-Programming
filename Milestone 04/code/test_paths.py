"""
The two test trajectories of Milestone 4, built from the cell's own poses.

joint_test_plan   joint space: above the station -> above the green bin (quintic)
                  -> above the blue bin (LSPB / trapezoid) -> a big all-joint move that
                  also turns the wrist (quintic) -> back above the station (quintic).
task_test_plan    task space: straight line from above the station to above the red bin,
                  one horizontal circle (r = 8 cm), straight down into the bin, back up,
                  straight line back over the station. Gripper pointing down throughout.

Both start and end at the waiting pose above the pick station, at rest.
"""
import numpy as np

from kinematics import HOME_Q, ROBOT_BASE_POS
from sorting_task import BIN_XY, CLEAR_Z, DROP_Z, GRIPPER_OPEN, STATION_XY, ik_tool_down, plan_to_station
from trajectories import Hold, JointLSPB, JointQuintic, Plan, TaskCircle, TaskLine

WAIT_Q = plan_to_station(HOME_Q).q_end       # above the pick station, gripper down
SWING_Q = np.array([-0.6, -1.9, 1.2, -1.2, -0.9, 0.8])   # all six joints away from the sorting poses


def _base(xyz):
    return np.asarray(xyz, dtype=float) - ROBOT_BASE_POS


def joint_test_plan(q0=WAIT_Q):
    q_green = ik_tool_down((*BIN_XY["green"], CLEAR_Z), q0)
    q_blue = ik_tool_down((*BIN_XY["blue"], CLEAR_Z), q_green)
    plan = Plan(q0)
    plan.add(Hold(q0, 0.5), GRIPPER_OPEN, "joint test: start")
    plan.add(JointQuintic(q0, q_green), GRIPPER_OPEN, "quintic: station -> green bin")
    plan.add(JointLSPB(q_green, q_blue), GRIPPER_OPEN, "LSPB (trapezoid): green -> blue bin")
    plan.add(JointQuintic(q_blue, SWING_Q), GRIPPER_OPEN, "quintic: all six joints")
    plan.add(JointQuintic(SWING_Q, q0), GRIPPER_OPEN, "quintic: back to the station")
    plan.add(Hold(q0, 1.0), GRIPPER_OPEN, "joint test: settle")
    return plan


def task_test_plan(q0=WAIT_Q):
    plan = Plan(q0)
    plan.add(Hold(q0, 0.5), GRIPPER_OPEN, "task test: start")
    plan.add(TaskLine(plan.q_end, _base((*BIN_XY["red"], CLEAR_Z)), speed=0.3), GRIPPER_OPEN, "line: station -> red bin")
    plan.add(TaskCircle(plan.q_end, 0.08, 4.0), GRIPPER_OPEN, "circle, r = 8 cm")
    plan.add(TaskLine(plan.q_end, _base((*BIN_XY["red"], DROP_Z)), speed=0.25), GRIPPER_OPEN, "line: down into the bin")
    plan.add(TaskLine(plan.q_end, _base((*BIN_XY["red"], CLEAR_Z)), speed=0.25), GRIPPER_OPEN, "line: up from the bin")
    plan.add(TaskLine(plan.q_end, _base((*STATION_XY, CLEAR_Z)), speed=0.3), GRIPPER_OPEN, "line: back to the station")
    plan.add(Hold(plan.q_end, 1.0), GRIPPER_OPEN, "task test: settle")
    return plan


PLANS = {"joint": joint_test_plan, "task": task_test_plan}
