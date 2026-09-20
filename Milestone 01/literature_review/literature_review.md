# Milestone 1 — Literature Review & Project Flow

**Course:** MCTR911 – Robotics Programming, Winter 2026
**Team:** MCTR911 Team (names/IDs — TBD, see root README)
**Selected robot:** Universal Robots UR5e (6-DOF collaborative industrial manipulator)
**Selected application:** Pick-and-place / palletizing on a conveyor line

## 1. Selected industrial application

We selected a **pick-and-place / palletizing** task as our industrial robotics
application: the UR5e picks parts arriving on a conveyor and places them at
target locations (e.g., stacking them onto a pallet/tray) with defined
orientation and timing constraints. This task is one of the most widely
deployed use cases for 6-DOF collaborative arms in manufacturing and
logistics because it directly combines all the challenges targeted by this
course: kinematic modeling of the arm, trajectory generation for the end
effector between pick and place poses, and closed-loop motion control to
reliably reach each target pose despite payload changes.

## 2. Literature review

Pick-and-place and palletizing with 6-DOF articulated arms is a mature but
still actively researched area:

- **Kinematics & DH modeling.** Classical DH-parameter based forward/inverse
  kinematics (as formalized by Denavit & Hartenberg, and used throughout
  Craig's *Introduction to Robotics*) remains the standard approach for
  6-DOF serial arms like the UR5e, and is directly applicable since the
  UR5e's joint configuration is a textbook elbow-type 6R manipulator.
- **Trajectory planning for pick-and-place.** Industrial implementations
  commonly use joint-space trapezoidal/quintic polynomial trajectories for
  point-to-point moves between pick and place poses, and task-space
  (Cartesian) trajectories when the end effector must follow a constrained
  path (e.g., avoiding collisions with conveyor guards or pallet walls).
- **Control.** PID and computed-torque control at the joint level are the
  most common approaches reported for collaborative arms performing
  pick-and-place, balancing simplicity with the precision needed for
  repeatable grasp poses.
- **Simulation tooling.** MuJoCo has become a standard physics simulator for
  manipulator research and industrial digital-twin work due to its accurate
  contact dynamics (important for grasp/place contact events) and native
  Python bindings, which pair naturally with ROS2 for sensor/actuator
  integration.

## 3. Draft project flow

```
[Conveyor feeds part] -> [Vision/known pose triggers pick] -> [UR5e Inverse
Kinematics computes joint targets for pick pose] -> [Joint-space trajectory
to pick pose] -> [Gripper closes / attach] -> [Task-space trajectory to
place pose above pallet] -> [Joint-space trajectory to place pose] ->
[Gripper opens / release] -> [Return to home pose] -> [repeat]
```

This flow will be implemented incrementally across the milestones:
- **Milestone 1 (this milestone):** application + robot selection, environment
  setup (ROS2 + MuJoCo), model acquisition.
- **Milestone 2:** DH convention, forward/inverse position kinematics,
  loading the UR5e into MuJoCo via ROS2, first GUI pass.
- **Milestone 3:** velocity/acceleration kinematics, kinematics validation in
  simulation, full GUI environment (conveyor, pallet, parts).
- **Milestone 4:** task-space and joint-space trajectory generation, motor-level
  control, closed-loop validation, final pick-and-place demonstration.

## 4. Selected robot model

**Universal Robots UR5e** — 6-DOF (within the required 4-7 DOF range),
open-source MJCF model available in Google DeepMind's `mujoco_menagerie`
repository (https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e),
which we use directly with MuJoCo. It is one of the most widely deployed
collaborative arms in real industrial pick-and-place/palletizing lines,
which supports the application choice above and gives us access to a large
body of published kinematic/dynamic parameters for validation in later
milestones.

**End effector:** the UR5e ships without a gripper (real deployments bolt on
whichever end effector suits the task), so we attach a **Robotiq 2F-85**
two-finger adaptive gripper — the most common real-world pairing with the
UR5e in industrial pick-and-place cells — to the arm's built-in
`attachment_site` at the wrist flange. Its open-source MJCF model also comes
from `mujoco_menagerie`
(https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85).

## References

1. Denavit, J., & Hartenberg, R. S. (1955). A kinematic notation for
   lower-pair mechanisms based on matrices.
2. Craig, J. J. *Introduction to Robotics: Mechanics and Control*.
3. Universal Robots UR5e technical specifications, universal-robots.com.
4. Google DeepMind, *MuJoCo Menagerie*, github.com/google-deepmind/mujoco_menagerie.
5. Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for
   model-based control. IROS 2012.
