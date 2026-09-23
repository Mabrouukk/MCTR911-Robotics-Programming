# Milestone 1 — Literature Review & Project Flow

**Course:** MCTR911 – Robotics Programming, Winter 2026
**Team:** Mahmoud Ghobashy (58-2432), Kevin Ezzat (58-0408), Seif Hassan (58-6034),
Mohamed Elsayed (58-16392), Fawzy Karim (58-9263)
**Selected robot:** Universal Robots UR5e (6-DOF collaborative industrial arm)
**Selected application:** Pick-and-place / palletizing on a conveyor line

## 1. Selected industrial application

For this project we're going with a **pick-and-place / palletizing** task.
The idea is simple: the UR5e picks up a part as it arrives on a conveyor and
places it somewhere else, like stacking it onto a pallet in a specific
position. This is one of the most common jobs given to 6-DOF arms in real
factories and warehouses, and we picked it because it naturally covers
everything this course is about, figuring out the arm's kinematics, planning
a path from the pick point to the place point, and controlling the motors so
the arm actually follows that path reliably, even if the part's weight
changes.

## 2. Literature review

Pick-and-place with articulated arms is a well-established area, but there's
still plenty of active work in it. Here's what we found relevant to our
project:

**Kinematics and the DH convention.** Most 6-DOF arms, including the UR5e,
are still modeled using the classic Denavit-Hartenberg convention for forward
and inverse kinematics. This method has been the standard since the 1950s
and is covered extensively in Craig's *Introduction to Robotics*, which is
basically the reference textbook for this. The UR5e is a fairly "standard"
elbow-type arm, so this approach applies to it directly without needing any
special tricks.

**Trajectory planning.** For pick-and-place tasks specifically, most
real-world systems move between poses using joint-space trajectories (things
like trapezoidal or quintic polynomial profiles) when the exact path doesn't
matter, and switch to task-space (Cartesian) trajectories when it does, for
example, when the arm needs to avoid hitting a conveyor guard or the edge of
a pallet on the way.

**Control.** Most papers and real deployments we looked at use PID control or
computed-torque control at the joint level for this kind of task. Nothing
too exotic, the goal is just to reach each pick/place pose precisely and
repeatably, not to do anything fancy.

**Simulation.** MuJoCo shows up a lot in recent manipulator research and in
industrial digital-twin work, mainly because its contact physics (useful for
simulating grasping) are considered quite accurate, and because it has
native Python bindings that make it easy to hook up to ROS2.

## 3. Draft project flow

```
Part arrives on conveyor
      -> arm calculates the pick pose (inverse kinematics)
      -> arm moves to the pick pose
      -> gripper closes on the part
      -> arm moves to the place pose (above the pallet)
      -> gripper opens, part is released
      -> arm returns to its home position
      -> cycle repeats
```

We're planning to build this up gradually across the milestones rather than
all at once:
- **Milestone 1 (this one):** pick the application and the robot, get ROS2 +
  MuJoCo working, get the robot's model into the project.
- **Milestone 2:** work out the DH parameters, get forward/inverse position
  kinematics working, load the arm into MuJoCo through ROS2, start on the GUI.
- **Milestone 3:** velocity and acceleration kinematics, check our kinematics
  against the simulator, finish building the full environment (conveyor,
  pallet, parts, etc.).
- **Milestone 4:** actual trajectory generation (task-space and joint-space),
  motor-level control, and the full closed-loop pick-and-place demo.

## 4. Selected robot model

We went with the **Universal Robots UR5e**. It's a 6-DOF arm, which fits the
4-7 DOF requirement, and it's one of the most common collaborative arms
actually used in factories right now, which made it easy to find both
research papers and real specifications to reference. Instead of building
the 3D model ourselves, we're using the open-source version from Google
DeepMind's `mujoco_menagerie` repository
(https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e),
which is already set up to work directly with MuJoCo.

**Gripper.** The UR5e doesn't come with a gripper by default — in real
factories, whoever installs the arm bolts on whatever gripper fits the job.
We went with the **Robotiq 2F-85**, a two-finger gripper that's probably the
most common one paired with the UR5e in actual pick-and-place setups. We
attached it to the arm's wrist using the `attachment_site` that's already
built into the UR5e model file. Its model also comes from `mujoco_menagerie`
(https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85).

## References

1. Denavit, J., & Hartenberg, R. S. (1955). A kinematic notation for
   lower-pair mechanisms based on matrices.
2. Craig, J. J. *Introduction to Robotics: Mechanics and Control*.
3. Universal Robots UR5e technical specifications, universal-robots.com.
4. Google DeepMind, *MuJoCo Menagerie*, github.com/google-deepmind/mujoco_menagerie.
5. Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for
   model-based control. IROS 2012.
