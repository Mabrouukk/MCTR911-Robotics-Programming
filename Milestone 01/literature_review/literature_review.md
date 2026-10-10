# Milestone 1 — Literature Review & Project Flow

**Course:** MCTR911 – Robotics Programming, Winter 2026
**Team:** Mahmoud Ghobashy (58-2432), Kevin Ezzat (58-0408), Seif Hassan (58-6034),
Mohamed Elsayed (58-16392), Fawzy Karim (58-9263)
**Selected robot:** Universal Robots UR5e (6-DOF collaborative industrial arm) + Robotiq 2F-85 gripper
**Selected application:** Colour-based sorting of parts from a conveyor into bins

## 1. Selected industrial application

Our UR5e works as a **colour sorting cell** at the end of a conveyor line.
Parts of different colours (red, green and blue boxes in our simulation) arrive
in a mixed order on an accumulating conveyor and queue up against an end stop,
which is the robot's pick station. A colour sensor looks at the part waiting
at the station. The robot picks that part up and drops it into the bin that
matches its colour, then goes back for the next one, until the conveyor is
empty.

The task is a real one: sorting incoming parts by a property such as colour,
material, size or a quality result is done in recycling plants, parcel
logistics, food packaging and electronics production. We chose it because it
covers every part of the course in a concrete way. Each pick and each drop is
an inverse kinematics problem. The moves between the conveyor and the bins
are trajectories, and the robot has to track them accurately enough to grasp
a 5 cm part and release it over the right bin. The cell is also easy to
extend, for example with a reject bin for defective parts.

## 2. Literature review

**Why sorting is a robot job.** Industrial robots are being installed faster
than ever. The International Federation of Robotics counted about 542,000 new
industrial robots in 2024, more than twice as many as ten years earlier [1].
A large share of these robots move material from one place to another, and
sorting is one of the most common of those jobs, because it is repetitive,
needs a decision for every part and has to keep up with the line.

**Colour sorting from a conveyor.** Sidehabi et al. [2] built almost the same
cell as ours on a small scale: a Dobot Magician arm, a conveyor with a
photoelectric sensor that detects when a part arrives, and a camera that
classifies the part by colour using HSV colour segmentation. Over ten trials
the system sorted parts of three colours, arriving in random order, with 100%
accuracy. Their layout (a sensor that triggers the robot when a part reaches
a fixed pick point) is the one we use in our simulation.

**Sorting with a Universal Robots cobot.** Kluziak and Kohut [3] used a UR5,
the previous generation of our UR5e, with a webcam to classify objects by
shape and sort them. They compared classical image features (Hu moments,
SIFT) with small neural networks (multilayer perceptrons) and found that the
neural networks classified better. Their work shows that the decision of
*where* a part goes can be swapped between methods (colour, shape, learned
classes) without changing the robot side of the cell, which is what our
project focuses on.

**Sorting in recycling.** Koskinopoulou et al. [4] deployed a robotic system in
a real waste processing plant that recognises recyclables (such as aluminium,
paper and plastic bottles) on a conveyor with a deep-learning vision module
and physically separates them by material. Prakash et al. [5] built a similar
platform with a YOLOv8 detector and a robot arm, and reached about 82% sorting
accuracy across their test batches. Both show that the main difficulty in
real sorting is recognition in messy conditions. Our cell avoids that by
using clean colours, so that we can concentrate on the kinematics, trajectory
and control of the arm.

**Kinematics, trajectories and control.** Like most 6-DOF arms, the UR5e is
modelled with the Denavit-Hartenberg convention [6, 7], and because three of
its joint axes are parallel it has a closed-form inverse kinematics with up to
eight solutions [8]. Pick-and-place cells normally move between poses with
joint-space trajectories (for example quintic polynomials) and switch to
straight-line Cartesian moves for the final approach to the part, then track
those trajectories with PID or computed-torque control at each joint [7].

**Simulation.** MuJoCo [9] is widely used for simulating manipulators, mainly
because its contact model handles grasping well and it has Python bindings
that are easy to connect to ROS2. We use the UR5e and Robotiq 2F-85 models
from Google DeepMind's MuJoCo Menagerie [10].

## 3. Draft project flow

```
Part travels along the conveyor and stops at the end stop (pick station)
      -> colour sensor reads the part's colour (red / green / blue)
      -> robot moves above the station, then down to the part (inverse kinematics)
      -> gripper closes, robot lifts the part
      -> robot moves above the bin of that colour
      -> robot lowers the part, gripper opens, part drops into the bin
      -> robot returns above the station
      -> cycle repeats until the conveyor is empty
```

We're building this up gradually across the milestones:
- **Milestone 1:** choose the application and the robot, get ROS2 + MuJoCo
  working, get the robot's model into the project.
- **Milestone 2:** DH parameters, forward/inverse position kinematics, link
  ROS2 with MuJoCo, and the first version of the sorting cell: a moving
  conveyor, a colour sensor at the pick station, three colour bins, and the
  sorting cycle running on IK waypoints.
- **Milestone 3:** velocity and acceleration kinematics, validated against the
  simulator, and a fuller environment (more parts, a reject bin, a worker and
  other details of the cell).
- **Milestone 4:** proper task-space and joint-space trajectories,
  motor-level control (removing the arm's sag under gravity), and the full
  closed-loop sorting demo.

## 4. Selected robot model

We went with the **Universal Robots UR5e**. It's a 6-DOF arm, which fits the
4-7 DOF requirement, with a 5 kg payload and 850 mm reach, which is plenty for
light parts on a conveyor. It's one of the most common collaborative arms in
factories right now, so research papers and real specifications were easy to
find. Instead of building the 3D model ourselves, we're using the open-source
version from Google DeepMind's `mujoco_menagerie` repository
(https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e),
which is already set up to work directly with MuJoCo.

**Gripper.** The UR5e doesn't come with a gripper. In real factories, whoever
installs the arm bolts on whatever gripper fits the job. We went with the
**Robotiq 2F-85**, a two-finger gripper that is very commonly paired with the
UR5e in pick-and-place cells. Its 85 mm stroke comfortably fits our 5 cm
parts. We attached it to the arm's wrist using the `attachment_site` that's
already built into the UR5e model file. Its model also comes from
`mujoco_menagerie`
(https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85).

## References

1. International Federation of Robotics, *World Robotics 2025 – Industrial
   Robots* (press release "Global robot demand in factories doubles over 10
   years", September 2025), ifr.org.
2. S. W. Sidehabi, M. F. Azis and M. Asbar, "Development of a Color-Based
   Image Recognition System for Robotic Sorting and Picking," *INTEK: Jurnal
   Penelitian*, vol. 11, no. 2, 2024. doi:10.31963/intek.v11i2.5009
3. S. Kluziak and P. Kohut, "Development of a UR5 Cobot Vision System with MLP
   Neural Network for Object Classification and Sorting," *Information*,
   vol. 16, no. 7, art. 550, 2025. doi:10.3390/info16070550
4. M. Koskinopoulou, F. Raptopoulos, G. Papadopoulos, N. Mavrakis and
   M. Maniadakis, "Robotic Waste Sorting Technology: Toward a Vision-Based
   Categorization System for the Industrial Robotic Separation of Recyclable
   Waste," *IEEE Robotics & Automation Magazine*, vol. 28, no. 2, pp. 50–60,
   2021.
5. U. Prakash, T. Datt, A. Prasad, W. Saraqia and U. V. Mehta, "Real-Time
   Solid Waste Sorting Using a Vision-Enabled Robotic Platform," *Waste*,
   vol. 4, no. 2, art. 16, 2026. doi:10.3390/waste4020016
6. J. Denavit and R. S. Hartenberg, "A kinematic notation for lower-pair
   mechanisms based on matrices," *Journal of Applied Mechanics*, 1955.
7. J. J. Craig, *Introduction to Robotics: Mechanics and Control*, Pearson.
8. K. P. Hawkins, "Analytic Inverse Kinematics for the Universal Robots
   UR-5/UR-10 Arms," Georgia Institute of Technology, 2013.
9. E. Todorov, T. Erez and Y. Tassa, "MuJoCo: A physics engine for
   model-based control," IROS 2012.
10. Google DeepMind, *MuJoCo Menagerie*, github.com/google-deepmind/mujoco_menagerie.
