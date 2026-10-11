# Robotics Programming Challenge 1: Panda FK and IK

Forward and inverse position kinematics for the Franka Emika Panda (7 arm joints, 2 finger joints), running in MuJoCo and controlled through the course `mujoco_ros2` bridge.

```
Challenge 01/
├── panda_kinematics/            # ROS 2 ament_python package (copy into your ros2_ws/src)
│   ├── panda_kinematics/kinematics.py   # DH table, FK, IK, ROS 2 node
│   ├── setup.py                 # entry point: kinematics = panda_kinematics.kinematics:main
│   └── package.xml
├── model/panda.xml              # the exact panda.xml from mujoco_menagerie @ 71f066a (Apache-2.0)
└── verify/verify_kinematics.py  # checks FK and IK against MuJoCo, no ROS 2 needed
```

## Run it

**1. Bridge in position mode** (in the course workspace, not this repo). In `example.py` and `mujoco_node.cpp` change `{"control_mode" : "VELOCITY"}` to `{"control_mode" : "POSITION"}`, then:

```bash
colcon build --packages-select mujoco_ros2
```

**2. Build and run this package**

```bash
cp -r "Challenge 01/panda_kinematics" ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select panda_kinematics
source install/setup.bash
ros2 run panda_kinematics kinematics
```

The node waits for the first `/joint_state`, solves IK once for `TARGET_POSE`, logs the solution with an FK check of its error, and then publishes it on `/joint_commands` at 20 Hz as 9 floats: 7 joint targets, then `255.0, 0.0` for the gripper (closed).

To change the target, edit `TARGET_JOINTS` in `kinematics.py`. `TARGET_POSE` is `forward_kinematics(TARGET_JOINTS)`, so it is reachable by construction.

**3. Verify FK and IK against MuJoCo** (from the repo root, uses the project `.venv`):

```bash
.venv/bin/python "Challenge 01/verify/verify_kinematics.py"
```

## DH table (standard / Spong convention)

`A_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)`. Frame *i* sits on the axis of joint *i+1*.

| i | α_i | a_i (m) | d_i (m) | θ_i | Source in panda.xml |
|---|---|---|---|---|---|
| 1 | −π/2 | 0 | 0.333 | q1 | link1 `pos z=0.333`, link2 `quat 1 -1 0 0` = Rx(−90°) |
| 2 | +π/2 | 0 | 0 | q2 | link3 `quat 1 1 0 0` = Rx(+90°) |
| 3 | +π/2 | 0.0825 | 0.316 | q3 | link3 `pos y=-0.316` (along z after the Rx(+90°)), link4 `pos x=0.0825`, Rx(+90°) |
| 4 | −π/2 | −0.0825 | 0 | q4 | link5 `pos x=-0.0825`, Rx(−90°) |
| 5 | +π/2 | 0 | 0.384 | q5 | link5 `pos y=0.384` (along z after the Rx(−90°)), link6 Rx(+90°) |
| 6 | +π/2 | 0.088 | 0 | q6 | link7 `pos x=0.088`, Rx(+90°) |
| 7 | 0 | 0 | 0 | q7 | frame 7 is the link7 body itself |

**How it is derived.** In MuJoCo every body is placed relative to its parent by `T(pos) · R(quat)`, and its joint then turns about its own z axis. Writing the chain link0 → link7 as one product and regrouping it into `Rz · Tz · Tx · Rx` blocks gives the table. A translation along the parent's z becomes a `d`, one along the new x becomes an `a`, and each `quat` of ±90° about x becomes an `α`. No θ offsets are needed because at q = 0 every DH frame coincides with its MuJoCo body.

**link7 → hand** (`T_L7_HAND`): `<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">` is a translation of 0.107 m along z followed by a rotation of −45° about z, since (cos 22.5°, 0, 0, −sin 22.5°) is −45° about z. FK is `A1 · A2 · … · A7 · T_L7_HAND`.

**Joint limits** (rad), from the `<joint>` tags. Joints 1, 3, 5 and 7 use the class default ±2.8973.

| | q1 | q2 | q3 | q4 | q5 | q6 | q7 |
|---|---|---|---|---|---|---|---|
| min | −2.8973 | −1.7628 | −2.8973 | −3.0718 | −2.8973 | −0.0175 | −2.8973 |
| max | 2.8973 | 1.7628 | 2.8973 | −0.0698 | 2.8973 | 3.7525 | 2.8973 |

The Panda has parallel and intersecting axes, so other valid DH tables exist. The commonly published Franka table, for example, uses *modified* (Craig) DH. Any consistent table is fine as long as FK matches the simulator, and ours does (below).

## How we know FK is correct

`verify_kinematics.py` loads the same `panda.xml` into MuJoCo and compares the `hand` body pose MuJoCo computes against our `forward_kinematics` on 5 002 configurations (home keyframe, the target, and 5 000 random ones inside the limits):

| check | result |
|---|---|
| worst position error | 1e-12 mm (floating-point round-off) |
| worst orientation error | 2e-6 deg (the XML quat is written to 7 digits) |
| Jacobian vs finite differences | 4e-7 max difference |

At the home keyframe `q = (0, 0, 0, −π/2, 0, π/2, −π/4)` the hand is at (0.5545, 0, 0.6245) m with its z axis pointing straight down.

## IK method: Levenberg–Marquardt with error-scaled damping

Each iteration:

1. FK and the 6×7 geometric Jacobian in the same pass. Joint *i* turns about `z_(i−1)` through `o_(i−1)`, so its column is `[z × (p_hand − o); z]`.
2. A 6-D error `e`: the position difference, plus the orientation difference as an axis-angle vector. That vector is computed through a quaternion so it stays exact at 0° and 180°.
3. Step `dq = (JᵀJ + λI)⁻¹ Jᵀ e` with `λ = ½|e|² + 1e-4` (Sugihara's damping).
4. Joints that sit on a limit and are pushed further into it get their Jacobian column removed, and the step is recomputed so the other joints compensate.
5. Each step is capped at 0.2 rad per joint, then clamped into the limits.
6. Stop when the error is below 0.1 mm and 0.06°.

If the seed does not converge, the solve restarts from the home keyframe and then from 10 fixed pseudo-random poses. Among the restarts that converge, it returns the one closest to the seed.

### Results (300 random reachable targets, solutions checked in MuJoCo)

| seed | converged | time, mean |
|---|---|---|
| near the answer (arm already close) | 300 / 300 | ~2 ms |
| home keyframe | 298 / 300 | ~26 ms |
| all zeros | 299 / 300 | ~68 ms |
| target at singularity q5 = 0 | 100 / 100 | ~1 ms |
| seed exactly at q5 = 0 | 100 / 100 | ~1 ms |

Every returned joint vector is inside the limits. Unreachable targets come back with the flag set to `False`: 1.5 m away takes 0.01 ms (reach check), and a reachable point with an impossible orientation takes about 250 ms (all restarts used).

## Evaluation notes: the questions in Section 9

**Why a numerical method and not a closed form?** The Panda has 7 joints for a 6-D pose, so there is a one-parameter family of solutions (the elbow can swivel around the shoulder–wrist line). A closed form, such as He & Liu's analytical Panda IK, has to fix one joint by hand and then enumerate up to 8 branches per value. The numerical solver deals with the extra joint by itself, gives the solution closest to the current arm, clamps limits in the loop as the brief asks, and needs nothing beyond the FK and Jacobian we already verified.

**Which solution do we return, and why?** The one nearest the seed. LM minimises `|J dq − e|² + λ|dq|²`, so every step reduces the error with as little joint motion as it can, and the solver settles on the solution it reaches first from the seed. In the node the seed is the arm's current joint state, so the arm moves the least, and it doesn't swing across a joint limit to a far-away solution. Verify check 8 shows the same target reached from two seeds with joint vectors 1.05 rad apart.

**Singularity at q5 = 0.** At q5 = 0, joints 4 and 6 are collinear, so turning one or the other moves the hand the same way. The Jacobian loses a rank: its smallest singular value drops (0.07 in the verify output, and exactly 0 in the worst case). A plain pseudo-inverse then divides by almost zero and asks for huge joint velocities. With `+λI` the matrix stays invertible: along the lost direction the step goes to zero instead of infinity, and the other directions keep converging. The 0.2 rad step cap is a second guard. Both singularity tests converge 100/100.

**Reachability.** First a quick reject: the hand can't be more than 0.858 m from the shoulder, which is the sum of the three rigid segments (triangle inequality). That is an upper bound, so it never rejects a reachable target. Inside that sphere, a pose is declared unreachable if no seed converges.

**Limitations.**
- *Local method.* From a bad seed it can stall in a local minimum or against a limit. The restarts recover 298–299 of 300 cases from far seeds, but it is not a guarantee: a pose reported unreachable after all restarts is very likely, not provably, unreachable.
- *Cost.* Each iteration is 7 matrix products plus a 7×7 solve, O(n) for FK and O(n³) with n = 7 for the solve, which is microseconds. A solve from a nearby seed takes a few ms. The worst case is all 11 seeds × 200 iterations, about 250 ms, which is why the node solves once and caches the answer rather than re-solving every tick.
- *Accuracy* is set by the tolerance (0.1 mm, 0.06°). It could go tighter for almost no cost, since convergence near the solution is quadratic.
- *No collision model.* IK can return a configuration that puts the arm through itself or the table. MuJoCo would then block the motion.

**Likely counterarguments.**
- *"Analytical IK is faster and finds every solution."* True. We traded that for a method that handles joint limits and the redundant joint without hand-picked conventions. A few ms is far within the 50 ms tick, and the brief asks for one sensible solution, not all of them.
- *"Why not a fixed-λ damped least squares?"* We tried it first. A fixed λ leaves a steady-state error near singular configurations, and it reached only 269/300 even from a near seed. Error-scaled λ goes to almost zero at the solution, so it lands exactly: 300/300.
- *"Why not a null-space term to stay near the seed?"* We tried that too. Combined with damping, the projector is no longer exact, so the term leaks into the task and fights convergence. The LM step already penalises joint motion, which gives the "stay near the seed" behaviour without it.
- *"Why change the node below DO NOT MODIFY?"* Two small changes. It waits for the first joint state, because the zero placeholder puts q4 = 0 outside its limits. And it solves once per target, because re-solving every 50 ms from a moving seed can hop between redundant solutions and make the arm jitter.
