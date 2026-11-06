# Robotics Programming Challenge 2: Panda velocity kinematics and trajectory planning

**Team 5, The Folded Approach.** The Panda starts stretched out in front of itself and folds, bringing the hand back close to its base.

We add three things to the Challenge 1 package:

- `compute_jacobian(q)`: the 3×7 linear Jacobian
- `compute_jacobian_dot(q, dq)`: its time derivative, written out analytically
- `plan_trajectory(q_start, q_end, T, dt)`: a quintic polynomial in joint space

We run the plan in both POSITION and VELOCITY control mode.

```
Challenge 02/
├── panda_kinematics/                         # the Challenge 1 package, plus:
│   ├── panda_kinematics/trajectory_node.py   # J, J-dot, planner, Q_START / Q_END (Team 5)
│   └── setup.py                              # new entry point: trajectory_node
├── sim/
│   ├── panda_sim.py          # MuJoCo on the Mac, --mode position | velocity
│   ├── panda_bridge_node.py  # ROS 2 bridge in Docker: /joint_commands -> sim -> /joint_state
│   ├── set_node_mode.py      # flips the two commented lines in tick() (POSITION <-> VELOCITY)
│   └── ros_free_import.py    # lets the sim and checks import trajectory_node without ROS 2
├── verify/verify_trajectory.py   # all the checks below + the plots
└── results/                      # plots, and the tracking logs of the last live runs
```

## Run it

```bash
scripts/run_ch2.sh position   # simulator in POSITION mode, node publishes q_traj
scripts/run_ch2.sh velocity   # simulator in VELOCITY mode, node publishes dq_traj
scripts/run_ch2.sh verify     # checks + plots only (about 4 s, no ROS 2)
NO_GRAVCOMP=1 scripts/run_ch2.sh velocity   # same, with gravity acting on the arm
```

Or in VS Code: ⌘⇧P → "Tasks: Run Task" → **Run Challenge 2** → pick a mode.

The MuJoCo window opens with the arm at `Q_START`. Yellow dots show the planned hand path, and an RGB frame marks the hand pose at `Q_END`. Once ROS starts, the node plans and sends one command every 50 ms. The simulator prints the hand's distance from the plan every second. Two seconds after the plan ends, it prints a summary and saves the log to `results/sim_<mode>.npz`.

**How the mode switch works here.** The brief says to change two things:
1. The bridge's `control_mode` in its launch file. Our stand-in for that is `panda_sim.py --mode`. In VELOCITY mode the 7 arm actuators become velocity servos (torque = kv · (q̇_cmd − q̇)). Nothing turns the command back into a position, so the physics integrates it. That is "relying on the bridge's internal integration".
2. The commented lines in `tick()`. The script runs `set_node_mode.py` on the copy it builds inside Docker. It flips exactly those 4 lines and nothing else. The file in the repo always stays in the default POSITION mode.

### On the course `mujoco_ros2` bridge

```bash
# 1. in example.py: {"control_mode": "POSITION"}  ->  {"control_mode": "VELOCITY"}
colcon build --packages-select mujoco_ros2
# 2. switch what the node publishes (or edit the two blocks in tick() by hand)
python3 "Challenge 02/sim/set_node_mode.py" velocity <ws>/src/panda_kinematics/panda_kinematics/trajectory_node.py
colcon build --packages-select panda_kinematics && source install/setup.bash
ros2 run panda_kinematics trajectory_node
```

**Watch out in VELOCITY mode.** The arm must already be at `Q_START`, because the velocities only say how far to move, not where to start. Our simulator starts the arm at `Q_START`. On the course bridge, run once in POSITION mode first: the first command sends the arm to `Q_START`. Then switch the mode.

## Our scenario: Team 5, The Folded Approach

| | joints (rad) | hand position (m) | gripper |
|---|---|---|---|
| `Q_START` (stretched) | `[0.3, 0.9, 0, -0.5, 0, 1.4, 0.785]` | (0.72, 0.22, 0.50), 0.75 m from the base axis | pointing down |
| `Q_END` (folded) | `[0, -1.0, 0, -2.95, 0, 1.95, 0.785]` | (0.25, 0, 0.40), 0.25 m from the base axis | pointing down |

How we picked them:
- **Start.** The elbow is almost straight (q4 = −0.5, its limit is −0.07). The shoulder leans forward (q2 = 0.9), and the base is turned 0.3 rad to the left.
- **End.** The shoulder leans back (q2 = −1.0) and the elbow folds to q4 = −2.95, which is 7° from its limit of −3.07. q6 is set to q2 − q4 so the gripper keeps pointing down.
- **Moving joints.** Only q1, q2, q4 and q6 move. Joints 3, 5 and 7 stay still, so the fold happens almost in one vertical plane.
- **Safety.** Both poses are inside the joint limits, and the planner refuses any pose that is not. The whole motion was checked for self-collision (section 4).

## 1. The linear Jacobian `compute_jacobian(q)`

All 7 joints turn (they are revolute). Joint *i* turns about the axis `z_{i-1}`, which passes through the point `o_{i-1}`. Both come straight from DH frame *i−1*, the product `A_1 ⋯ A_{i-1}`. If only joint *i* turns, at rate q̇_i, the hand point *p* moves on a circle around that axis, so

```
Jv_i = z_{i-1} × (p − o_{i-1})          p_dot = Jv · q_dot   (3×7)
```

**What it means physically.** Column *i* is the hand's velocity in m/s, in the base frame, when joint *i* turns at 1 rad/s and every other joint is still. So its units are m/rad.

**Deriving column 1 from the DH table.** Frame 0 is the base: `z_0 = (0,0,1)`, `o_0 = (0,0,0)`.
`Jv_1 = (0,0,1) × (px, py, pz) = (−py, px, 0)`.
Turning the base just moves the hand sideways on a circle, with no vertical part.
At `Q_START`, p = (0.718, 0.222, 0.504), so `Jv_1 = (−0.222, 0.718, 0)`, which is exactly the code's output.

**Deriving column 2.** `A_1 = Rz(q1)·Tz(0.333)·Rx(−π/2)`. The third column of `Rx(−π/2)` is (0, 1, 0), so
`z_1 = Rz(q1)·(0,1,0) = (−sin q1, cos q1, 0)` and `o_1 = (0, 0, 0.333)`, the shoulder. Then

```
Jv_2 = z_1 × (p − o_1) = ( cos q1 (pz − 0.333),  sin q1 (pz − 0.333),  −(cos q1 px + sin q1 py) )
```

Joint 2 is the shoulder pitch. It moves the hand in the vertical plane of the arm. The vertical part is minus the horizontal reach: the further out the hand, the faster it drops.
At `Q_START` (q1 = 0.3): `(0.164, 0.051, −0.751)`, again equal to the code.

**Column 7 is always zero.** Joint 7's axis runs straight through the hand origin (`T_L7_HAND` only shifts along that same axis). Turning joint 7 spins the gripper but does not move the hand point. So the linear Jacobian has only 6 useful columns for 3 rows.

**Checks:**
- matches central finite differences of FK to 3e-10
- matches MuJoCo's own `mj_jac` on the same model to 1e-15

Both were checked on 200 random poses.

## 2. The Jacobian derivative `compute_jacobian_dot(q, dq)`

Analytical. We differentiate each column with the product rule:

```
d/dt Jv_i = ż_{i-1} × (p − o_{i-1}) + z_{i-1} × (ṗ − ȯ_{i-1})
ż_{i-1} = ω_{i-1} × z_{i-1}           (the axis turns with its own frame)
ṗ = Jv q̇
```

The angular velocity `ω` of each frame, and the velocity `ȯ` of each frame origin, are built up one link at a time, going out from the base:

```
ω_k = ω_{k-1} + q̇_k z_{k-1}
ȯ_k = ȯ_{k-1} + ω_k × (o_k − o_{k-1})
```

**What it means physically.** `J̇ q̇` is the hand acceleration you get even when every joint spins at a *constant* speed. It is the centripetal and Coriolis-type part, which comes from the arm's own geometry changing:

```
p_ddot = Jv q̈ + J̇ q̇
```

**Checks:**
- matches `(J(q + h q̇) − J(q − h q̇)) / 2h` to 2e-10 (200 random q and q̇)
- along our planned motion, `Jv q̈ + J̇ q̇` matches the second derivative of the FK hand path to 4e-7 m/s²

**Why J̇ matters here.** If you leave the `J̇ q̇` term out, the hand acceleration is wrong by up to 0.26 m/s². The peak hand acceleration is only 0.30 m/s². During this fold, almost all of the hand's acceleration comes from the joints swinging the links around, not from the joints speeding up.

## 3. The trajectory `plan_trajectory(q_start, q_end, T, dt)`

A quintic in joint space, with all joints on one clock:

```
q(t) = q_start + h(t/T)·(q_end − q_start),     h(s) = 10s³ − 15s⁴ + 6s⁵
q̇(t) = (Δq / T)·30 s²(1−s)²                    q̈(t) = (Δq / T²)·60 s(1−s)(1−2s)
```

**Boundary conditions.** A 5th-order polynomial has 6 coefficients. They are fixed by 6 conditions: position, velocity and acceleration at both ends,
`q(0) = q_start, q(T) = q_end, q̇(0) = q̇(T) = 0, q̈(0) = q̈(T) = 0`.
Walking through it:
- At t = 0 the arm is still and is not accelerating. Acceleration builds up smoothly.
- At t = T/2 the velocity peaks and the acceleration crosses zero.
- At t = T everything comes back to zero together.

A cubic has only 4 coefficients. It can make the velocity zero at the ends, but not the acceleration. The acceleration then jumps from 0 to 6Δq/T² the moment the motion starts, so the motor torque jumps too, and the arm shakes.

Output: 81 samples (t = 0, 0.05, … 4.00 s). The four arrays have equal length, and the last sample is exactly `Q_END`. See [results/trajectory_profiles.png](results/trajectory_profiles.png).

**No overshoot of the joint limits.** `h'(s) = 30 s²(1−s)² ≥ 0`, so h only goes up. Every joint moves one way only and always stays between its start and end value. If both ends are inside the limits, every point in between is too. We check this anyway at 1 ms steps:
- every joint is inside its limits at every sample
- the tightest is joint 4, 0.12 rad (7°) from its limit, at the very end
- no joint ever moves backwards

**Velocity and acceleration limits.** The peaks are known in closed form: `|q̇|max = 1.875 Δq/T` and `|q̈|max = 5.77 Δq/T²`.
- If the requested T would use more than 80% of any joint's limit (Franka datasheet values), `plan_trajectory` stretches T to the shortest safe value and prints a warning.
- Our move needs at least 2.64 s. We use 4 s.
- Peak velocity is 1.15 rad/s on joint 4, which is 53% of its limit. Peak acceleration is 9% of the limit, on joint 2.

**Why joint space and not task space (straight line for the hand)?**
- **Limits.** Joint space guarantees no overshoot and gives the velocity and acceleration peaks in closed form.
- **Singularities.** It never inverts J, so it cannot blow up near a singularity.
- **Our scenario.** It is the natural fit: we care about the arm's shape while folding, not about a straight hand path.

The price: the hand path is not a straight line. It is the arc in [results/path_clearance_singularity.png](results/path_clearance_singularity.png): it rises about 9 cm, then comes down and in.

## 4. Debate focus: self-collision while folding

**How we checked.** `verify_trajectory.py` puts the arm in MuJoCo at every 2 ms sample of the plan (2 001 poses). It uses the real collision meshes from `panda.xml`. At each pose it measures, with `mj_geomDistance`, the distance between every pair of links that are not joined by a joint (196 mesh pairs), plus every moving link against the floor. It also asks MuJoCo's own contact detection.

| check | result |
|---|---|
| MuJoCo link-link contacts | 0 at all 2 001 samples |
| closest pair of any links | 2.2 cm (link5–link7, see below) ≥ 1.5 cm margin |
| **fold clearance** (closest pair, wrist pairs left out) | **5.5 cm, upper arm (link2) to forearm (link4), at t = 4 s** |
| lowest moving link above the floor | 27.8 cm |

- **Why the wrist pairs are reported separately.** link5's housing wraps around link7, and link6 around the hand. These two pairs sit 2–3 cm apart in *every* pose, so their gap tells you nothing about the fold. They are still checked against the 1.5 cm margin. The fold clearance leaves them out, so it shows how close the fold really brings the arm to itself. That gap is smallest at the end, which is when the elbow is most bent.
- **Why the samples are enough (nothing can hide between them).** A point x on the arm moves at `Σ q̇_i z_i × (x − o_i)`, so its speed is at most `Σ|q̇_i| · 1.25 m`. Two points close in on each other at most twice that fast. At our peak speeds that is 1.22 cm per 2 ms step. That is less than the 1.5 cm margin every sample keeps, so no contact can happen between two clean samples.
- **The check really catches collisions.** Take the same fold with the wrist tilted back (q6 = 0.8 instead of 1.95). The fingers go 7.3 cm into link1, and both our distance check and MuJoCo's contacts flag it.

**What we would do if the plan did collide:**
1. **Use the extra joints.** The Panda has 7 joints for a 3-D hand position, so the same hand position can be reached with a different arm shape. The q6 example above is exactly this: same fold, different wrist angle, and the collision is gone. With IK we could also search for a `Q_END` with the same hand pose and more clearance.
2. **Add a via-point.** If the ends are fine but the middle collides, plan start → via → end. Pick a via pose that keeps the links apart (for example, open the elbow before swinging the shoulder back). Join the pieces with quintics or a spline that does not stop at the via-point.
3. **For harder cases,** use a sampling planner (RRT) in joint space, with this same distance check as its collision test.
4. **On a real robot,** use a bigger margin, because the meshes are only approximate.

Note: joint-space interpolation does not care about collision at all. It is the check that makes it safe, not the method. That is why the check runs on the whole path, not only on the two end poses.

## 5. Singularities

The planner never inverts the Jacobian. q̇ comes straight from the polynomial, so even at a singularity the joint velocities stay exactly as planned. The hand just moves in whatever direction the joints happen to push it. Along our path we still watch for singularities, using the smallest singular value σ_min of Jv:

| | value |
|---|---|
| smallest σ_min of Jv on the path | 0.133 m/rad at the start (arm most stretched) |
| manipulability √det(Jv Jvᵀ) | 0.099 at start, 0.030 at the folded end |
| worst condition number | 6.8 |

Jv keeps full rank along the whole path, so there is no singularity.
- The start is the nearest point to one: a straight elbow (q4 → 0) puts the hand on the edge of the workspace, where it cannot move outward. We stop the elbow at q4 = −0.5.
- The folded end has the lowest manipulability. The links overlap, so moving the hand takes bigger joint motions.

**If we planned in task space (straight hand line), what would happen at a singularity?** The joint velocities are `q̇ = J⁺ ṗ`, and their size grows like |ṗ| / σ_min. As σ_min → 0 they shoot past the 2.175 rad/s limit. The fixes would be damped least squares (`q̇ = Jᵀ(JJᵀ + λ²I)⁻¹ ṗ`, the same idea as the λ in our Challenge 1 IK), slowing down near the singularity, or going around it.

## 6. POSITION vs VELOCITY mode: what we observed

Live runs through the full ROS 2 pipeline, plus the same thing replayed offline in `verify_trajectory.py`. Both send one command every 50 ms, like the node. Hand error is the distance between the simulated hand and the planned hand at the same moment. See [results/tracking_position_vs_velocity.png](results/tracking_position_vs_velocity.png).

| run | worst joint error during motion | worst hand error during motion | hand error at T | hand error 2 s after T |
|---|---|---|---|---|
| POSITION (live, ROS 2) | 8.0° | 37.8 mm | 0.8 mm | **0.00 mm** |
| VELOCITY (live, ROS 2) | 1.7° | **8.6 mm** | 2.1 mm | 2.08 mm, and it stays |
| VELOCITY + 5 ms timer jitter (offline) | 2.2° | 12.4 mm | 12.4 mm | 12.5 mm, and it stays |
| POSITION, no gravity compensation (offline) | 7.8° | 38.0 mm | 2.6 mm | 3.4 mm |
| VELOCITY, no gravity compensation (offline) | 16.0° | 180 mm | 127 mm | 130 mm and growing |

**Does one mode track more closely?** Yes, but it depends on *when* you look:
- **During the motion, VELOCITY mode is about 4× closer.** The position servo is a spring-damper (kp, kd). To push the joint it needs an error first, so it always trails the moving target by about kd/kp ≈ 0.1 s. At 1.15 rad/s that is about 7°. Holding each command for 50 ms adds about 1.5° more. The velocity command does not have to wait for an error: it says how fast to move *now*, so the arm keeps pace.
- **At the end, POSITION mode is exact and VELOCITY mode is not.** POSITION mode always pulls toward an absolute angle, so the lag disappears once the target stops at `Q_END`. VELOCITY mode has no idea where the arm actually is. Any small error during the motion (the servo lag, the joints pushing on each other) is integrated and never corrected. It stays as a fixed offset after the plan ends.

**Does VELOCITY mode drift or lag toward the end?** Yes. With perfect 50 ms timing the drift is small (2 mm), but it stays after the end. Any timing noise makes it worse:
- **Timer jitter.** When a ROS timer callback runs late, that velocity is held a little too long. 5 ms of jitter left 12.5 mm of permanent error.
- **Gravity.** Without gravity compensation it is far worse. The velocity servo only fights velocity *error*, so gravity pulls the arm down a bit at every instant and nothing pulls it back. The hand ends 13 cm off and keeps sinking while "holding" zero velocity. Position mode only sags 3 mm, because its spring pushes back.

**The fix if you must use VELOCITY mode:** close the loop in the node. Send `q̇_cmd = q̇_plan + K (q_plan − q_measured)`, using `/joint_state` (the "small integrator in your node" the brief mentions). Then the drift is corrected all the time. We did not do it, because the infrastructure part of the template is fixed. That is why the drift shows up in the numbers.

## 7. Likely questions and counterarguments

**"A quintic is overkill, a cubic is enough."** A cubic can't make the acceleration zero at the ends. The acceleration jumps from 0 to 6Δq/T² at t = 0 and back at t = T. On a real motor that is a torque step, which shakes the arm and the gripped object. The quintic costs nothing extra: same function, two more coefficients.

**"A trapezoidal velocity profile (LSPB) is faster."** True: it reaches the limit velocity sooner, so it finishes sooner for the same limits. But its acceleration is a square wave that jumps at 4 instants. For a slow 4-second fold, speed isn't the concern; smoothness and an easy proof of safety are.

**"Your hand path isn't a straight line."** Right, and for this scenario it doesn't need to be. The risk while folding is the arm hitting *itself*. That depends on the joint angles, and joint space controls them directly. A straight hand line could push the elbow into a worse shape, and near the stretched start it would go through `J⁺`, close to a singularity.

**"Joint-space interpolation can still collide in the middle."** Yes. That's why we check every 2 ms of the path, and not only the end poses, and why we prove that nothing can hide between samples (section 4).

**"Numerical J̇ (finite differences) would be simpler."** It would, and the brief allows it. The analytical form is exact and needs only the frames FK already computes. We still use finite differences, but as a check on the analytical version (agreement 2e-10).

**"Why stop the elbow at q4 = −2.95 and not fold all the way to the −3.07 limit?"** Two reasons. First, 0.12 rad of headroom: the position servo can overshoot slightly when it stops, and if the target sits on the limit, that overshoot hits the joint stop. Second, folding further makes link2 and link4 even closer (5.5 cm now).

**"Why did you change the simulator to start at Q_START?"** In VELOCITY mode the robot only receives velocities, so it can only end at `Q_END` if it starts at `Q_START`. In POSITION mode, starting elsewhere would make the first command a big jump. Both modes are only comparable if they start from the same place.
