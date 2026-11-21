# Milestone 4: UR5e trajectories, dynamics and torque control

Team 5, colour-sorting cell with a UR5e and a Robotiq 2F-85 gripper. Due Wednesday 25 November 2026.

## What the milestone asks, and where it is

| The brief asks | Where |
|---|---|
| Task-space trajectories for the end effector | [code/trajectories.py](code/trajectories.py): `TaskLine` (straight line + orientation), `TaskCircle` |
| Joint-space trajectories for the joints | same file: `JointQuintic`, `JointLSPB` (trapezoid), `quintic_coefficients` (the 6×6 solve) |
| Control algorithms for the motor-level control | [code/control.py](code/control.py): PD, PD + gravity compensation, computed torque (+ integral), built on the Lagrangian model in [code/dynamics.py](code/dynamics.py) |
| Validate the task-space and joint-space trajectories on ROS 2 and the simulator | [code/ur5e_controller_node.py](code/ur5e_controller_node.py) `task` / `joint` modes, through [code/mujoco_bridge_node.py](code/mujoco_bridge_node.py) and [code/sim_bridge.py](code/sim_bridge.py) |
| Send the joint angles of both trajectories to the simulator, in the GUI | the controller plays them in the Milestone 3 cell; a green ball shows the planned flange, the panel shows error and torques |
| Test the closed loop with the visualization | the simulator is in **torque mode** with gravity on; the loop is closed through ROS 2 at 250 Hz |
| Perform the industrial application | `sort` mode: all six parts sorted under computed-torque control |
| Report with all steps | [report/](report/) |
| Videos of trajectory and control validation, commented | [videos/ms4_trajectory_control.mp4](videos/ms4_trajectory_control.mp4), [videos/ms4_sorting_application.mp4](videos/ms4_sorting_application.mp4) |

## Run it

```bash
scripts/run_ms4.sh                 # colour sorting, computed torque (default)
scripts/run_ms4.sh joint ctc       # joint-space test trajectory, summary at the end
scripts/run_ms4.sh task ctc        # task-space test trajectory, summary at the end
scripts/run_ms4.sh joint pd        # same with PD only (watch the arm sag), or pdg / ctci
PAYLOAD=2 scripts/run_ms4.sh joint ctci   # hidden 2 kg in the gripper, computed torque + integral
scripts/run_ms4.sh validate        # every offline test + the report figures, no ROS 2 (about 2 min)
```

Controllers: `ctc` computed torque · `ctci` computed torque + integral · `pdg` PD + gravity compensation · `pd` PD only.

**What you'll see:** the cell from Milestone 3. The arm starts on its brakes. When the controller node starts, the brakes release and the arm is held up only by the torques it sends. The green see-through ball is where the plan wants the flange. The panel (top right) shows the controller, the step, the joint and flange error, and the six motor torques. Every test run prints a tracking summary and saves its log to `results/ros_<mode>_<controller>.npz`.

## How it fits together

```
 ur5e_controller_node.py (ROS 2, Docker)                     sim_bridge.py (MuJoCo, Mac)
   plan:  trajectories.py / sorting_task.py                    torque motors, gravity ON,
   law:   control.py  (uses dynamics.py)                       brakes when no commands
          |                                                          ^
          |  /ur5e/effort_commands  (6 torques + gripper)            |  TCP
          v                                                          |
   mujoco_bridge_node.py  <---------------------------------------->
          |  /ur5e/joint_states  (q, q_dot, applied torque), 250 Hz, sim time stamps
          +--> back to the controller: one torque command for every state
```

## The idea, in short

**Trajectories.** Every segment answers "at time t, what should q, q̇, q̈ be?".
- *Joint space:* all six joints go from q0 to q1 on one clock. Quintic `h(s) = 10s³ − 15s⁴ + 6s⁵` (q, q̇, q̈ fixed at both ends, so 6 conditions for 6 coefficients), or LSPB (trapezoid in velocity). The duration comes from the joint limits: peak speed is (15/8)·Δq/T, peak acceleration (10/√3)·Δq/T².
- *Task space:* the flange moves on a straight line (or a circle) with the same quintic time law. The joints follow from Milestone 3: `q = IK(p, R)`, `q̇ = J⁻¹[v; ω]`, `q̈ = J⁻¹([a; α] − J̇q̇)`.
- *In the sorting:* straight down / up to the part and the bin are task-space lines (the gripper must not swing into the part or the bin wall). The transfers are joint-space quintics (fast, no IK on the way, no singularity risk).

**Dynamics (Lagrangian).** `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ`, see [code/dynamics.py](code/dynamics.py).
- *Kinetic energy:* each link's CoM moves with `v = Jv_i q̇` and spins with `ω = Jω_i q̇`, so `K = ½ q̇ᵀ M q̇` with `M = Σ (m Jvᵀ Jv + Jωᵀ R I Rᵀ Jω) + diag(rotor inertia)`.
- *Potential energy:* `P = Σ m g z_c`, so `g(q) = ∂P/∂q = Σ m g (row z of Jv_i)`.
- *C:* from the Christoffel symbols of M. We compute `∂M/∂q_k` in closed form: turning joint k rotates everything after it about its axis.
- *Numbers:* masses, CoMs and inertias from ur5e.xml / 2f85.xml, moved into our DH frames. The whole gripper is one rigid body of 1.05 kg on link 6.

**Control.** Errors `e = q_d − q`, `ė = q̇_d − q̇`.
- *PD:* `τ = Kp e + Kd ė`. Gravity is a constant load, so the arm sags until `Kp e` equals it.
- *PD + gravity compensation:* `τ = Kp e + Kd ė + g(q)`. No sag at rest, but it still lags while moving (the PD has to supply the inertia torques).
- *Computed torque:* `τ = M(q)(q̈_d + Kd ė + Kp e) + C q̇ + g`. Put into the dynamics, the model cancels and every joint becomes `ë + Kd ė + Kp e = 0`, a linear decoupled system. Kp = ωn² = 400, Kd = 2ωn = 40 (critically damped, settles in about 0.2 s).
- *+ integral:* adds `Ki ∫e` (Ki = 2000, stable since Kd·Kp > Ki) for loads the model doesn't know.

## Results

Offline (control every 2 ms, [report/validation_results.json](report/validation_results.json)) and through ROS 2 (250 Hz loop, Docker on the Mac):

| Test | Result |
|---|---|
| Our M(q), C q̇ + g, g(q) and M q̈ + C q̇ + g vs MuJoCo, 1000 random states | agree to 6·10⁻¹², 1·10⁻¹⁰, 3·10⁻¹¹, 1·10⁻¹⁰ N·m (torques up to 125 N·m) |
| Closed-form ∂M/∂q vs finite differences; Ṁ − 2C skew-symmetric | 1·10⁻⁹; 4·10⁻¹⁵ |
| Quintic boundary conditions; 6×6 coefficient solve vs closed form | 1·10⁻¹⁶; 1·10⁻¹⁴ |
| Task-space plan: FK, J q̇, J q̈ + J̇ q̇ of the planned joints vs the planned flange motion | 4·10⁻¹⁵ |

Worst error (joint / flange) during the test trajectories:

| Trajectory | PD only | PD + gravity | Computed torque |
|---|---|---|---|
| joint space, offline | 2.72° / 47.7 mm | 0.46° / 6.1 mm | **0.017° / 0.21 mm** |
| joint space, via ROS 2 | 2.72° / 47.8 mm | 0.45° / 6.1 mm | **0.027° / 0.33 mm** |
| task space, offline | 2.45° / 44.1 mm | 0.19° / 2.2 mm | **0.011° / 0.08 mm** |
| task space, via ROS 2 | 2.45° / 43.4 mm | 0.19° / 2.2 mm | **0.021° / 0.15 mm** |
| error left at the end | 43 mm (sag) | 0 | 0 |

The ROS 2 numbers change a little from run to run (the loop's timing does). A repeat of the computed-torque runs gave 0.33 mm (joint space), 0.10 mm (task space) and 0.63 mm while sorting.

Robustness (joint-space test, error left at the end):

| What is wrong with the model | PD + gravity | Computed torque | Computed torque + integral |
|---|---|---|---|
| nothing | 0 | 0 | 0 |
| hidden 2 kg payload | 19 mm | 12 mm | **0.002 mm** |
| upper arm + forearm 20 % heavier | 2.3 mm | 1.6 mm | **0** |
| 250 Hz loop + 4 ms delay | 0 (worst 6.2 mm) | 0 (worst 0.84 mm) | 0 (worst 0.84 mm) |

The application, computed torque through ROS 2: **all 6 parts sorted in 70 s** (Milestone 3 needed 83 s). While moving: 0.05° / 0.78 mm worst. During the grasps: 0.66° (the closing fingers press the part into the belt with about 15 N, which is a real force the model can't know).

## Files

| File | What it does |
|---|---|
| `dynamics.py` | **Lagrangian model:** M(q), ∂M/∂q, C(q, q̇), g(q), inverse / forward dynamics, energy |
| `control.py` | PD, PD + gravity, computed torque, computed torque + integral; torque limits |
| `trajectories.py` | Joint quintic / LSPB, task-space line / circle, `Plan` (segments in a row) |
| `sorting_task.py` | Cell positions and the sorting cycle as a plan; `SortingApp` |
| `test_paths.py` | The joint-space and task-space test trajectories |
| `ur5e_controller_node.py` | ROS 2 node: plan + control law, one torque command per joint state, logs and summary |
| `mujoco_bridge_node.py` | ROS 2 bridge: effort commands in, joint states (with torque) out |
| `sim_bridge.py` | MuJoCo simulator in torque mode: gravity on, brakes, target ball, live panel |
| `validate_control.py` | All offline tests and the report figures |
| `render_videos.py` | The two videos |
| `kinematics.py`, `velocity_kinematics.py`, `ee_sensors.py`, `environment.xml` | From Milestones 2–3 |

## Questions you may be asked

**"How do you get M(q) from the kinetic energy?"**
- A link's kinetic energy is ½ m v² for its CoM plus ½ ωᵀ I ω for its spin.
- Both v and ω are linear in q̇: `v = Jv_i q̇`, `ω = Jω_i q̇`. So each link gives `½ q̇ᵀ (m Jvᵀ Jv + Jωᵀ R I Rᵀ Jω) q̇`.
- Add all links (and the motor rotors) and you get `K = ½ q̇ᵀ M q̇`. M is what sits in the middle.

**"And g(q)?"**
- The potential energy is the weight of each link times its CoM height: `P = Σ m g z_c`.
- `g_k = ∂P/∂q_k`: how much the total weight-times-height changes when joint k turns. The change of z_c is the z row of the CoM Jacobian, so `g = Σ m g Jv_iᵀ ẑ`.

**"What do M q̈, C q̇ and g physically mean?"**
- `M q̈`: the torque to *speed up* the links (inertia). M also couples joints: accelerating the shoulder pushes on the elbow.
- `C q̇`: torques that appear only because things are already *moving*: centrifugal (q̇_i²) and Coriolis (q̇_i q̇_j).
- `g`: the torque just to *hold* the arm against its own weight, even when nothing moves.
- At our speeds gravity is by far the biggest: on the shoulder up to 37 N·m, against about 3 N·m of inertia and 1.7 N·m of Coriolis ([figure](report/figures/dynamics_terms.png)).

**"How do you know your dynamics are right?"**
- MuJoCo computes the same quantities its own way (recursive Newton–Euler on the XML). It never sees our DH frames or our Lagrangian.
- On 1000 random q, q̇, q̈ the torques agree to 10⁻¹⁰ N·m.
- Also Ṁ − 2C is skew-symmetric. That only holds if C really comes from M's Christoffel symbols.

**"What happens if you remove the model and use only PD?"**
- Gravity has no one to cancel it. The arm sags until the spring `Kp e` carries its weight: 1.9° on the shoulder, 2.4° on the elbow, 43 mm at the flange, even at rest.
- Adding only `g(q)` removes the sag at rest. It still lags 6 mm while moving.
- Full computed torque gives 0.2 mm.

**"Why is the feedback part of computed torque so small?"**
- Because the model already gives almost the exact torque the plan needs. The feedback only fixes what the model misses: ±0.2 N·m out of 40 ([figure](report/figures/ctc_torque_split.png)).

**"Why do you see steps in the torque around t = 2.3 to 3.2 s?"**
- That segment is the LSPB (trapezoid). Its acceleration jumps at the corners, so `M q̈` jumps too, and that is a jerk on the motors.
- The quintic segments have smooth acceleration, so smooth torque. That's why the sorting uses quintics.

**"Counter-argument: computed torque needs a perfect model. Real models are never perfect."**
- True. With a hidden 2 kg payload it leaves 12 mm, because the error must grow until the PD part supplies the missing torque.
- But it degrades gracefully: it is still better than PD + gravity in every case we tried.
- Adding an integral term removes the steady error (0.002 mm with the payload, 0 with 20 % heavier links).
- The integral has its own cost: it slowly winds up during contact, so we clamp it.

**"Why 250 Hz, and does the ROS 2 delay matter?"**
- The simulator sends one joint state every 4 ms. The node answers each one with a torque, and the round trip through Docker takes a few ms.
- With ωn = 20 rad/s, a few ms of delay is a small phase loss. Through ROS 2 the error goes from 0.21 to 0.33 mm.
- Much higher gains would make the delay matter (it could go unstable).

**"Why are pick and place in task space but the transfers in joint space?"**
- Down to the part: the gripper must move straight down, or it would hit the part or the belt. Only a task-space line guarantees that.
- Transfers: nothing is close. Joint space is faster to compute (no IK), always respects joint limits, and can't pass through a singularity on the way.

**"What are the brakes?"**
- A real UR5e has brakes in every joint that hold it when the controller is off.
- Without them, in torque mode the arm would fall the moment the simulator starts. The simulator engages them whenever no torque command came for 0.1 s.
