# Milestone 3: UR5e velocity and acceleration kinematics

Team 5, colour-sorting cell with a UR5e and a Robotiq 2F-85 gripper. Due Wednesday 11 November 2026.

## What the milestone asks, and where it is

| The brief asks | Where |
|---|---|
| Forward and inverse **velocity** kinematics in Python | [code/velocity_kinematics.py](code/velocity_kinematics.py): `jacobian`, `forward_velocity`, `inverse_velocity` |
| Forward and inverse **acceleration** kinematics in Python | same file: `jacobian_dot`, `forward_acceleration`, `inverse_acceleration` |
| Derive them | [code/derive_symbolic.py](code/derive_symbolic.py) (sympy) → [report/derivation_output.txt](report/derivation_output.txt), and the report, sections 3–4 |
| Test position and velocity kinematics on the simulator: joint values in, sense the end effector, and the inverse, within a tolerance | [code/validate_kinematics.py](code/validate_kinematics.py) (offline) and [code/kinematics_validator.py](code/kinematics_validator.py) (live, through ROS 2). We also tested acceleration. |
| Continue the GUI: chairs, conveyors, people, tools… | [code/environment.xml](code/environment.xml), plus the live kinematics panel in [code/sim_bridge.py](code/sim_bridge.py) |
| Report on the velocity and acceleration analysis | [report/Milestone_03_Report.docx](report/Milestone_03_Report.docx) / [.pdf](report/Milestone_03_Report.pdf) |
| Codes for the velocity kinematics and the motion visualization | [code/](code/) |
| Video of the kinematics validation, commenting on the results | [videos/ms3_kinematics_validation.mp4](videos/ms3_kinematics_validation.mp4) |
| Videos of the GUI | [videos/ms3_gui_environment.mp4](videos/ms3_gui_environment.mp4) |
| Videos of testing the equations: joint angles in, end-effector position sensed | part 1 of the validation video (parts 2–3 do velocity and acceleration) |

The videos have on-screen comments. They have no voice; record one over them if the evaluation needs it.

## Run it

```bash
scripts/run_ms3.sh            # the colour sorting in the full environment, checks running (default)
scripts/run_ms3.sh forward    # test: every joint follows a sine wave, every instant checked live
scripts/run_ms3.sh inverse    # test: a flange path (line + circle) -> IK / J^-1 -> joint commands
scripts/run_ms3.sh validate   # all offline tests + report figures, no ROS 2 (about 6 s)
scripts/run_ms3.sh derive     # symbolic derivation with sympy (about 5 s)
```

Or in VS Code: ⌘⇧P → "Tasks: Run Task" → **Run Milestone 3**.

**What you'll see:**
- **The viewer:** the cell, with a small **live kinematics panel** at the top right. Each check (position, velocity, acceleration, inverse) gets one line: OK or FAIL, with the worst error so far. Below them are the flange speed and how far the arm is from a singularity. Run `sim_bridge.py --no-panel` to hide it; Tab / Shift+Tab open the viewer's side menus.
- **The terminal:** the validator prints the errors every second, then a PASS/FAIL table.

The videos are rendered by `.venv/bin/python "Milestone 03/code/render_videos.py"`.

## The idea, in short

- **Velocity kinematics.** The Jacobian J (6×6) maps joint rates to the flange's twist: `[v; ω] = J q̇`.
  - Column i is `[ z(i−1) × (p − o(i−1)) ; z(i−1) ]`, where z is joint i's axis, o a point on it, and p the flange.
  - The inverse is `q̇ = J⁻¹ [v; ω]`. The UR5e has 6 joints for 6 numbers, so J is square.
- **Acceleration kinematics.** Differentiate in time: `[a; α] = J q̈ + J̇ q̇`. The inverse is `q̈ = J⁻¹ ([a; α] − J̇ q̇)`.
  - J̇ comes from the product rule, with `ż = ω × z` for each joint axis.
- **Singularities.** `det J = a2 a3 sin q3 sin q5 (a2 cos q2 + a3 cos(q2+q3) + d5 sin(q2+q3+q4))`, so there are three:
  - **elbow:** q3 = 0, the arm is straight,
  - **wrist:** q5 = 0, axes 4 and 6 line up,
  - **shoulder:** the wrist centre is on the base axis.
  - Near any of them, the inverse switches to damped least squares, so the joint rates stay bounded.
- **Validation.** MuJoCo computes the same quantities its own way. Six sensors on the flange measure its position, orientation, velocity and acceleration, and we compare them with our equations.

## Results

| Test | Result |
|---|---|
| 1000 random joint states put into the simulator: FK, J q̇, J q̈ + J̇ q̇ vs the sensors, and IK, J⁻¹, J⁻¹(…) back to the joints | all 9 checks pass, worst error 3·10⁻¹³ (round-off). Tolerance 10⁻⁶ (10⁻⁵ for accelerations). |
| Same, while the physics runs (servos follow sine waves, flange up to 1.0 m/s and 5 m/s²) | all pass, worst 5·10⁻¹⁴ |
| Same, live through ROS 2 | forward run: 265 instants checked; inverse run: 353. All pass. |
| Inverse kinematics driving the robot along a task-space path (line + circle, gripper down) | 68 mm off with joint angles only (servo lag); **0.55 mm** with velocity feed-forward (0.5 mm from the path through ROS 2, in every run) |
| Symbolic J vs numeric J, symbolic J̇ vs recursive J̇, det J closed form | agree to 7·10⁻¹⁶, 2·10⁻¹⁵, 6·10⁻¹⁷ |
| Near the wrist singularity (q5 = 0.05°), for 5 cm/s | plain J⁻¹ asks 146 rad/s; damped asks 0.02 rad/s (limit π) |
| Sorting still works in the new environment | all 6 parts sorted in 83 s, as in Milestone 2 |

## Files

| File | What it does |
|---|---|
| `velocity_kinematics.py` | **The main file.** Jacobian, J̇, forward/inverse velocity and acceleration, det J, singularity handling |
| `derive_symbolic.py` | Derives J two ways, J̇, and det J with sympy, and checks them against the code |
| `ee_sensors.py` | The six MuJoCo flange sensors; removes gravity from the accelerometer |
| `kinematic_checks.py` | One instant: our equations vs the sensors, forward and inverse (used everywhere) |
| `task_path.py` | The task-space test path, turned into joint motion with IK / J⁻¹ / J⁻¹(…) |
| `validate_kinematics.py` | Offline Tests 1–4 + report figures |
| `kinematics_validator.py` | ROS 2 node: drives the arm (forward / inverse / sort) and checks live |
| `sim_bridge.py` | MuJoCo simulator: cell, sensors, gravity compensation, live panel, TCP to ROS 2 |
| `mujoco_bridge_node.py` | ROS 2 bridge: joint states, joint accelerations, flange pose / twist / accel topics |
| `environment.xml` | The cell environment (fence, people, chairs, desk, tools, conveyors, shelves…) |
| `render_videos.py`, `render_scene.py` | Videos and environment screenshots |
| `kinematics.py`, `sorting_task.py`, `joint_command_publisher.py` | From Milestone 2 (DH, FK, IK, sorting) |

## Questions you may be asked

**"How did you get the Jacobian?"**
- Each joint i is revolute. If only it turns, the flange rotates about the axis z(i−1) and the flange point moves on a circle around it.
- That gives column i: `[ z(i−1) × (p − o(i−1)) ; z(i−1) ]`. z(i−1) and o(i−1) come from the FK frames `A1…A(i−1)`.
- We checked it by differentiating the FK symbolically (∂p/∂q) and against MuJoCo's own Jacobian.

**"Derive column 1 / column 6."**
- **Column 1:** z0 = (0,0,1), o0 = 0, so J1 = (−py, px, 0, 0, 0, 1). The base turns the flange on a horizontal circle.
- **Column 6:** joint 6's axis passes through the flange, so its linear part is 0. It only spins the flange.

**"What is J̇ q̇?"**
- It's the flange acceleration you get even when every joint turns at constant speed (centripetal / Coriolis-like).
- On our test path it reaches 0.23 m/s², a third of the peak acceleration (0.69 m/s²). So you can't ignore it.

**"How did you compute J̇?"**
- With the product rule on each column: `ż × (p − o) + z × (ṗ − ȯ)`, where `ż = ω × z`.
- ω and ȯ are built up link by link from the base.
- It was checked against the symbolic Σ ∂J/∂q_k · q̇_k (2·10⁻¹⁵) and against finite differences (5·10⁻¹⁰).

**"How do you know the simulator really measured something, and didn't just reuse your equations?"**
- MuJoCo never sees our DH table. It builds the arm from the XML body positions and computes velocities and accelerations with its own algorithm.
- The sensors are MuJoCo's frame sensors on the flange site.
- The accelerometer even includes gravity, like a real one. We had to add g back.

**"Your errors are 10⁻¹⁴. Isn't that suspicious?"**
- No. Both sides are exact formulas for the same rigid-body motion, so they can only differ by floating-point round-off.
- Any real mistake (a wrong sign, a wrong axis, a missing J̇ term) shows up as an error of order 0.1–1, not 10⁻¹⁴.
- The tolerance (10⁻⁶) is the "simple tolerance" the brief asks for.

**"Then where do real errors appear?"**
- In Test 3, where the inverse kinematics drives the robot. The servos are spring-dampers (kd/kp = 0.2 s), so the arm trails its command by 0.2 s: 68 mm off the path.
- Adding `(kd/kp) · J⁻¹ v` to the command (velocity feed-forward, from our inverse velocity) brings it to 0.55 mm.

**"What happens at a singularity?"**
- det J → 0, so plain J⁻¹ asks for joint rates that grow like 1/σ_min: 146 rad/s at q5 = 0.05°, against a limit of π rad/s.
- Below σ_min = 0.02 we use damped least squares, `Jᵀ (JJᵀ + λ²I)⁻¹`. It gives up the impossible direction instead of asking for infinite speed.
- Away from singularities it is exact.
- Our sorting motions stay at σ_min ≥ 0.23.

**"Why gravity compensation?"**
- The real UR5e controller compensates gravity.
- Without it, the position servos sag under the arm's weight, and that sag would look like a kinematics error.
- With it, what's left is only the servo lag, which we explain and cancel.

**"Why does the ROS 2 test need time stamps?"**
- The joint and flange readings must come from the same instant.
- The bridge stamps every message with the simulator time, and the validator matches them by stamp.

**"How do you measure 'follows the path' through ROS 2?"**
- As the distance from the sensed flange to the nearest point of the planned path (the contour error).
- A command and the state answering it each take a few ms through Docker, and that round trip changes from run to run.
- Measured against where the plan says the flange should be *right now*, 20 ms of delay at 0.3 m/s reads as 6 mm of "error" even though the flange is on the path. Our first version did that and gave 0.7–5.8 mm depending on the run.
- Measured as distance to the path, every run gives 0.5 mm. The delay is printed separately (2–8 ms).
