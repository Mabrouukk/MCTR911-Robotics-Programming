# Robotics Programming Challenge 3: Panda Lagrangian dynamics and fuzzy logic control

**Team 5, Gravity-Compensated FLC.** Same scenario as Challenge 2 (the folded approach): the Panda starts stretched out in front of itself and folds back in 4 s.

What we added to the Challenge 2 package:

- `compute_dynamics(q, dq, ddq)`: M(q) (7×7) and C(q,q̇)q̇ + g(q) (7×1) by the Lagrangian method
- `fuzzy_controller(error, delta_error, dt)`: 7 decoupled FLCs (5 sets, 25 rules, centroid)
- `compute_torques(...)`: model + FLC (main design), or g(q) + FLC (our variant), or model only / FLC only for comparison
- `plan_trajectory(...)`: the Challenge 2 quintic

The robot runs in **TORQUE mode** with gravity acting on it.

```
Challenge 03/
├── panda_kinematics/                           # the package, plus:
│   ├── panda_kinematics/controller_node.py     # dynamics, FLC, compute_torques (above the marked line)
│   └── setup.py                                # entry point: controller_node = panda_kinematics.controller_node:main
├── sim/
│   ├── panda_sim.py          # MuJoCo on the Mac in TORQUE mode (stand-in for mujoco_ros2 with control_mode TORQUE)
│   ├── panda_bridge_node.py  # ROS 2 bridge in Docker: /joint_commands (torques) -> sim -> /joint_state
│   └── ros_free_import.py    # lets the sim and the checks import controller_node without ROS 2
├── verify/
│   ├── verify_dynamics.py    # our M q̈ + C q̇ + g vs MuJoCo, random states
│   └── flc_study.py          # the FLC plots, the +20 % mass test, the rule walk-through
└── results/                  # logs of the live runs, figures, summaries
```

**About the template.** `controller_node.py` is the course template. Our code is above `DO NOT MODIFY BELOW`. Below the line, the file is the template word for word, with two changes:

- `Q_START` / `Q_END`: our Team 5 configurations (the template leaves them as zeros, marked "Team-specific").
- One line fixed, with a comment: `self.destroy_timer(self.timers[0])` → `self.destroy_timer(list(self.timers)[0])`. In ROS 2 Humble, `Node.timers` is a generator, so `self.timers[0]` raises `TypeError: 'generator' object is not subscriptable` and the node dies when the trajectory starts. We tested this in our Humble container.

What the template decides for us:

- The control loop runs at **100 Hz** (`TRAJ_DT = 0.01`). That limits how stiff the FLC can be (see the scaling below).
- `compute_dynamics(q, dq, ddq)` gets `ddq` too. M and C q̇ + g do not depend on it, so we accept it and do not use it.
- **After the plan ends, the template stops calling `compute_torques`.** It sends only `C q̇ + g` at q̇ = 0, which is g(q) from our model. No FLC, no feedback. If g(q) is exact, the arm stays. If it is wrong (the +20 % mass test), the arm falls.

The brief lists `plan_trajectory(q, dq, q_des, dq_des, ddq_des)`; the template has `plan_trajectory(q_start, q_end, T, dt)`, and we follow the template.

## Run it

```bash
scripts/run_ch3.sh                  # model + FLC (default)
scripts/run_ch3.sh gravity_flc      # Team 5: g(q) + FLC
scripts/run_ch3.sh model_only       # the FLC removed: tau = M q̈_des + C q̇ + g
scripts/run_ch3.sh flc_only         # no model at all
MASS20=1 scripts/run_ch3.sh gravity_flc   # every simulated link 20 % heavier than the model (debate question)
scripts/run_ch3.sh verify           # dynamics check + FLC study and plots (no ROS 2, about 1 min)
```

The arm waits on its brakes at `Q_START`. The first torque command releases them. The simulator prints the hand error every second, then a summary 2 s after the plan ends, and saves the log to `results/sim_ros_<mode>.npz`. The mode is set on the copy that colcon builds inside Docker. The file in the repo always stays at `CONTROL_MODE = "MODEL_FLC"`.

**On the course `mujoco_ros2` bridge:** set `{"control_mode": "TORQUE"}` in the launch file. The C++ bridge must then write the 7 received values into `data->ctrl` of motor actuators (gain 1, no bias) instead of position targets, and gravity compensation must be off. Then `colcon build` and `ros2 run panda_kinematics controller_node`. `panda_sim.py` does exactly this for us (`build_model`: gain 1, bias 0, ctrlrange = forcerange 87 / 12 N·m).

## a) Dynamics by the Lagrangian method

Same method as our Milestone 4 (UR5e). Every link i has a mass m_i, a CoM c_i and an inertia I_i from the `<inertial>` tags of panda.xml. We move them into our DH frames (each MuJoCo link frame is a constant rotation and shift away from our DH frame of the same link). The hand and both fingers (open) are bolted onto link 7 as one body.

- **Kinetic energy → M.** The CoM moves with `v = Jv_i q̇` and the link spins with `ω = Jω_i q̇` (Jacobians of that link, only joints before it). So `K = ½ q̇ᵀ M q̇` with
  `M(q) = Σ (m_i Jv_iᵀ Jv_i + Jω_iᵀ R_i I_i R_iᵀ Jω_i) + 0.1·I`. The 0.1 kg·m² is each motor's rotor inertia ("armature").
- **Potential energy → g.** `P = Σ m_i g0 z_ci`, so `g_k = ∂P/∂q_k = Σ m_i g0 [Jv_i]_(z,k)`.
- **Euler–Lagrange → C.** `C_kj = Σ_i ½ (∂M_kj/∂q_i + ∂M_ki/∂q_j − ∂M_ij/∂q_k) q̇_i`. Here ∂M/∂q is in closed form: turning joint k rigidly rotates everything after it about z_k.
- **Not in M, C, g:** the 1 N·m·s/rad joint damping of panda.xml. It is friction, not energy, so our model leaves it out and the FLC has to cover it.

**Check against MuJoCo** (`verify_dynamics.py`, 1000 random q, q̇, q̈ inside the limits):

| Ours | MuJoCo | Worst difference |
|---|---|---|
| M(q) | mj_fullM | 3·10⁻⁸ kg·m² |
| C q̇ + g | qfrc_bias | 7·10⁻⁸ N·m |
| g(q) | qfrc_bias at q̇ = 0 | 3·10⁻⁹ N·m |
| M q̈ + C q̇ + g | mj_inverse − damping·q̇ | 4·10⁻⁷ N·m (torques up to 99 N·m) |
| ∂M/∂q closed form | finite differences | 1·10⁻⁹ |
| Ṁ − 2C skew-symmetric | (property) | 2·10⁻¹⁵ |

Why 10⁻⁸ and not 10⁻¹⁵: MuJoCo turns every `fullinertia` into principal axes with an iterative eigen-solver. The inertia it keeps differs from the XML numbers by up to 5·10⁻⁷ kg·m² (link 1's Iyz becomes 0.0191694570 instead of 0.019169). We use the XML numbers exactly. (We first guessed the 7-digit hand quaternion was the cause; we tested it and it was not.)

## b) The fuzzy logic controller

**Structure.** Decoupled MIMO: 7 FLCs, one per joint. They all share the same sets and rules; each has its own scaling.

**Inputs and output.** The tracking error and its rate of change, normalised and clipped to [−1, 1]:
`E = e / GE`, `DE = (Δe / dt) / GDE`. The output u ∈ [−1, 1] is multiplied by GU.

**Membership functions** ([results/flc_membership.png](results/flc_membership.png)): LN, MN, S, MP, LP.
- *Shape:* triangles centred at −1, −0.5, 0, 0.5, 1, each reaching zero at its neighbours' centres. Any value belongs to at most two sets, and the two memberships add up to 1.
- *Why triangles:* they are cheap and have no flat top. A flat top would be a dead zone where the controller ignores small changes in the error.
- *Why evenly spaced:* together with the rule table, the controller behaves like a PD near zero, so we can choose its gains like a PD's (below).
- *On the inputs,* LN and LP are shoulders: an error beyond the range still counts as fully "large".
- *On the output,* the universe runs to ±1.5, so LN and LP are whole triangles and the centroid can reach ±1.

**Rule base** (25 rules, a PD-like table):

| E \ DE | LN | MN | S | MP | LP |
|---|---|---|---|---|---|
| **LN** | LN | LN | LN | MN | S |
| **MN** | LN | LN | MN | S | MP |
| **S** | LN | MN | S | MP | LP |
| **MP** | MN | S | MP | LP | LP |
| **LP** | S | MP | LP | LP | LP |

- The output grows with the error and with how fast the error grows.
- Along the anti-diagonal the two cancel. For example, "error large positive but already falling fast" gives S, so the controller stops pushing before it overshoots.

**Inference and defuzzification.**
- Each rule fires with strength `μ_E · μ_DE`.
- Each output set is scaled by the strengths of its rules, and the scaled sets are added up.
- The output is the **centroid** of that shape over the universe.

**Scaling and output.** Near zero this FLC is exactly `u = E + DE`, so each one acts like a PD with `Kp = GU/GE` and `Kd = GU/GDE`. We choose those like a critically damped joint, `Kp = I ωn²`, `Kd = 2 I ωn`:

| Mode | FLC output | I used | ωn | GU |
|---|---|---|---|---|
| model + FLC | corrective **acceleration**; torque = M(q)(q̈_des + u) + C q̇ + g | 1 (M(q) does the coupling) | 20 rad/s | 30 rad/s² |
| g(q) + FLC (Team 5) | correction **torque** | smallest effective inertia 1/(M⁻¹)_ii on the path | 15 rad/s | 40, 40, 20, 30, 6, 6, 6 N·m |
| FLC only | torque | same | 15 rad/s | 87, 87, 87, 87, 12, 12, 12 N·m (the limits) |

**Why these ωn.** The loop runs at 100 Hz, and through ROS 2 the torque reaches the arm about one period (10 ms) late. A delay eats phase, so ωn must stay well below 1/delay.
- Our first tuning was for a 250 Hz loop (ωn = 40 and 30). At 100 Hz it oscillated: model + FLC reached 10.7° error offline.
- We swept ωn offline with 1 and 2 periods of delay and kept the largest value that stays calm with 2 periods.
- Model + FLC: 20 is calm, 25 oscillates with 2 periods. g(q) + FLC: 15 is calm, 17 already starts to chatter.

**Walk through one rule** ([results/rule_walkthrough.txt](results/rule_walkthrough.txt)). Joint 2, Team 5 controller (GE = 0.209 rad, GDE = 1.57 rad/s, GU = 40 N·m). Error 7.2°, already falling at 0.39 rad/s:

1. *Normalise:* E = +0.60, DE = −0.25.
2. *Fuzzify:* E is MP 0.8 and LP 0.2; DE is MN 0.5 and S 0.5.
3. *Four rules fire:*
   - "IF error is MP AND rate is MN THEN S" (0.40)
   - "IF error is MP AND rate is S THEN MP" (0.40)
   - "IF error is LP AND rate is MN THEN MP" (0.10)
   - "IF error is LP AND rate is S THEN LP" (0.10)
4. *Weight the output sets:* S 0.4, MP 0.5, LP 0.1.
5. *Centroid:* u = (0.4·0 + 0.5·0.5 + 0.1·1)/1.0 = **+0.35**.
6. *Torque:* 0.35 × 40 = **+14 N·m**, added to g2(q).

## Results (live, through ROS 2 at 100 Hz, the template's loop)

| Controller | Worst joint error during the motion | Worst hand error | Hand error 2 s after (template holds with g(q) only) |
|---|---|---|---|
| **model + FLC** | 0.41° | 1.96 mm | 0.19 mm |
| **g(q) + FLC** (Team 5) | 1.22° | 15.4 mm | 7.9 mm |
| model only (FLC removed) | 150° | 555 mm | 536 mm |
| FLC only (no model) | 24.0° | 435 mm | 62 mm |

- One live run each. Offline (same model, 1 period of delay): model + FLC 0.57° / 1.3 mm, g(q) + FLC 1.8° / 16 mm. So the live runs match the offline study.
- The live runs vary a little from run to run, because the ROS 2 timer does not tick exactly every 10 ms.
- If two controllers ever publish at once (an old `controller_node` still alive in another container: ROS 2 nodes in different containers see each other), the simulator prints a warning (more than 120 commands/s).
- After the plan, the error does not go back to zero by itself: the template sends g(q) only. Whatever small speed or error the arm has when the plan ends, nothing corrects it. That is the slow drift after t = 4 s in [results/tracking_desired_vs_actual.png](results/tracking_desired_vs_actual.png).

**Does the FLC reduce the tracking error?** Yes, completely. Without it (model only) the arm is open loop. The model even gives exactly the right torque at the start. But the joint friction (not in the model) and every small difference add up, and nothing pulls the arm back. Within a second it runs away, ending 147° off. With the FLC, the same model plus small corrections keeps it within 0.41°.

## Team 5 debate: the gravity-compensated FLC

**"The FLC no longer fights gravity, so it sees smaller errors. How did you scale the sets?"**
- Without the model, the FLC must hold the arm up by itself. It needs the full torque range (87 N·m) and still sags 62 mm at rest.
- With g(q) handled by the model, the FLC only has to move the arm. So its output range is 40 N·m instead of 87, and its sets are narrower by the same factor: "large error" starts at 12.0° instead of 26.1° on joint 2 ([results/error_scaling.png](results/error_scaling.png)).
- We set the width from the PD it stands for: `GE = GU / (I ωn²)`. Near zero error the FLC acts like a PD with `Kp = GU/GE`, so the width of the sets *is* the stiffness.
- Why not narrower? Narrower sets = stiffer FLC. At 100 Hz with the ROS delay, ωn = 17 already chatters. So the loop rate, not the size of the errors, sets the limit. The errors (0.01–0.7° on joint 2) stay in the middle of the universe, where the FLC is smooth and PD-like.
- I is the joint's **effective** inertia 1/(M⁻¹)_ii, not M_ii. The other joints are free to move, so a joint feels lighter than when they are held. On our path, joint 1 feels 0.23–1.1 kg·m², while M_11 is 0.6–3.1.
- Our first design used M_ii. Coupled to joint 3, joint 1 was much lighter than that, and with the delay it oscillated between the torque limits ([results/chatter.png](results/chatter.png)).

**"What if a link's mass in the XML is 20 % wrong?"** We made every simulated link 20 % heavier than the model (`MASS20=1`), [results/mass_plus20.png](results/mass_plus20.png):

| | Nominal: during / 2 s after | Links +20 %: during / 2 s after |
|---|---|---|
| g(q) + FLC | 15.4 mm / 7.9 mm | 94 mm / **146 mm (the arm falls)** |
| model + FLC | 1.96 mm / 0.19 mm | 8.6 mm / **146 mm (the arm falls)** |

- **During the motion** the FLC fights the missing weight. g(q) is 20 % too small, so the FLC must give the rest. An FLC with no integral only gives torque when there is an error, so the arm sags until the FLC output equals the missing torque. Model + FLC sags less, because its FLC is stiffer (ωn 20 vs 15).
- **After the motion** the template sends only g(q) from our model, with no FLC. That torque is 20 % too small and nothing corrects it, so the arm falls under its own weight: 2 s later joints 2, 3 and 6 are 44°, 50–53° and 31° off. This is the template's hold, not our controller.
- When our own node kept the FLC running after the plan (our first version), the arm sagged only 1–9 mm at rest instead of falling.
- *Fixes:* keep the FLC on after the plan (target Q_END); add an integral-like input (∫e) to the FLC with a few more rules; or estimate the mass.

## Questions you may be asked

**"Why did you change the inference from min-max to sum-product?"**
- Min-max (classic Mamdani) is 1.8× stiffer near zero error than the PD it stands for: u(0.02, 0.02) = 0.073 instead of 0.04. So the gain is highest exactly where the arm should be calm.
- With the loop delay, that helped the chatter in our first design ([results/flc_surface.png](results/flc_surface.png)).
- Sum-product, with these sets, gives a smooth surface with slope exactly 1 near zero. Same sets, same rules, same centroid; only how the rules are combined changed.
- Average torque change per control step: up to 5.9 N·m (first design) → 0.1 N·m (final), offline at 100 Hz.

**"Isn't multiplying the FLC by M(q) cheating? The brief says add a correction torque."**
- It *is* a correction torque: `M(q) u`. M turns the FLC's wish ("this joint needs 2 rad/s² more") into the torque that delivers it on a coupled arm.
- Without M, one FLC's torque on joint 1 also accelerates joint 3, and each FLC has to be detuned for the worst case.
- Our Team 5 variant shows what happens without M: it works, but needs the conservative scaling, and tracks more loosely (15.4 mm against 1.96 mm with M, live).

**"Why do the model-only runs fail so badly if your model is correct to 10⁻⁷?"**
- The model is exact about inertia, Coriolis and gravity, but not about friction (1 N·m·s/rad).
- More importantly, it has **no feedback**. A tiny error at the start makes g(q) slightly wrong, the arm drifts, and nothing pulls it back.
- An open-loop robot is like driving with your eyes closed using a perfect map.

**"What does the Team 5 FLC see at the end of the motion?"**
- Nothing: the template stops calling `compute_torques` when the plan ends. It sends g(q) only.
- With an exact model that holds the arm, but it does not remove the small error and speed the arm still has at t = 4 s. So the hand drifts 7.9 mm in the next 2 s.
- With a wrong model (+20 % mass) it does not hold the arm at all (see above).

**"Why did you change a line below 'DO NOT MODIFY'?"**
- `self.timers[0]` crashes in ROS 2 Humble: `Node.timers` is a generator, not a list. The node died the moment the trajectory started.
- We changed only that line, to `list(self.timers)[0]`, and left a comment. Everything else below the line is the template as given.
