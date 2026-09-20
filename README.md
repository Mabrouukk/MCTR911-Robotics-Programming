# MCTR911 – Robotics Programming Project (Winter 2026)

**Robot:** Universal Robots UR5e (6-DOF industrial manipulator)
**Application:** Pick-and-place / palletizing on a conveyor line
**Tools:** ROS2 (Humble), MuJoCo, Python, Docker

## Team

| Name | ID | GitHub |
|------|----|--------|
| Mahmoud Ghobashy | 58-2432 |  |
| Kevin Ezzat | 58-0408 | ... |
| Seif Hassan | 58-6034 | ... |
| Mohamed Elsayed | 58-16392 | ... |
| fawzy karim | 58-9263| ... |


Contact person (per registration form): TBD

## Repository structure

```
MCTR/
├── Milestone 01/
│   ├── literature_review/   # 1-2 page literature review + project flow
│   ├── cad_models/          # UR5e MJCF/URDF model (from mujoco_menagerie)
│   └── videos/              # demo videos of ROS2 + MuJoCo running
├── docker/                  # ROS2 + MuJoCo containerized environment
└── README.md
```

Later milestones (`Milestone 02`, `Milestone 03`, `Milestone 04`) will be added as their own top-level folders following the same pattern.

## Environment setup

See [docker/README.md](docker/README.md) for how to build and run the ROS2 + MuJoCo container.
