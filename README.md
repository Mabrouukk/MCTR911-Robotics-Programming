# MCTR911 – Robotics Programming Project

**Robot:** Universal Robots UR5e (6-DOF industrial manipulator)
**Application:** Pick-and-place / palletizing on a conveyor line
**Tools:** ROS2 (Humble), MuJoCo, Python, Docker

## Team

| Name | ID | GitHub |
|------|----|--------|
| Mahmoud Ghobashy | 58-2432 | [@Mabrouukk](https://github.com/Mabrouukk) |
| Kevin Ezzat | 58-0408 | [@kevinezzat](https://github.com/kevinezzat) |
| Seif Hassan | 58-6034 | [@seifeldin](https://github.com/seifeldin) |
| Mohamed Elsayed | 58-16392 | [@ME312241](https://github.com/ME312241) |
| fawzy karim | 58-9263 | [@FawzyKareem](https://github.com/FawzyKareem) |


Contact person : Mahmoud.Ghobashy

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
