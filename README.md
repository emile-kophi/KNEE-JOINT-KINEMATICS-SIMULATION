# Development of a virtual environment for reproduction of experimentally controlled knee joint kinematic

A computational framework for reconstructing and reproducing **subject-specific knee joint kinematics** from optical motion capture data inside a Unity-based 3D simulation environment.

The project bridges the gap between experimental motion capture acquisition and anatomical visualization, enabling realistic simulation and analysis of the relative motion of the **femur, tibia, and patella**.

---

# Overview

Accurate biomechanical analysis of joint motion often requires information that cannot be fully extracted from motion capture data alone. Visual inspection of marker trajectories or joint angle signals provides only partial insight into the spatial relationships between anatomical structures.

This project introduces a pipeline that reconstructs and reproduces three-dimensional knee joint motion recorded with an optical motion capture system (OptiTrack) using subject-specific bone models obtained from CT imaging.

The reconstructed motion is reproduced in a Unity-based virtual environment, allowing the anatomical bone meshes to move according to the experimentally recorded kinematics while preserving the rigid-body relationships between bone segments.

The resulting system provides a powerful tool for:

- visualizing joint motion in 3D
- analyzing relative bone kinematics
- validating motion capture pipelines

---

# Key Features

- Reconstruction of **femur, tibia, and patella motion** from optical tracking data
- Integration of CT-derived subject-specific anatomical meshes
- Rigid-body registration using the **Kabsch algorithm**
- Reconstruction of joint kinematics over time
- Simulation of motion capture trials inside Unity
- Quantitative validation using landmark reconstruction error
- Analysis of kinematic consistency and systematic geometric bias

---

# System Architecture


The framework combines experimental acquisition, computational reconstruction, and real-time visualization.
```
Motion Capture (OptiTrack)
│
│ Rigid body tracking
▼
Marker set trajectories
│
│ Rigid-body transformation estimation
▼
Pose reconstruction (MATLAB / Python)
│
│ Mapping onto CT-derived meshes
▼
Anatomical bone models
│
│ Export of time-resolved transforms
▼
Unity simulation environment
│
▼
3D visualization and biomechanical analysis
```
