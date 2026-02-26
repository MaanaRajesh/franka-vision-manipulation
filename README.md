# franka-vision-manipulation
Perception-to-control manipulation pipeline for the Franka Panda integrating kinematics, Jacobian-based control, collision detection, and trajectory planning in simulation.

![franka](https://github.com/user-attachments/assets/5aa44b76-6e83-4da0-8cb6-f5c844c64097)

## Overview

The system integrates:

- Forward and inverse kinematics
- Jacobian-based velocity control
- Collision detection
- Trajectory generation
- Vision-based object localization
- Sim-to-real calibration strategies

The objective was to execute structured pick-and-place behaviors under static and dynamic conditions.

## Key Components

### Kinematics & Control
- Analytical forward kinematics
- Jacobian computation
- Velocity and null-space inverse kinematics
- Manipulability analysis

### Motion Planning
- RRT-based path planning
- Potential field refinement
- Collision checking

### Perception Integration
- Vision-based pose estimation
- Calibration and frame alignment
- Closed-loop execution

## Results

- Stable pick-and-place performance
- Robust handling of dynamic object motion
- Sim-to-real offset correction for accurate grasp execution

## Tools

- ROS Noetic
- Gazebo
- Panda Simulator
- Python
- NumPy / SciPy

---

*Note: This repository contains only project-specific implementations and excludes course scaffolding code.*
