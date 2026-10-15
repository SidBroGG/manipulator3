# Manipulator3: 6-DOF Robotic Arm with Eye-in-Hand Vision

Autonomous 6-DOF robotic manipulator designed to detect objects pointed at by a human finger on a flat table, approach using eye-in-hand camera (OV5647) and ToF distance sensor (VL53L0X), grasp, and relocate them.

## Tech Stack
- **Thin Client (Raspberry Pi Zero 2 W)**: Zenoh, Picamera2/libcamera, VL53L0X (I2C), PWM servo driver.
- **Compute Host (Laptop)**: Zenoh, OpenCV, MediaPipe Hands, YOLO11n, IKPy, MuJoCo.
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (Python >= 3.12).

## Documentation
- [PLAN.md](PLAN.md) — Master development plan and roadmap.
- [GEMINI.md](GEMINI.md) — Architecture, development guidelines, and verification checklists.
