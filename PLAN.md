# Master Development Plan: 6-DOF Robotic Manipulator with Eye-in-Hand Vision

## Project Summary
Development of an autonomous 6-DOF robotic arm with a parallel gripper equipped with an eye-in-hand OV5647 camera and VL53L0X ToF rangefinder. The system detects where a user's finger points on a flat table (even if outside the initial field of view), scans and locates the target object, approaches it using ToF distance measurement, grasps it, and transfers it to a drop-off zone near the manipulator base.

Computing is split between:
- **Thin Client (Raspberry Pi Zero 2 W)**: Camera streaming, ToF reading, servo PWM control, Zenoh communication.
- **Compute Host (Laptop: Ryzen 5, Radeon 740M, 16GB)**: Perception (MediaPipe, YOLO11n), Kinematics (IKPy, MuJoCo), State Machine, Zenoh broker/client.

---

## Stage 1: URDF Adaptation, Mesh Sanitization & MuJoCo Model
- [x] **1.1 URDF Verification & Cleanup**
  - [x] 1.1.1 Inspect `assets/root.urdf` joint definitions, limits, and axis orientations.
  - [x] 1.1.2 Normalize STL mesh filenames and paths (replace Cyrillic/space characters with standard ASCII naming to ensure multi-platform loader compatibility in MuJoCo and IKPy).
  - [x] 1.1.3 Verify kinematic chain continuity from `base__подставка` (`base_link`) to end-effector `main__gripperbase` and jaws `main__gripper1`, `main__gripper2`.
- [x] **1.2 MuJoCo Model Definition (MJCF)**
  - [x] 1.2.1 Convert/load URDF into MuJoCo MJCF format (`assets/manipulator.xml`).
  - [x] 1.2.2 Add visual and collision geoms for links and gripper fingers with realistic friction parameters.
  - [x] 1.2.3 Define actuators for joints: `revolute_1` (base yaw), `revolute_2` (shoulder pitch), `revolute_3` (elbow pitch), `revolute_4` (wrist pitch), `revolute_5` (wrist roll), and gripper actuator (`revolute_7` with mimic coupling to `revolute_6`).
- [x] **1.3 Sensor Rig Modeling in MuJoCo**
  - [x] 1.3.1 Mount a virtual camera on `main__gripperbase` aligned with the gripper squeeze direction, matching the OV5647 FOV (~62° horizontal).
  - [x] 1.3.2 Mount a virtual rangefinder sensor (ToF) on `main__gripperbase` parallel to the gripper axis, matching VL53L0X specs (up to 2m, cone ~25°).
- [x] **1.4 Stage 1 Unit Tests**
  - [x] 1.4.1 Test model compilation and forward kinematics validation in MuJoCo.

---

## Stage 2: Simulation Environment & Virtual Sensor Feeds
- [x] **2.1 MuJoCo Tabletop Scene Setup**
  - [x] 2.1.1 Create simulation environment with a flat table, ambient lighting, and base mounting coordinates.
  - [x] 2.1.2 Populate table with test objects (primitive shapes, textured blocks, colored cylinders/cubes).
  - [x] 2.1.3 Add a top/side overview camera for third-person visualization and debugging.
- [x] **2.2 Virtual Camera & ToF Data Acquisition**
  - [x] 2.2.1 Implement offscreen renderer pipeline to retrieve RGB frames from the gripper-mounted OV5647 camera.
  - [x] 2.2.2 Implement raycast reading from the virtual VL53L0X rangefinder sensor.
  - [x] 2.2.3 Build interactive OpenCV debug preview displaying gripper camera feed and current ToF distance overlay.
- [x] **2.3 Stage 2 Unit Tests**
  - [x] 2.3.1 Test camera render output format (RGB/BGR, resolution, FPS).
  - [x] 2.3.2 Test ToF range accuracy against known object distances on table.


---

## Stage 3: Zenoh Communication Layer (Digital Twin & Hardware Agnostic)
- [x] **3.1 Unified Communication Specification**
  - [x] 3.1.1 Define Zenoh topic naming convention:
    - `manipulator/camera/raw`: compressed JPEG image bytes.
    - `manipulator/tof/range`: float range in mm + status.
    - `manipulator/joints/command`: target angles (rad) for joints 1-6 + gripper.
    - `manipulator/joints/state`: actual joint positions, velocities.
    - `manipulator/status`: heartbeat and arm operational state.
- [x] **3.2 Laptop Zenoh Node**
  - [x] 3.2.1 Implement lightweight Zenoh session manager with zero-copy or compact serialization.
  - [x] 3.2.2 Implement subscriber handlers for camera feed and ToF data with ring buffers.
  - [x] 3.2.3 Implement publisher for joint position commands.
- [x] **3.3 Simulation Zenoh Bridge**
  - [x] 3.3.1 Connect MuJoCo simulation to Zenoh so that the simulation can be driven as a drop-in replacement for the real RPi Zero 2 W hardware.
- [x] **3.4 Stage 3 Unit Tests**
  - [x] 3.4.1 Test roundtrip latency and throughput of image frames and sensor packets over Zenoh.

---

## Stage 4: Kinematics Engine & Trajectory Planner
- [x] **4.1 Kinematics Setup (IKPy & MuJoCo IK)**
  - [x] 4.1.1 Initialize IKPy kinematic chain from URDF with exact link lengths, joint limits, and active joint masks.
  - [x] 4.1.2 Implement inverse kinematics (IK) solver calculating joint angles for desired 3D end-effector position `(x, y, z)` and pitch/yaw orientation.
  - [x] 4.1.3 Implement fallback/refinement using MuJoCo Jacobian-based IK for singularity avoidance.
- [x] **4.2 Trajectory Generation & Safety Constraints**
  - [x] 4.2.1 Implement minimum-jerk / cubic polynomial joint trajectory interpolation between waypoints.
  - [x] 4.2.2 Enforce software safety limits (table collision boundary, maximum servo velocities).
  - [x] 4.2.3 Define key baseline poses: `HOME`, `SCAN`, `DROP_OFF`.
- [x] **4.3 Stage 4 Unit Tests**
  - [x] 4.3.1 Test IK precision on reachable grid points on the table surface.
  - [x] 4.3.2 Test collision avoidance boundary enforcement.

---

## Stage 5: Perception - Hand Pointing Ray & Target Estimation
- [ ] **5.1 MediaPipe Hand Tracking**
  - [ ] 5.1.1 Initialize MediaPipe Hands pipeline for detecting hand landmarks from external camera or simulated user view.
  - [ ] 5.1.2 Extract 3D landmarks for index finger MCP (knuckle), PIP, and TIP.
- [ ] **5.2 Pointing Ray Geometry & Table Intersection**
  - [ ] 5.2.1 Compute pointing direction vector: $\vec{v} = \text{TIP} - \text{MCP}$.
  - [ ] 5.2.2 Project 3D ray onto the 2D table plane $(Z = 0)$ to determine intersection coordinate $(X_{target}, Y_{target})$.
  - [ ] 5.2.3 Define search sector / confidence radius around the intersection point to guide manipulator orientation.
- [ ] **5.3 Stage 5 Unit Tests**
  - [ ] 5.3.1 Test ray-plane intersection math with simulated finger vectors.
  - [ ] 5.3.2 Verify handling of parallel rays (pointing horizon) and out-of-bounds projections.

---

## Stage 6: Perception - Object Detection (YOLO11n) & Search Scanning
- [ ] **6.1 YOLO11n Integration**
  - [ ] 6.1.1 Load and initialize `yolo11n.pt` for target object detection on camera frames.
  - [ ] 6.1.2 Filter bounding boxes and class IDs relevant to pickable table objects.
  - [ ] 6.1.3 Extract 2D bounding box center and bounding box area/aspect ratio.
- [ ] **6.2 Out-of-FOV Search Strategy**
  - [ ] 6.2.1 When target object is not immediately visible, orient arm towards pointing sector from Stage 5.
  - [ ] 6.2.2 Execute scanning sweep trajectory (yaw sweep + tilt angle adjustment) until YOLO11n detects candidate object.
  - [ ] 6.2.3 Center target in camera frame using proportional visual servoing (adjusting base yaw and wrist pitch).
- [ ] **6.3 Stage 6 Unit Tests**
  - [ ] 6.3.1 Benchmark YOLO11n inference time on CPU/iGPU.
  - [ ] 6.3.2 Test target tracking centering loop with synthetic simulated frames.

---

## Stage 7: Grasping Strategy & ToF-Assisted Pick-and-Place State Machine
- [ ] **7.1 Fine Approach Using ToF Rangefinder**
  - [ ] 7.1.1 Once object is centered visually, drive end-effector downward along the line of sight.
  - [ ] 7.1.2 Continuously poll VL53L0X distance reading until target distance threshold is reached (e.g., 25–40 mm inside gripper jaws).
  - [ ] 7.1.3 Stop approach and verify gripper clearance before closing.
- [ ] **7.2 Grasping & Transport Sequence**
  - [ ] 7.2.1 Close gripper (`revolute_7` -> closed threshold) and verify contact/grip.
  - [ ] 7.2.2 Lift end-effector vertically by 80 mm (pre-transfer lift).
  - [ ] 7.2.3 Trajectory move to drop-off location on the table adjacent to the manipulator base.
  - [ ] 7.2.4 Lower end-effector, open gripper to release object, retract to `HOME`.
- [ ] **7.3 Core State Machine Execution**
  - [ ] 7.3.1 Build finite state machine: `IDLE` -> `POINTING_DETECTED` -> `SECTOR_SCAN` -> `VISUAL_ALIGN` -> `TOF_APPROACH` -> `GRASP` -> `TRANSPORT` -> `RELEASE` -> `HOME`.
- [ ] **7.4 Stage 7 Unit Tests**
  - [ ] 7.4.1 Test full autonomous pick-and-place loop in MuJoCo simulation from pointing input to release.

---

## Stage 8: Thin Client for Raspberry Pi Zero 2 W
- [ ] **8.1 RPi Zero 2 W Runtime Environment**
  - [ ] 8.1.1 Lightweight Python client with minimal dependency footprint (zenoh, picamera2/libcamera, smbus2/vl53l0x, rpi-hardware-pwm or PCA9685 I2C driver).
- [ ] **8.2 Camera Streamer**
  - [ ] 8.2.1 Configure OV5647 via `picamera2` at 640x480 @ 15-30 FPS, JPEG compression, published to `manipulator/camera/raw`.
- [ ] **8.3 ToF Reader Node**
  - [ ] 8.3.1 Read VL53L0X over I2C at 20-50 Hz, published to `manipulator/tof/range`.
- [ ] **8.4 Servo Driver Controller**
  - [ ] 8.4.1 Subscribe to `manipulator/joints/command`.
  - [ ] 8.4.2 Map radians to calibrated PWM microsecond pulses (500–2500 µs) for each joint servo and gripper.
  - [ ] 8.4.3 Implement emergency cutoff if Zenoh heartbeat is lost.
- [ ] **8.5 Stage 8 Unit Tests**
  - [ ] 8.5.1 Test sensor mock and PWM pulse conversion mapping.

---

## Stage 9: Hardware Calibration & Physical Integration
- [ ] **9.1 Hand-Eye & Sensor Calibration**
  - [ ] 9.1.1 Calibrate camera intrinsic parameters and eye-in-hand offset matrix relative to `main__gripperbase`.
  - [ ] 9.1.2 Calibrate VL53L0X optical axis offset relative to gripper center and camera frame.
- [ ] **9.2 Servo Zero-Offset & Range Calibration**
  - [ ] 9.2.1 Calibrate physical zero angles and directional signs for all 6 servos to match URDF coordinates.
- [ ] **9.3 End-to-End Real World Validation**
  - [ ] 9.3.1 Point finger at object on table -> manipulator scans -> centers -> approaches via ToF -> grasps -> transports to base area.
  - [ ] 9.3.2 Latency, reliability, and edge-case tuning.
