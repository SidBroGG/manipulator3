import pytest
import numpy as np
import cv2

from manipulator3.detector import (
    DetectedObject,
    compute_detected_object,
    YOLOObjectDetector,
    ColorBlobDetector,
)
from manipulator3.servoing import (
    VisualServoController,
    SectorScanner,
    VisualSearchTracker,
    TrackingState,
)
from manipulator3.simulation import SimulationEnvironment
from manipulator3.kinematics import POSE_SCAN


def test_detected_object_geometry():
    """Verify 2D bounding box center, normalized error coordinates, and aspect ratio."""
    det = compute_detected_object(
        box=(100.0, 100.0, 300.0, 200.0),
        class_id=0,
        class_name="cup",
        confidence=0.85,
        frame_width=640,
        frame_height=480,
    )

    assert det.class_name == "cup"
    assert det.confidence == pytest.approx(0.85)
    assert det.center_xy == (200.0, 150.0)
    assert det.width == pytest.approx(200.0)
    assert det.height == pytest.approx(100.0)
    assert det.area == pytest.approx(20000.0)
    assert det.aspect_ratio == pytest.approx(2.0)

    # Frame center is (320, 240): u_norm = (200 - 320) / 320 = -0.375
    assert det.norm_center[0] == pytest.approx(-0.375)
    assert det.norm_center[1] == pytest.approx(-0.375)


def test_gripper_finger_exclusion():
    """Verify that gripper fingers visible at frame borders are identified and excluded."""
    from manipulator3.detector import is_gripper_finger, EXCLUDED_ROBOT_CLASSES

    # Left gripper jaw
    assert is_gripper_finger((0.2, 157.0, 78.6, 284.6), 640, 480) is True
    # Right gripper jaw
    assert is_gripper_finger((590.6, 153.5, 639.8, 278.2), 640, 480) is True
    # Valid target in table center
    assert is_gripper_finger((280.0, 200.0, 360.0, 280.0), 640, 480) is False

    assert "traffic light" in EXCLUDED_ROBOT_CLASSES


def test_yolo_benchmark_inference():
    """6.3.1 Benchmark YOLO11n inference latency on CPU."""
    detector = YOLOObjectDetector()
    bench = detector.benchmark_inference(iterations=5, img_size=(640, 480))

    assert "mean_ms" in bench
    assert "fps" in bench
    assert bench["mean_ms"] > 0.0
    assert bench["mean_ms"] < 1000.0  # reasonable CPU inference bound
    assert bench["fps"] > 1.0


def test_color_blob_detector_synthetic():
    """Verify color segmentation detection on synthetic test canvas."""
    h, w = 480, 640
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw a synthetic blue box in center
    cv2.rectangle(canvas, (280, 200), (360, 280), (220, 40, 20), -1)  # BGR

    detector = ColorBlobDetector(target_colors=["blue_box"])
    detections = detector.detect(canvas)

    assert len(detections) >= 1
    best = detections[0]
    assert best.class_name == "blue_box"
    assert abs(best.norm_center[0]) < 0.05
    assert abs(best.norm_center[1]) < 0.05


def test_sector_scanner_waypoints():
    """Verify sector orientation and sweep waypoint generation."""
    scanner = SectorScanner()

    # Orientation to azimuth
    azimuth = float(np.deg2rad(30.0))
    pose = scanner.get_sector_pose(azimuth)
    assert pose[0] == pytest.approx(-azimuth)

    # Sweep waypoints across sector
    waypoints = scanner.generate_sweep_waypoints(azimuth_rad=azimuth, span_rad=0.6)
    assert len(waypoints) >= 4
    for wp in waypoints:
        # Check joint angle within hardware limits
        assert -3.14159 <= wp[0] <= 3.14159
        assert -1.5708 <= wp[3] <= 1.5708


def test_visual_servo_proportional_convergence():
    """6.3.2 Test target tracking centering loop convergence on off-center target."""
    servo = VisualServoController(k_yaw=0.35, k_pitch=0.30, tolerance_norm=0.03)

    # Simulated target moving towards center as robot joint adjusts
    target_world_yaw = 0.0
    target_world_pitch = 0.0

    current_q = POSE_SCAN.copy()
    current_q[0] = 0.20  # initial base yaw error
    current_q[3] = POSE_SCAN[3] + 0.15  # initial pitch error

    for step in range(30):
        # Target error in camera frame relative to current joint
        e_u = float(np.clip((current_q[0] - target_world_yaw) / 0.35, -1.0, 1.0))
        e_v = float(np.clip((current_q[3] - (POSE_SCAN[3] + target_world_pitch)) / 0.30, -1.0, 1.0))

        cx = 320.0 + e_u * 320.0
        cy = 240.0 + e_v * 240.0
        det = compute_detected_object(
            box=(cx - 20, cy - 20, cx + 20, cy + 20),
            class_id=0,
            class_name="target",
            confidence=0.9,
            frame_width=640,
            frame_height=480,
        )

        current_q, is_aligned = servo.step(det, current_q)
        if is_aligned:
            break
    else:
        pytest.fail("Visual servo loop failed to converge within 30 steps")

    assert is_aligned is True
    assert abs(current_q[0] - target_world_yaw) < 0.02


def test_simulation_visual_servoing_loop():
    """Test closed-loop visual servo centering on tabletop object in MuJoCo simulation."""
    env = SimulationEnvironment()
    try:
        # Start in scan pose facing slightly off-center from red_cube
        q_init = POSE_SCAN.copy()
        q_init[0] = 0.06
        q_init[3] = 1.30
        env.set_actuator_targets(q_init)
        for _ in range(150):
            env.step()

        detector = ColorBlobDetector(target_colors=["red_cube"])
        servo = VisualServoController(k_yaw=0.35, k_pitch=0.30, tolerance_norm=0.05)

        current_q = q_init.copy()
        converged = False

        for step in range(25):
            frame = env.render_bgr("gripper_camera")
            detections = detector.detect(frame)
            if not detections:
                continue

            best = detections[0]
            current_q, is_aligned = servo.step(best, current_q)
            env.set_actuator_targets(current_q)
            for _ in range(25):
                env.step()

            if is_aligned:
                converged = True
                break

        assert converged is True
        # Verify ToF sensor detects object
        reading_mm = env.get_tof_distance_mm()
        assert reading_mm > 0.0  # object beam intercepted
    finally:
        env.close()
