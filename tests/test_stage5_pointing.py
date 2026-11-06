import numpy as np
import pytest

from manipulator3.pointing import (
    HandTracker,
    HandLandmarks,
    PointingStatus,
    PointingTarget,
    TableBounds,
    DEFAULT_TABLE_BOUNDS,
    compute_pointing_vector,
    intersect_ray_with_table,
    transform_ray_camera_to_world,
    is_pointing_gesture,
)
from manipulator3.kinematics import REACH_R_MIN, REACH_R_MAX


def test_compute_pointing_vector():
    """Verify index finger unit direction vector computation."""
    mcp = np.array([0.1, 0.2, 0.3])
    tip = np.array([0.1, 0.2, 0.1])  # Pointing straight down (-Z)
    v = compute_pointing_vector(mcp, tip)
    np.testing.assert_allclose(v, [0.0, 0.0, -1.0], atol=1e-6)
    assert np.isclose(np.linalg.norm(v), 1.0)

    # Diagonal pointing vector
    tip_diag = np.array([0.2, 0.3, 0.2])
    v_diag = compute_pointing_vector(mcp, tip_diag)
    expected = np.array([0.1, 0.1, -0.1]) / np.linalg.norm([0.1, 0.1, -0.1])
    np.testing.assert_allclose(v_diag, expected, atol=1e-6)


def test_intersect_ray_valid_table_target():
    """Verify ray-plane intersection and reachability calculation on table surface."""
    # Hand at (0.25, 0.10, 0.30) pointing towards (0.15, 0.0, 0.0)
    origin = np.array([0.25, 0.10, 0.30])
    target_pos = np.array([0.15, 0.0, 0.0])
    v = compute_pointing_vector(origin, target_pos)

    target = intersect_ray_with_table(origin, v)
    assert target.status == PointingStatus.VALID
    assert target.is_reachable is True
    assert target.point_2d is not None
    assert target.point_3d is not None

    np.testing.assert_allclose(target.point_2d, [0.15, 0.0], atol=1e-5)
    np.testing.assert_allclose(target.point_3d, [0.15, 0.0, 0.0], atol=1e-5)
    assert np.isclose(target.distance, 0.15, atol=1e-5)
    assert np.isclose(target.azimuth, 0.0, atol=1e-5)

    # Search sector contains target azimuth
    yaw_min, yaw_max = target.search_sector
    assert yaw_min <= target.azimuth <= yaw_max
    assert np.isclose(yaw_max - yaw_min, np.deg2rad(40.0), atol=1e-4)


def test_intersect_ray_parallel_and_upward():
    """Verify handling of rays that are horizontal or pointing upwards."""
    origin = np.array([0.20, 0.0, 0.30])

    # Horizontal ray (dz = 0)
    v_horiz = np.array([1.0, 0.0, 0.0])
    target_horiz = intersect_ray_with_table(origin, v_horiz)
    assert target_horiz.status == PointingStatus.PARALLEL_OR_UPWARD
    assert target_horiz.is_reachable is False

    # Upward ray (dz > 0)
    v_up = np.array([0.0, 0.0, 1.0])
    target_up = intersect_ray_with_table(origin, v_up)
    assert target_up.status == PointingStatus.PARALLEL_OR_UPWARD
    assert target_up.is_reachable is False


def test_intersect_ray_out_of_table_boundaries():
    """Verify detection of pointing rays intersecting outside table boundaries."""
    origin = np.array([0.20, 0.0, 0.30])

    # Target far beyond table X limit (X > 0.60m)
    v_far_x = compute_pointing_vector(origin, np.array([1.50, 0.0, 0.0]))
    target_far_x = intersect_ray_with_table(origin, v_far_x)
    assert target_far_x.status == PointingStatus.OUT_OF_TABLE
    assert target_far_x.is_reachable is False
    assert target_far_x.point_2d[0] > DEFAULT_TABLE_BOUNDS.x_max

    # Target beyond table Y limit (|Y| > 0.35m)
    v_far_y = compute_pointing_vector(origin, np.array([0.20, 0.80, 0.0]))
    target_far_y = intersect_ray_with_table(origin, v_far_y)
    assert target_far_y.status == PointingStatus.OUT_OF_TABLE
    assert abs(target_far_y.point_2d[1]) > DEFAULT_TABLE_BOUNDS.y_max


def test_intersect_ray_out_of_reach():
    """Verify detection of points on table that exceed manipulator reach limits."""
    origin = np.array([0.20, 0.0, 0.30])

    # Target on table at (0.45, 0.0, 0.0): R = 0.45m > REACH_R_MAX (0.32m)
    v_far = compute_pointing_vector(origin, np.array([0.45, 0.0, 0.0]))
    target_far = intersect_ray_with_table(origin, v_far)
    assert target_far.status == PointingStatus.OUT_OF_REACH
    assert target_far.is_reachable is False
    assert target_far.distance > REACH_R_MAX

    # Target too close to base (0.04, 0.0, 0.0): R = 0.04m < REACH_R_MIN (0.08m)
    v_near = compute_pointing_vector(origin, np.array([0.04, 0.0, 0.0]))
    target_near = intersect_ray_with_table(origin, v_near)
    assert target_near.status == PointingStatus.OUT_OF_REACH
    assert target_near.is_reachable is False
    assert target_near.distance < REACH_R_MIN


def test_transform_ray_camera_to_world():
    """Verify camera to world coordinate frame transformation."""
    cam_pos = np.array([0.5, 0.0, 0.4])
    cam_rot = np.eye(3)  # Identity rotation

    orig_cam = np.array([0.0, 0.0, 0.2])
    dir_cam = np.array([0.0, 0.0, 1.0])

    orig_w, dir_w = transform_ray_camera_to_world(orig_cam, dir_cam, cam_pos, cam_rot, is_opengl=False)
    np.testing.assert_allclose(orig_w, [0.5, 0.0, 0.6])
    np.testing.assert_allclose(dir_w, [0.0, 0.0, 1.0])


def test_pointing_gesture_discrimination():
    """Verify discrimination between pointing gesture and closed fist / open hand."""
    lms = np.zeros((21, 3))
    lms[0] = [0.0, 0.0, 0.0]  # wrist
    # Extended index finger
    lms[5] = [0.02, 0.08, 0.0]
    lms[6] = [0.02, 0.12, 0.0]
    lms[7] = [0.02, 0.15, 0.0]
    lms[8] = [0.02, 0.18, 0.0]
    # Curled other fingers
    lms[9] = [0.0, 0.08, 0.0]
    lms[10] = [0.0, 0.10, 0.0]
    lms[12] = [0.0, 0.08, 0.03]  # middle tip curled back
    lms[16] = [-0.02, 0.07, 0.03]  # ring
    lms[20] = [-0.04, 0.06, 0.03]  # pinky

    hl = HandLandmarks(landmarks=lms, world_landmarks=lms)
    is_pointing, score = is_pointing_gesture(hl)
    assert is_pointing is True
    assert score > 0.8

    # Modify to curled index finger (fist)
    lms[8] = [0.02, 0.08, 0.03]
    hl_fist = HandLandmarks(landmarks=lms, world_landmarks=lms)
    is_pointing_fist, _ = is_pointing_gesture(hl_fist)
    assert is_pointing_fist is False


def test_hand_tracker_empty_frame():
    """Verify HandTracker behavior on empty/blank image."""
    try:
        tracker = HandTracker()
    except ImportError:
        pytest.skip("mediapipe is not installed (thin client environment)")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    lms = tracker.detect_landmarks(blank)
    assert lms is None

    target, detected_lms = tracker.estimate_pointing_target(blank)
    assert target.status == PointingStatus.NO_HAND
    assert target.is_reachable is False
    assert detected_lms is None
