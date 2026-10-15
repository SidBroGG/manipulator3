import pytest
import numpy as np
from manipulator3.kinematics import (
    KinematicsEngine,
    TableCollisionError,
    WorkspaceLimitError,
    BASELINE_POSES,
    POSE_HOME,
    POSE_SCAN,
    POSE_DROP_OFF,
    TABLE_Z_MIN,
    world_to_base_pos,
    base_to_world_pos,
)
from manipulator3.trajectory import (
    TrajectoryPlanner,
    Trajectory,
)


@pytest.fixture(scope="module")
def engine():
    return KinematicsEngine()


@pytest.fixture(scope="module")
def planner(engine):
    return TrajectoryPlanner(kinematics_engine=engine)


def test_forward_kinematics_urdf_mujoco_equivalence(engine):
    """Verify forward kinematics exact mathematical equivalence between URDF/IKPy and MuJoCo."""
    test_angles = np.array([-0.35, 0.25, -0.65, 0.45, -0.15], dtype=np.float64)
    pos_mj, mat_mj = engine.forward_kinematics(test_angles)

    q_ikpy = [0.0] + list(test_angles)
    T_base = engine.ikpy_chain.forward_kinematics(q_ikpy)
    pos_ikpy_base = T_base[:3, 3]
    pos_ikpy_world = base_to_world_pos(pos_ikpy_base)

    assert np.allclose(pos_mj, pos_ikpy_world, atol=1e-5), f"FK mismatch: MuJoCo={pos_mj}, IKPy={pos_ikpy_world}"


def test_inverse_kinematics_table_grid(engine):
    """Verify target points on table grid are successfully resolved into joint angles within limits."""
    test_points = [
        np.array([0.18, 0.0, 0.12]),
        np.array([0.22, 0.08, 0.08]),
        np.array([0.16, -0.10, 0.15]),
        np.array([0.24, -0.05, 0.05]),
        np.array([0.20, 0.12, 0.18]),
    ]

    for pt in test_points:
        q_sol = engine.inverse_kinematics(pt, tolerance=1e-3, enforce_safety=True)

        # Check joint limits
        for i in range(5):
            low, high = engine.joint_limits[i]
            assert low - 1e-4 <= q_sol[i] <= high + 1e-4, f"Joint {engine.arm_joints[i]} angle {q_sol[i]} out of bounds"

        # Check reached precision
        pos_reached, _ = engine.forward_kinematics(q_sol)
        err = np.linalg.norm(pt - pos_reached)
        assert err < 1.0e-3, f"IK error {err*1000:.3f} mm exceeded 1 mm threshold at {pt}"


def test_inverse_kinematics_with_orientation(engine):
    """Verify IK resolves both position and downward approach vector."""
    target_pos = np.array([0.20, 0.0, 0.10])
    target_z_axis = np.array([0.3, 0.0, -0.95])
    target_z_axis /= np.linalg.norm(target_z_axis)

    q_sol = engine.inverse_kinematics(target_pos, target_z_axis=target_z_axis, tolerance=1e-3)
    pos_reached, mat_reached = engine.forward_kinematics(q_sol)

    assert np.linalg.norm(target_pos - pos_reached) < 1.0e-3
    cur_z = mat_reached[:, 2]
    # Cosine similarity between target_z and reached_z
    alignment = np.dot(cur_z, target_z_axis)
    assert alignment > 0.95, f"Gripper approach vector misalignment: cos={alignment:.4f}"


def test_baseline_poses(engine):
    """Verify key baseline poses (HOME, SCAN, DROP_OFF) respect joint limits and have safe clearance."""
    for name, pose in [("HOME", POSE_HOME), ("SCAN", POSE_SCAN), ("DROP_OFF", POSE_DROP_OFF)]:
        for i in range(5):
            low, high = engine.joint_limits[i]
            assert low - 1e-4 <= pose[i] <= high + 1e-4, f"{name} pose joint {i} violates limits"

        pos, _ = engine.forward_kinematics(pose)
        assert pos[2] >= TABLE_Z_MIN, f"{name} pose end-effector violates table safety boundary (Z={pos[2]:.4f}m)"

    # Specifically check DROP_OFF is above drop-off bin (X=-0.02, Y=0.18)
    pos_drop, _ = engine.forward_kinematics(POSE_DROP_OFF)
    assert np.isclose(pos_drop[0], -0.02, atol=0.03)
    assert np.isclose(pos_drop[1], 0.18, atol=0.03)


def test_minimum_jerk_trajectory(planner):
    """Verify smooth trajectory interpolation between HOME, SCAN, and DROP_OFF poses without jerk/velocity jumps."""
    traj = planner.generate_trajectory(POSE_HOME, POSE_SCAN, duration=1.0)

    assert np.allclose(traj.positions[0], POSE_HOME)
    assert np.allclose(traj.positions[-1], POSE_SCAN)

    # Start and end velocities must be zero (minimum-jerk boundary conditions)
    assert np.allclose(traj.velocities[0], np.zeros(5), atol=1e-4)
    assert np.allclose(traj.velocities[-1], np.zeros(5), atol=1e-4)

    # Peak velocities must respect planner maximum velocity
    peak_v = float(np.max(np.abs(traj.velocities)))
    assert peak_v <= planner.max_velocity + 0.01

    # End-effector along path must stay above table safety margin
    planner.validate_trajectory(traj, check_table_safety=True)


def test_software_safety_limit_interceptions(engine, planner):
    """Verify software safety limits prevent commanded motion below the table or outside workspace."""
    # Attempt target below table surface
    with pytest.raises(TableCollisionError):
        engine.inverse_kinematics([0.20, 0.0, -0.05], enforce_safety=True)

    with pytest.raises(TableCollisionError):
        engine.inverse_kinematics([0.20, 0.0, 0.005], enforce_safety=True)  # < TABLE_Z_MIN (0.015)

    # Out of workspace radius
    with pytest.raises(WorkspaceLimitError):
        engine.inverse_kinematics([0.45, 0.0, 0.10], enforce_safety=True)

    with pytest.raises(WorkspaceLimitError):
        engine.inverse_kinematics([0.02, 0.01, 0.10], enforce_safety=True)  # < REACH_R_MIN (0.08)

    # Clamping allows safe bounding
    safe_pos = engine.check_safety(np.array([0.20, 0.0, -0.05]), clamp=True)
    assert safe_pos[2] == TABLE_Z_MIN
