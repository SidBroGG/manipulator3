import pytest
import numpy as np
import cv2

from manipulator3.grasping import (
    ApproachStatus,
    GraspController,
    ToFApproachController,
    GRIPPER_OPEN,
    GRIPPER_GRASP_CMD,
    GRIPPER_CLOSED_EMPTY,
    TOF_TARGET_MIN_MM,
    TOF_TARGET_MAX_MM,
    TOF_TARGET_NOMINAL_MM,
    TABLE_CLEARANCE_MIN_EE_Z,
)
from manipulator3.kinematics import (
    KinematicsEngine,
    POSE_HOME,
    POSE_SCAN,
    POSE_DROP_OFF,
    TABLE_Z_MIN,
)
from manipulator3.pointing import PointingTarget, PointingStatus
from manipulator3.servoing import SectorScanner
from manipulator3.simulation import SimulationEnvironment
from manipulator3.state_machine import (
    ManipulatorState,
    PickAndPlaceStateMachine,
    StateMachineOutput,
)


def test_tof_approach_controller_step_and_thresholds():
    """7.1.1 & 7.1.2 Verify ToF approach step along line of sight and distance thresholds."""
    ke = KinematicsEngine()
    approach = ToFApproachController(ke)

    assert approach.is_at_grasp_distance(TOF_TARGET_NOMINAL_MM) is True
    assert approach.is_at_grasp_distance(30.0) is True
    assert approach.is_at_grasp_distance(15.0) is False
    assert approach.is_at_grasp_distance(120.0) is False

    # Out of range test
    q_scan = POSE_SCAN.copy()
    q_out, status_out = approach.compute_approach_step(q_scan, tof_mm=-1.0)
    assert status_out == ApproachStatus.OUT_OF_RANGE
    assert np.allclose(q_out, q_scan)

    # Step towards target along line of sight
    q_next, status = approach.compute_approach_step(q_scan, tof_mm=120.0)
    assert status == ApproachStatus.APPROACHING

    pos_init, rot_init = ke.forward_kinematics(q_scan)
    pos_next, rot_next = ke.forward_kinematics(q_next)

    # End-effector must advance in the direction of rot_init[:, 2]
    step_vec = pos_next - pos_init
    cos_sim = np.dot(step_vec, rot_init[:, 2]) / (np.linalg.norm(step_vec) * np.linalg.norm(rot_init[:, 2]))
    assert cos_sim > 0.95

    # Target reached threshold test
    q_reached, status_reached = approach.compute_approach_step(q_scan, tof_mm=32.0)
    assert status_reached == ApproachStatus.TARGET_REACHED
    assert np.allclose(q_reached, q_scan)


def test_tof_approach_table_safety_boundary():
    """7.1.3 Verify approach stops and avoids table collision when clearance limit is reached."""
    ke = KinematicsEngine()
    approach = ToFApproachController(ke, min_ee_z=0.195)  # threshold just below scan height (0.200m)

    q_scan = POSE_SCAN.copy()
    q_res, status = approach.compute_approach_step(q_scan, tof_mm=300.0)
    assert status == ApproachStatus.TABLE_LIMIT

    pos_res, _ = ke.forward_kinematics(q_res)
    assert pos_res[2] >= TABLE_Z_MIN


def test_grasp_grip_verification():
    """7.2.1 Verify gripper contact / grip verification logic."""
    controller = GraspController()

    # Normal object gripped: jaw stopped partially closed
    assert controller.verify_grip(-0.25) is True
    assert controller.verify_grip(-0.35) is True
    assert controller.verify_grip(-0.15) is True

    # Empty jaws: closed fully past contact threshold
    assert controller.verify_grip(GRIPPER_CLOSED_EMPTY) is False
    assert controller.verify_grip(-0.52) is False

    # Open jaws: failed to close
    assert controller.verify_grip(GRIPPER_OPEN) is False
    assert controller.verify_grip(0.0) is False


def test_grasp_pre_transfer_lift_and_transport_plans():
    """7.2.2 & 7.2.3 Verify pre-transfer vertical lift (80 mm) and transport trajectories."""
    ke = KinematicsEngine()
    grasp = GraspController(ke)

    q_init = POSE_SCAN.copy()
    pos_init, _ = ke.forward_kinematics(q_init)

    traj_lift = grasp.plan_pre_transfer_lift(q_init, duration=0.8)
    assert len(traj_lift.positions) > 5

    pos_lift, _ = ke.forward_kinematics(traj_lift.positions[-1])
    assert pos_lift[2] - pos_init[2] == pytest.approx(0.080, abs=0.005)

    traj_transport = grasp.plan_transport(traj_lift.positions[-1], duration=1.5)
    assert np.allclose(traj_transport.positions[-1], POSE_DROP_OFF)

    traj_home = grasp.plan_return_home(POSE_DROP_OFF, duration=1.2)
    assert np.allclose(traj_home.positions[-1], POSE_HOME)


def test_state_machine_transitions():
    """7.3.1 Verify state machine transition sequence from IDLE to HOME."""
    sm = PickAndPlaceStateMachine(target_name="red_cube")

    # Initial state
    assert sm.state == ManipulatorState.IDLE
    out = sm.step()
    assert out.state == ManipulatorState.IDLE

    # Gesture trigger
    target = PointingTarget(
        status=PointingStatus.VALID,
        point_2d=np.array([0.18, 0.0]),
        azimuth=0.0,
        is_reachable=True,
    )
    sm.trigger_pointing(target)
    assert sm.state == ManipulatorState.POINTING_DETECTED

    # Advance to SECTOR_SCAN
    out = sm.step()
    assert out.state == ManipulatorState.SECTOR_SCAN

    # Synthesize frame with red blob to trigger VISUAL_ALIGN
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (310, 230), (330, 250), (20, 20, 220), -1)  # centered red cube BGR

    out = sm.step(frame=frame)
    assert out.state == ManipulatorState.VISUAL_ALIGN

    # Complete visual alignment -> TOF_APPROACH
    for _ in range(20):
        out = sm.step(frame=frame, tof_mm=80.0)
        if out.state == ManipulatorState.TOF_APPROACH:
            break
    assert sm.state == ManipulatorState.TOF_APPROACH

    # Approach until target ToF distance -> GRASP
    out = sm.step(frame=frame, tof_mm=34.0)
    assert out.state == ManipulatorState.GRASP
    assert sm.commanded_gripper == GRIPPER_GRASP_CMD

    # Step through grasp squeeze and lift (15 squeeze + ~41 lift steps)
    for _ in range(80):
        out = sm.step(gripper_joint_pos=-0.25)
        if out.state == ManipulatorState.TRANSPORT:
            break
    assert sm.state == ManipulatorState.TRANSPORT

    # Step through transport -> RELEASE (~107 trajectory steps)
    for _ in range(130):
        out = sm.step()
        if out.state == ManipulatorState.RELEASE:
            break
    assert sm.state == ManipulatorState.RELEASE
    assert sm.commanded_gripper == GRIPPER_OPEN

    # Step through release -> HOME -> IDLE
    for _ in range(150):
        out = sm.step()
        if out.is_finished:
            break
    assert out.is_finished is True
    assert sm.state == ManipulatorState.IDLE


def test_simulation_e2e_pick_and_place():
    """7.4.1 Test full autonomous pick-and-place loop in MuJoCo simulation."""
    env = SimulationEnvironment()
    try:
        sm = PickAndPlaceStateMachine(target_name="red_cube")

        cube_id = env.model.body("target_cube").id
        init_pos = env.data.xpos[cube_id].copy()

        # Run autonomous cycle with pointing input at azimuth 0.0
        success = sm.run_simulation_cycle(
            env=env,
            pointing_target=0.0,
            max_steps=500,
            physics_substeps=15,
        )

        assert success is True
        final_pos = env.data.xpos[cube_id].copy()

        # Cube must be significantly displaced from its starting tabletop coordinates (0.18, 0.0)
        displacement = float(np.linalg.norm(final_pos[:2] - init_pos[:2]))
        assert displacement > 0.08  # at least 80 mm displacement to drop-off zone
    finally:
        env.close()
