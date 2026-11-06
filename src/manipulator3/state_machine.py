import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np

from manipulator3.detector import ColorBlobDetector, DetectedObject, YOLOObjectDetector
from manipulator3.grasping import (
    ApproachStatus,
    GraspController,
    ToFApproachController,
    GRIPPER_OPEN,
    GRIPPER_GRASP_CMD,
)
from manipulator3.kinematics import (
    POSE_HOME,
    POSE_SCAN,
    POSE_DROP_OFF,
    KinematicsEngine,
)
from manipulator3.pointing import PointingTarget
from manipulator3.servoing import SectorScanner, VisualServoController
from manipulator3.trajectory import Trajectory, TrajectoryPlanner

logger = logging.getLogger(__name__)


class ManipulatorState(str, Enum):
    IDLE = "IDLE"
    POINTING_DETECTED = "POINTING_DETECTED"
    SECTOR_SCAN = "SECTOR_SCAN"
    VISUAL_ALIGN = "VISUAL_ALIGN"
    TOF_APPROACH = "TOF_APPROACH"
    GRASP = "GRASP"
    TRANSPORT = "TRANSPORT"
    RELEASE = "RELEASE"
    HOME = "HOME"


@dataclass
class StateMachineOutput:
    state: ManipulatorState
    commanded_q: np.ndarray
    commanded_gripper: float
    is_finished: bool = False
    message: str = ""


class PickAndPlaceStateMachine:
    """Core autonomous pick-and-place finite state machine."""

    def __init__(
        self,
        detector=None,
        kinematics_engine: Optional[KinematicsEngine] = None,
        trajectory_planner: Optional[TrajectoryPlanner] = None,
        servo_controller: Optional[VisualServoController] = None,
        scanner: Optional[SectorScanner] = None,
        approach_controller: Optional[ToFApproachController] = None,
        grasp_controller: Optional[GraspController] = None,
        target_name: Optional[str] = "red_cube",
    ) -> None:
        self.kinematics = kinematics_engine or KinematicsEngine()
        self.planner = trajectory_planner or TrajectoryPlanner(self.kinematics)
        self.detector = detector or ColorBlobDetector(
            target_colors=[target_name] if target_name else None
        )
        self.servo = servo_controller or VisualServoController()
        self.scanner = scanner or SectorScanner()
        self.approach = approach_controller or ToFApproachController(self.kinematics)
        self.grasp = grasp_controller or GraspController(self.kinematics, self.planner)
        self.target_name = target_name

        self.state: ManipulatorState = ManipulatorState.IDLE
        self.pointing_target: Optional[PointingTarget] = None
        self.locked_target: Optional[DetectedObject] = None

        self.commanded_q: np.ndarray = POSE_HOME.copy()
        self.commanded_gripper: float = self.open_gripper_command

        # Sector sweep tracking
        self.active_waypoints: list[np.ndarray] = []
        self.waypoint_idx: int = 0

        # Trajectory following tracking
        self.active_trajectory: Optional[Trajectory] = None
        self.traj_step_idx: int = 0

        # Counters
        self.state_step_count: int = 0
        self.lost_frames_count: int = 0
        self.max_lost_frames: int = 8
        self.grasp_substate: str = "CLOSE"

    @property
    def open_gripper_command(self) -> float:
        """Target opening angle for gripper (radians). Narrower for blue_box to avoid clipping adjacent objects."""
        if self.target_name == "blue_box":
            return 0.28
        return GRIPPER_OPEN

    def reset(self) -> None:
        """Reset state machine to initial IDLE state."""
        self.state = ManipulatorState.IDLE
        self.pointing_target = None
        self.locked_target = None
        self.commanded_q = POSE_HOME.copy()
        self.commanded_gripper = self.open_gripper_command
        self.active_waypoints.clear()
        self.waypoint_idx = 0
        self.active_trajectory = None
        self.traj_step_idx = 0
        self.state_step_count = 0
        self.lost_frames_count = 0
        self.grasp_substate = "CLOSE"

    def set_target(self, target_name: str) -> None:
        """Configure target object name and update detector."""
        self.target_name = target_name
        if isinstance(self.detector, ColorBlobDetector):
            self.detector.target_colors = {target_name}
        if self.state in (
            ManipulatorState.IDLE,
            ManipulatorState.SECTOR_SCAN,
            ManipulatorState.VISUAL_ALIGN,
            ManipulatorState.TOF_APPROACH,
        ):
            self.commanded_gripper = self.open_gripper_command
        logger.info("Configured state machine target: %s", target_name)

    def trigger_pointing(
        self,
        target: Optional[PointingTarget | float] = None,
        azimuth_rad: Optional[float] = None,
        span_rad: Optional[float] = None,
    ) -> None:
        """Transition from IDLE to POINTING_DETECTED upon human gesture input."""
        az = 0.0
        span = self.scanner.default_span_rad

        if isinstance(target, PointingTarget):
            # Check reachability if status or is_reachable present
            if (
                hasattr(target, "is_reachable")
                and not target.is_reachable
                and hasattr(target, "status")
            ):
                if (
                    getattr(target.status, "value", str(target.status))
                    != "TARGET_REACHABLE"
                ):
                    logger.warning("Rejected pointing target: not reachable")
                    return
            az = target.azimuth if target.azimuth is not None else 0.0
            if target.search_sector is not None:
                span = abs(target.search_sector[1] - target.search_sector[0])
            self.pointing_target = target
        elif isinstance(target, (int, float)):
            az = float(target)
        elif azimuth_rad is not None:
            az = float(azimuth_rad)

        if span_rad is not None:
            span = float(span_rad)

        self.active_waypoints = self.scanner.generate_sweep_waypoints(
            azimuth_rad=az,
            span_rad=span,
            tilt_offsets=[0.38, 0.15],
        )
        self.waypoint_idx = 0
        self.state = ManipulatorState.POINTING_DETECTED
        self.state_step_count = 0
        logger.info(
            "Pointing detected. Target azimuth: %.2f rad, span: %.2f rad", az, span
        )

    def step(
        self,
        frame: Optional[np.ndarray] = None,
        tof_mm: float = -1.0,
        actual_q: Optional[np.ndarray] = None,
        gripper_joint_pos: float = GRIPPER_OPEN,
    ) -> StateMachineOutput:
        """Execute one control step of the state machine."""
        current_q = actual_q if actual_q is not None else self.commanded_q.copy()

        if self.state == ManipulatorState.IDLE:
            self.commanded_gripper = GRIPPER_OPEN
            return StateMachineOutput(
                state=self.state,
                commanded_q=self.commanded_q.copy(),
                commanded_gripper=self.commanded_gripper,
                message="Waiting for trigger",
            )

        elif self.state == ManipulatorState.POINTING_DETECTED:
            if not self.active_waypoints:
                az = getattr(self.pointing_target, "azimuth", 0.0) or 0.0
                self.active_waypoints = self.scanner.generate_sweep_waypoints(
                    azimuth_rad=az
                )
            self.waypoint_idx = 0
            self.commanded_q = self.active_waypoints[0].copy()
            self.state = ManipulatorState.SECTOR_SCAN
            logger.info(
                "Transition -> SECTOR_SCAN (waypoints: %d)", len(self.active_waypoints)
            )
            return StateMachineOutput(
                state=self.state,
                commanded_q=self.commanded_q.copy(),
                commanded_gripper=self.commanded_gripper,
                message="Orienting to scan sector",
            )

        elif self.state == ManipulatorState.SECTOR_SCAN:
            self.commanded_gripper = self.open_gripper_command
            # Scan frame for candidates
            if frame is not None:
                detections = self.detector.detect(frame)
                if detections:
                    self.locked_target = detections[0]
                    self.state = ManipulatorState.VISUAL_ALIGN
                    self.lost_frames_count = 0
                    logger.info(
                        "Target acquired: %s at norm %s. Transition -> VISUAL_ALIGN",
                        self.locked_target.class_name,
                        self.locked_target.norm_center,
                    )
                    return StateMachineOutput(
                        state=self.state,
                        commanded_q=self.commanded_q.copy(),
                        commanded_gripper=self.commanded_gripper,
                        message="Target acquired, starting visual servoing",
                    )

            # Advance sweep waypoints
            if self.active_waypoints:
                target_wp = self.active_waypoints[self.waypoint_idx]
                dist = float(np.linalg.norm(target_wp - current_q))
                if dist < 0.06:
                    self.waypoint_idx = (self.waypoint_idx + 1) % len(
                        self.active_waypoints
                    )
                    target_wp = self.active_waypoints[self.waypoint_idx]

                step_dir = target_wp - current_q
                step_norm = float(np.linalg.norm(step_dir))
                if step_norm > 0.04:
                    step_dir = (step_dir / step_norm) * 0.04
                self.commanded_q = current_q + step_dir

            return StateMachineOutput(
                state=self.state,
                commanded_q=self.commanded_q.copy(),
                commanded_gripper=self.commanded_gripper,
                message="Scanning sector",
            )

        elif self.state == ManipulatorState.VISUAL_ALIGN:
            self.commanded_gripper = self.open_gripper_command
            best_det = None
            if frame is not None:
                dets = self.detector.detect(frame)
                if dets:
                    best_det = dets[0]

            if best_det is not None:
                self.lost_frames_count = 0
                self.commanded_q, is_aligned = self.servo.step(
                    best_det, self.commanded_q, actual_q=actual_q
                )
                if is_aligned:
                    logger.info(
                        "Visual alignment locked. ToF: %.1f mm. Transition -> TOF_APPROACH",
                        tof_mm,
                    )
                    self.state = ManipulatorState.TOF_APPROACH
                    self.state_step_count = 0
            else:
                self.lost_frames_count += 1
                if self.lost_frames_count > self.max_lost_frames:
                    logger.warning(
                        "Target lost during visual alignment, resuming SECTOR_SCAN"
                    )
                    self.state = ManipulatorState.SECTOR_SCAN

            return StateMachineOutput(
                state=self.state,
                commanded_q=self.commanded_q.copy(),
                commanded_gripper=self.commanded_gripper,
                message="Centering target in frame",
            )

        elif self.state == ManipulatorState.TOF_APPROACH:
            self.commanded_gripper = self.open_gripper_command
            self.state_step_count += 1

            ee_pos, _ = self.kinematics.forward_kinematics(current_q)

            if self.approach.is_at_grasp_distance(tof_mm) or (
                ee_pos[2] <= self.approach.min_ee_z and self.state_step_count >= 3
            ):
                logger.info(
                    "Target reached (ToF: %.1f mm, EE Z: %.3f m). Transition -> GRASP",
                    tof_mm,
                    ee_pos[2],
                )
                self.state = ManipulatorState.GRASP
                self.grasp_substate = "CLOSE"
                self.commanded_gripper = GRIPPER_GRASP_CMD
                self.state_step_count = 0
                return StateMachineOutput(
                    state=self.state,
                    commanded_q=self.commanded_q.copy(),
                    commanded_gripper=self.commanded_gripper,
                    message="Target distance reached",
                )

            next_q, status = self.approach.compute_approach_step(
                self.commanded_q, tof_mm
            )
            if (
                status == ApproachStatus.TARGET_REACHED
                or status == ApproachStatus.TABLE_LIMIT
            ):
                logger.info(
                    "Approach concluded (%s). Transition -> GRASP", status.value
                )
                self.state = ManipulatorState.GRASP
                self.grasp_substate = "CLOSE"
                self.commanded_gripper = GRIPPER_GRASP_CMD
                self.state_step_count = 0
            elif status == ApproachStatus.APPROACHING:
                self.commanded_q = next_q
                self.lost_frames_count = 0
            elif status == ApproachStatus.OUT_OF_RANGE:
                if ee_pos[2] <= self.approach.min_ee_z and self.state_step_count >= 5:
                    logger.info(
                        "Close to target at table Z=%.3fm, concluding approach -> GRASP",
                        ee_pos[2],
                    )
                    self.state = ManipulatorState.GRASP
                    self.grasp_substate = "CLOSE"
                    self.commanded_gripper = GRIPPER_GRASP_CMD
                    self.state_step_count = 0
                else:
                    self.lost_frames_count += 1
                    if self.lost_frames_count > 30:
                        logger.warning(
                            "ToF signal lost during approach, returning to VISUAL_ALIGN"
                        )
                        self.state = ManipulatorState.VISUAL_ALIGN
                        self.lost_frames_count = 0

            return StateMachineOutput(
                state=self.state,
                commanded_q=self.commanded_q.copy(),
                commanded_gripper=self.commanded_gripper,
                message=f"ToF approach ({tof_mm:.1f} mm)",
            )

        elif self.state == ManipulatorState.GRASP:
            if self.grasp_substate == "CLOSE":
                self.commanded_gripper = GRIPPER_GRASP_CMD
                self.state_step_count += 1
                if self.state_step_count >= 20:
                    is_gripped = self.grasp.verify_grip(gripper_joint_pos)
                    logger.info(
                        "Grip verification: %s (pos=%.4f rad)",
                        is_gripped,
                        gripper_joint_pos,
                    )
                    # Plan pre-transfer lift
                    self.active_trajectory = self.grasp.plan_pre_transfer_lift(
                        self.commanded_q, duration=0.8
                    )
                    self.traj_step_idx = 0
                    self.grasp_substate = "LIFT"

            elif self.grasp_substate == "LIFT":
                if self.active_trajectory is not None and self.traj_step_idx < len(
                    self.active_trajectory.positions
                ):
                    self.commanded_q = self.active_trajectory.positions[
                        self.traj_step_idx
                    ].copy()
                    self.traj_step_idx += 1
                else:
                    logger.info("Pre-transfer lift completed. Transition -> TRANSPORT")
                    self.state = ManipulatorState.TRANSPORT
                    self.active_trajectory = self.grasp.plan_transport(
                        self.commanded_q, duration=1.5
                    )
                    self.traj_step_idx = 0

            return StateMachineOutput(
                state=self.state,
                commanded_q=self.commanded_q.copy(),
                commanded_gripper=self.commanded_gripper,
                message=f"Grasping ({self.grasp_substate})",
            )

        elif self.state == ManipulatorState.TRANSPORT:
            self.commanded_gripper = GRIPPER_GRASP_CMD
            if self.active_trajectory is not None and self.traj_step_idx < len(
                self.active_trajectory.positions
            ):
                self.commanded_q = self.active_trajectory.positions[
                    self.traj_step_idx
                ].copy()
                self.traj_step_idx += 1
            else:
                logger.info("Transport completed. Transition -> RELEASE")
                self.state = ManipulatorState.RELEASE
                self.commanded_gripper = self.open_gripper_command
                self.state_step_count = 0

            return StateMachineOutput(
                state=self.state,
                commanded_q=self.commanded_q.copy(),
                commanded_gripper=self.commanded_gripper,
                message="Transporting object to drop-off pad",
            )

        elif self.state == ManipulatorState.RELEASE:
            self.commanded_gripper = self.open_gripper_command
            self.state_step_count += 1
            if self.state_step_count >= 15:
                logger.info("Release complete. Transition -> HOME")
                self.state = ManipulatorState.HOME
                self.active_trajectory = self.grasp.plan_return_home(
                    self.commanded_q, duration=1.2
                )
                self.traj_step_idx = 0

            return StateMachineOutput(
                state=self.state,
                commanded_q=self.commanded_q.copy(),
                commanded_gripper=self.commanded_gripper,
                message="Releasing object",
            )

        elif self.state == ManipulatorState.HOME:
            self.commanded_gripper = self.open_gripper_command
            if self.active_trajectory is not None and self.traj_step_idx < len(
                self.active_trajectory.positions
            ):
                self.commanded_q = self.active_trajectory.positions[
                    self.traj_step_idx
                ].copy()
                self.traj_step_idx += 1
            else:
                logger.info("Autonomous cycle complete. Returning to IDLE")
                self.state = ManipulatorState.IDLE
                return StateMachineOutput(
                    state=self.state,
                    commanded_q=POSE_HOME.copy(),
                    commanded_gripper=self.open_gripper_command,
                    is_finished=True,
                    message="Cycle completed successfully",
                )

            return StateMachineOutput(
                state=self.state,
                commanded_q=self.commanded_q.copy(),
                commanded_gripper=self.commanded_gripper,
                message="Returning to HOME pose",
            )

        return StateMachineOutput(
            state=self.state,
            commanded_q=self.commanded_q.copy(),
            commanded_gripper=self.commanded_gripper,
        )

    def run_simulation_cycle(
        self,
        env,
        pointing_target: Optional[PointingTarget | float] = None,
        max_steps: int = 500,
        physics_substeps: int = 15,
    ) -> bool:
        """Autonomous simulation runner helper executing end-to-end pick-and-place loop."""
        if pointing_target is not None:
            self.trigger_pointing(pointing_target)
        else:
            # Default target sector at X=0.18, Y=0.0
            from manipulator3.pointing import PointingTarget

            self.trigger_pointing(
                PointingTarget(
                    x_table=0.18, y_table=0.0, is_valid=True, sector_center_rad=0.0
                )
            )

        curr_grip = self.open_gripper_command
        for step_idx in range(max_steps):
            frame = env.render_bgr("gripper_camera")
            tof_mm = env.get_tof_distance_mm()
            jpos = env.get_joint_positions()
            actual_q = np.array(
                [jpos[f"revolute_{i + 1}"] for i in range(5)], dtype=np.float64
            )
            grip_pos = jpos.get("revolute_7", self.open_gripper_command)

            output = self.step(
                frame=frame,
                tof_mm=tof_mm,
                actual_q=actual_q,
                gripper_joint_pos=grip_pos,
            )

            # Smooth gripper rate limiting
            diff = output.commanded_gripper - curr_grip
            curr_grip += float(np.clip(diff, -0.04, 0.04))

            # Apply actuator targets
            targets = list(output.commanded_q) + [curr_grip]
            env.set_actuator_targets(targets)

            # Step physics
            for _ in range(physics_substeps):
                env.step()

            if output.is_finished:
                return True

        return False
