import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from manipulator3.detector import DetectedObject
from manipulator3.kinematics import POSE_SCAN
from manipulator3.pointing import PointingTarget
from manipulator3.trajectory import TrajectoryPlanner, Trajectory

logger = logging.getLogger(__name__)

# Joint indices for 5-DOF arm
JOINT_YAW = 0    # revolute_1 (base yaw)
JOINT_PITCH = 3  # revolute_4 (wrist pitch)

DEFAULT_JOINT_LIMITS: Dict[int, Tuple[float, float]] = {
    JOINT_YAW: (-3.14159, 3.14159),
    JOINT_PITCH: (-1.5708, 1.5708),
}


class TrackingState(str, Enum):
    IDLE = "IDLE"
    ORIENTING = "ORIENTING"
    SWEEPING = "SWEEPING"
    CENTERING = "CENTERING"
    ALIGNED = "ALIGNED"
    LOST = "LOST"


class VisualServoController:
    """Proportional visual servoing controller for centering target object in gripper camera."""

    def __init__(
        self,
        k_yaw: float = 0.35,
        k_pitch: float = 0.30,
        max_step_rad: float = 0.08,
        tolerance_norm: float = 0.04,
        joint_limits: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> None:
        self.k_yaw = k_yaw
        self.k_pitch = k_pitch
        self.max_step_rad = max_step_rad
        self.tolerance_norm = tolerance_norm
        self.joint_limits = joint_limits or DEFAULT_JOINT_LIMITS

    def is_centered(self, target: DetectedObject) -> bool:
        """Check if target center is within normalized tolerance deadband."""
        eu, ev = target.norm_center
        return abs(eu) <= self.tolerance_norm and abs(ev) <= self.tolerance_norm

    def compute_joint_deltas(self, target: DetectedObject) -> Tuple[float, float]:
        """Compute (dq_yaw, dq_pitch) joint increments to drive visual error to zero."""
        eu, ev = target.norm_center
        # Negative proportional law: positive error requires negative joint delta
        dq_yaw = float(np.clip(-self.k_yaw * eu, -self.max_step_rad, self.max_step_rad))
        dq_pitch = float(np.clip(-self.k_pitch * ev, -self.max_step_rad, self.max_step_rad))
        return dq_yaw, dq_pitch

    def step(
        self,
        target: DetectedObject,
        current_q: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Perform one visual servo control step.
        Returns:
            (new_q, is_aligned): updated joint angles and alignment boolean flag.
        """
        new_q = np.asarray(current_q, dtype=np.float64).copy()
        if self.is_centered(target):
            return new_q, True

        dq_yaw, dq_pitch = self.compute_joint_deltas(target)
        new_q[JOINT_YAW] += dq_yaw
        new_q[JOINT_PITCH] += dq_pitch

        # Clamp within hardware limits
        lim_yaw = self.joint_limits.get(JOINT_YAW, (-3.14159, 3.14159))
        lim_pitch = self.joint_limits.get(JOINT_PITCH, (-1.5708, 1.5708))
        new_q[JOINT_YAW] = float(np.clip(new_q[JOINT_YAW], lim_yaw[0], lim_yaw[1]))
        new_q[JOINT_PITCH] = float(np.clip(new_q[JOINT_PITCH], lim_pitch[0], lim_pitch[1]))

        return new_q, False


class SectorScanner:
    """Generates sector observation poses and sweep trajectories for out-of-FOV object search."""

    def __init__(
        self,
        base_pose: Optional[np.ndarray] = None,
        default_span_rad: float = float(np.deg2rad(40.0)),
    ) -> None:
        self.base_pose = base_pose if base_pose is not None else POSE_SCAN.copy()
        self.default_span_rad = default_span_rad
        self.trajectory_planner = TrajectoryPlanner()

    def get_sector_pose(
        self,
        azimuth_rad: float,
        pitch_rad: Optional[float] = None,
    ) -> np.ndarray:
        """Return baseline scan pose rotated towards target azimuth."""
        q = self.base_pose.copy()
        # In manipulator frame, base yaw revolute_1 rotates around -Z relative to world
        q[JOINT_YAW] = -float(azimuth_rad)
        if pitch_rad is not None:
            q[JOINT_PITCH] = float(pitch_rad)
        return q

    def orient_to_pointing(self, pointing_target: PointingTarget) -> np.ndarray:
        """Orient arm towards Stage 5 pointing target sector center."""
        return self.get_sector_pose(pointing_target.sector_center_rad)

    def generate_sweep_waypoints(
        self,
        azimuth_rad: float,
        span_rad: Optional[float] = None,
        n_waypoints: int = 5,
        tilt_offsets: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        """
        Generate sequence of joint waypoints sweeping across the pointing sector.
        Sweep order: center -> min_yaw -> max_yaw -> center.
        """
        span = span_rad if span_rad is not None else self.default_span_rad
        tilts = tilt_offsets or [0.0]

        half_span = span / 2.0
        min_az = azimuth_rad - half_span
        max_az = azimuth_rad + half_span

        waypoints: List[np.ndarray] = []
        for dtilt in tilts:
            pitch = float(self.base_pose[JOINT_PITCH] + dtilt)
            # Oscillating sweep across azimuth
            azimuth_sequence = [azimuth_rad, min_az, max_az, azimuth_rad]
            for az in azimuth_sequence:
                q = self.get_sector_pose(az, pitch_rad=pitch)
                waypoints.append(q)

        return waypoints

    def generate_sweep_trajectories(
        self,
        waypoints: List[np.ndarray],
        segment_duration: float = 0.8,
        dt: float = 0.02,
    ) -> List[Trajectory]:
        """Generate smooth minimum-jerk trajectory segments connecting sweep waypoints."""
        trajectories: List[Trajectory] = []
        for i in range(len(waypoints) - 1):
            q_start = waypoints[i]
            q_goal = waypoints[i + 1]
            traj = self.trajectory_planner.generate_joint_trajectory(
                q_start=q_start,
                q_goal=q_goal,
                duration=segment_duration,
                dt=dt,
            )
            trajectories.append(traj)
        return trajectories


class VisualSearchTracker:
    """Autonomous search, scanning sweep, and visual servoing pipeline."""

    def __init__(
        self,
        detector,
        servo_controller: Optional[VisualServoController] = None,
        scanner: Optional[SectorScanner] = None,
    ) -> None:
        self.detector = detector
        self.servo = servo_controller or VisualServoController()
        self.scanner = scanner or SectorScanner()

        self.state: TrackingState = TrackingState.IDLE
        self.active_waypoints: List[np.ndarray] = []
        self.waypoint_idx: int = 0
        self.lost_counter: int = 0
        self.max_lost_frames: int = 5

    def start_search(
        self,
        azimuth_rad: Optional[float] = None,
        pointing_target: Optional[PointingTarget] = None,
    ) -> np.ndarray:
        """Initialize search in specified pointing sector."""
        if pointing_target is not None:
            az = pointing_target.sector_center_rad
            span = pointing_target.sector_span_rad
        elif azimuth_rad is not None:
            az = float(azimuth_rad)
            span = self.scanner.default_span_rad
        else:
            az = 0.0
            span = self.scanner.default_span_rad

        self.active_waypoints = self.scanner.generate_sweep_waypoints(
            azimuth_rad=az,
            span_rad=span,
            tilt_offsets=[0.0, 0.12],
        )
        self.waypoint_idx = 0
        self.lost_counter = 0
        self.state = TrackingState.ORIENTING
        return self.active_waypoints[0]

    def update(
        self,
        frame: np.ndarray,
        current_q: np.ndarray,
    ) -> Tuple[np.ndarray, TrackingState, Optional[DetectedObject]]:
        """
        Process incoming frame and current joint configuration.
        Returns:
            (commanded_q, state, best_detection)
        """
        detections = self.detector.detect(frame)
        best_target = self.detector.get_best_target(detections) if hasattr(self.detector, "get_best_target") else (detections[0] if detections else None)

        if best_target is not None:
            # Target acquired -> switch to or maintain visual centering
            self.lost_counter = 0
            new_q, is_aligned = self.servo.step(best_target, current_q)
            self.state = TrackingState.ALIGNED if is_aligned else TrackingState.CENTERING
            return new_q, self.state, best_target

        # No target detected in frame
        if self.state in (TrackingState.CENTERING, TrackingState.ALIGNED):
            self.lost_counter += 1
            if self.lost_counter <= self.max_lost_frames:
                # Keep current pose temporarily
                return current_q.copy(), TrackingState.CENTERING, None
            # Lost target completely -> resume search sweep
            logger.info("Target lost, resuming sweep")
            self.state = TrackingState.SWEEPING

        if self.state in (TrackingState.ORIENTING, TrackingState.SWEEPING):
            if not self.active_waypoints:
                return current_q.copy(), TrackingState.IDLE, None

            target_wp = self.active_waypoints[self.waypoint_idx]
            dist_to_wp = float(np.linalg.norm(target_wp - current_q))
            if dist_to_wp < 0.05:
                self.waypoint_idx = (self.waypoint_idx + 1) % len(self.active_waypoints)
                self.state = TrackingState.SWEEPING
                target_wp = self.active_waypoints[self.waypoint_idx]

            # Step towards target waypoint
            step_dir = target_wp - current_q
            step_norm = float(np.linalg.norm(step_dir))
            if step_norm > 0.04:
                step_dir = (step_dir / step_norm) * 0.04
            new_q = current_q + step_dir
            return new_q, self.state, None

        return current_q.copy(), self.state, None
