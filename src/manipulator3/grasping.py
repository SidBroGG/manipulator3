import logging
from enum import Enum
from typing import Optional, Tuple
import numpy as np

from manipulator3.kinematics import (
    KinematicsEngine,
    POSE_HOME,
    POSE_DROP_OFF,
    TABLE_Z_MIN,
)
from manipulator3.trajectory import TrajectoryPlanner, Trajectory

logger = logging.getLogger(__name__)

# Gripper actuator setpoints and limits (radians)
GRIPPER_OPEN: float = 0.40
GRIPPER_GRASP_CMD: float = -0.52
GRIPPER_CLOSED_EMPTY: float = -0.58

# Contact detection thresholds for revolute_7
GRIPPER_CONTACT_MIN_RAD: float = -0.48
GRIPPER_CONTACT_MAX_RAD: float = -0.05

# ToF approach parameters (millimeters and meters)
TOF_TARGET_MIN_MM: float = 25.0
TOF_TARGET_MAX_MM: float = 42.0
TOF_TARGET_NOMINAL_MM: float = 35.0

# Software clearance safety boundary for end-effector during angled approach
TABLE_CLEARANCE_MIN_EE_Z: float = 0.090
PRE_TRANSFER_LIFT_M: float = 0.080


class ApproachStatus(str, Enum):
    APPROACHING = "APPROACHING"
    TARGET_REACHED = "TARGET_REACHED"
    TABLE_LIMIT = "TABLE_LIMIT"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    IK_FAILED = "IK_FAILED"


class ToFApproachController:
    """Closed-loop distance-based approach controller driving end-effector along line of sight."""

    def __init__(
        self,
        kinematics_engine: Optional[KinematicsEngine] = None,
        target_tof_mm: float = TOF_TARGET_NOMINAL_MM,
        tof_range_mm: Tuple[float, float] = (TOF_TARGET_MIN_MM, TOF_TARGET_MAX_MM),
        max_step_m: float = 0.010,
        min_step_m: float = 0.002,
        min_ee_z: float = TABLE_CLEARANCE_MIN_EE_Z,
    ) -> None:
        self.kinematics = kinematics_engine or KinematicsEngine()
        self.target_tof_mm = target_tof_mm
        self.tof_range_mm = tof_range_mm
        self.max_step_m = max_step_m
        self.min_step_m = min_step_m
        self.min_ee_z = min_ee_z

    def is_at_grasp_distance(self, tof_mm: float) -> bool:
        """Check if ToF reading is within the optimal grasp distance window."""
        return self.tof_range_mm[0] <= tof_mm <= self.tof_range_mm[1]

    def compute_approach_step(
        self,
        current_q: np.ndarray,
        tof_mm: float,
    ) -> Tuple[np.ndarray, ApproachStatus]:
        """Compute one incremental step along the sensor optical axis towards target."""
        if tof_mm <= 0.0:
            return current_q.copy(), ApproachStatus.OUT_OF_RANGE

        if self.is_at_grasp_distance(tof_mm):
            return current_q.copy(), ApproachStatus.TARGET_REACHED

        ee_pos, ee_rot = self.kinematics.forward_kinematics(current_q)
        line_of_sight = ee_rot[:, 2]

        # Step proportional to remaining distance
        dist_err_m = (tof_mm - self.target_tof_mm) * 0.001
        step_m = float(np.clip(dist_err_m * 0.35, self.min_step_m, self.max_step_m))

        target_pos = ee_pos + line_of_sight * step_m

        if target_pos[2] < self.min_ee_z:
            logger.warning("Approach reached table clearance safety limit: Z=%.4fm", target_pos[2])
            return current_q.copy(), ApproachStatus.TABLE_LIMIT

        try:
            next_q = self.kinematics.inverse_kinematics(
                target_pos,
                target_z_axis=line_of_sight,
                initial_guess=current_q,
                tolerance=3e-3,
            )
            return next_q, ApproachStatus.APPROACHING
        except Exception as e:
            logger.error("IK failed during ToF approach: %s", e)
            return current_q.copy(), ApproachStatus.IK_FAILED

    def compute_grasp_pose_from_tof(
        self,
        current_q: np.ndarray,
        tof_mm: float,
    ) -> Tuple[np.ndarray, ApproachStatus]:
        """Directly calculate target grasp pose at target ToF distance along line of sight."""
        if tof_mm <= 0.0:
            return current_q.copy(), ApproachStatus.OUT_OF_RANGE

        ee_pos, ee_rot = self.kinematics.forward_kinematics(current_q)
        line_of_sight = ee_rot[:, 2]

        total_disp_m = (tof_mm - self.target_tof_mm) * 0.001
        target_pos = ee_pos + line_of_sight * total_disp_m

        # Enforce table clearance limit
        if target_pos[2] < self.min_ee_z:
            logger.warning("Clamping grasp target to safety Z: %.4fm", self.min_ee_z)
            target_pos[2] = self.min_ee_z

        try:
            grasp_q = self.kinematics.inverse_kinematics(
                target_pos,
                target_z_axis=line_of_sight,
                initial_guess=current_q,
                tolerance=3e-3,
            )
            return grasp_q, ApproachStatus.TARGET_REACHED
        except Exception as e:
            logger.error("Failed to solve grasp pose from ToF: %s", e)
            return current_q.copy(), ApproachStatus.IK_FAILED


class GraspController:
    """Manages gripper actuations, contact verification, and trajectory planning for pick-and-place."""

    def __init__(
        self,
        kinematics_engine: Optional[KinematicsEngine] = None,
        trajectory_planner: Optional[TrajectoryPlanner] = None,
        drop_off_pose: Optional[np.ndarray] = None,
        home_pose: Optional[np.ndarray] = None,
        lift_height_m: float = PRE_TRANSFER_LIFT_M,
    ) -> None:
        self.kinematics = kinematics_engine or KinematicsEngine()
        self.planner = trajectory_planner or TrajectoryPlanner(self.kinematics)
        self.drop_off_pose = drop_off_pose if drop_off_pose is not None else POSE_DROP_OFF.copy()
        self.home_pose = home_pose if home_pose is not None else POSE_HOME.copy()
        self.lift_height_m = lift_height_m

    def verify_grip(self, gripper_joint_pos: float) -> bool:
        """Verify that gripper jaws stopped against an object rather than closing empty."""
        return GRIPPER_CONTACT_MIN_RAD <= gripper_joint_pos <= GRIPPER_CONTACT_MAX_RAD

    def plan_pre_transfer_lift(
        self,
        current_q: np.ndarray,
        duration: float = 0.8,
    ) -> Trajectory:
        """Plan vertical pre-transfer lift by lift_height_m."""
        ee_pos, _ = self.kinematics.forward_kinematics(current_q)
        lift_pos = ee_pos.copy()
        lift_pos[2] = min(lift_pos[2] + self.lift_height_m, 0.28)

        try:
            q_lift = self.kinematics.inverse_kinematics(
                lift_pos,
                initial_guess=current_q,
                tolerance=6e-3,
            )
        except Exception as e:
            logger.warning("Pre-transfer lift IK fallback (%s)", e)
            q_lift = current_q.copy()
            q_lift[1] = max(q_lift[1] - 0.20, -1.2)
            q_lift[2] = min(q_lift[2] + 0.15, 1.2)

        return self.planner.generate_trajectory(
            current_q,
            q_lift,
            duration=duration,
            validate=False,
        )

    def plan_transport(
        self,
        current_q: np.ndarray,
        target_pose: Optional[np.ndarray] = None,
        duration: float = 1.5,
    ) -> Trajectory:
        """Plan smooth trajectory lifting through air with clearance arch to drop-off zone."""
        goal = target_pose if target_pose is not None else self.drop_off_pose
        q0 = np.asarray(current_q, dtype=np.float64)
        q1 = np.asarray(goal, dtype=np.float64)

        traj = self.planner.generate_trajectory(q0, q1, duration=duration, validate=False)
        positions = traj.positions.copy()

        # Add vertical clearance arch: smooth bell curve peaking in mid-transit
        tau = traj.times / traj.duration
        arch = np.sin(np.pi * tau) ** 2
        positions[:, 1] += 0.40 * arch  # lift shoulder
        positions[:, 2] += 0.25 * arch  # lift elbow

        vels = np.gradient(positions, traj.dt, axis=0)
        accs = np.gradient(vels, traj.dt, axis=0)

        return Trajectory(
            times=traj.times,
            positions=positions,
            velocities=vels,
            accelerations=accs,
            duration=traj.duration,
            dt=traj.dt,
        )

    def plan_return_home(
        self,
        current_q: np.ndarray,
        duration: float = 1.2,
    ) -> Trajectory:
        """Plan smooth return trajectory to POSE_HOME through high clearance arch."""
        goal = self.home_pose
        q0 = np.asarray(current_q, dtype=np.float64)
        q1 = np.asarray(goal, dtype=np.float64)

        traj = self.planner.generate_trajectory(q0, q1, duration=duration, validate=False)
        positions = traj.positions.copy()

        tau = traj.times / traj.duration
        arch = np.sin(np.pi * tau) ** 2
        positions[:, 1] += 0.35 * arch
        positions[:, 2] += 0.20 * arch

        vels = np.gradient(positions, traj.dt, axis=0)
        accs = np.gradient(vels, traj.dt, axis=0)

        return Trajectory(
            times=traj.times,
            positions=positions,
            velocities=vels,
            accelerations=accs,
            duration=traj.duration,
            dt=traj.dt,
        )
