from dataclasses import dataclass
from typing import Optional, List, Union
import numpy as np
from manipulator3.kinematics import (
    KinematicsEngine,
    BASELINE_POSES,
    TABLE_Z_MIN,
    TableCollisionError,
    SafetyLimitError,
)

DEFAULT_MAX_VELOCITY = 1.5      # rad/s per joint
DEFAULT_MAX_ACCELERATION = 3.0  # rad/s^2 per joint
DEFAULT_DT = 0.02               # 50 Hz control rate


@dataclass
class Trajectory:
    """Discrete trajectory for 5-DOF arm with time, positions, velocities, and accelerations."""
    times: np.ndarray          # (N,)
    positions: np.ndarray      # (N, 5)
    velocities: np.ndarray     # (N, 5)
    accelerations: np.ndarray  # (N, 5)
    duration: float
    dt: float

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate position, velocity, and acceleration at arbitrary time t."""
        t_clamped = float(np.clip(t, 0.0, self.duration))
        idx = int(t_clamped / self.dt)
        if idx >= len(self.times) - 1:
            return self.positions[-1].copy(), self.velocities[-1].copy(), self.accelerations[-1].copy()

        t0, t1 = self.times[idx], self.times[idx + 1]
        alpha = (t_clamped - t0) / (t1 - t0 + 1e-9)
        pos = (1 - alpha) * self.positions[idx] + alpha * self.positions[idx + 1]
        vel = (1 - alpha) * self.velocities[idx] + alpha * self.velocities[idx + 1]
        acc = (1 - alpha) * self.accelerations[idx] + alpha * self.accelerations[idx + 1]
        return pos, vel, acc


class TrajectoryPlanner:
    """Minimum-jerk polynomial joint trajectory generator with safety validation."""

    def __init__(
        self,
        kinematics_engine: Optional[KinematicsEngine] = None,
        max_velocity: float = DEFAULT_MAX_VELOCITY,
        max_acceleration: float = DEFAULT_MAX_ACCELERATION,
        dt: float = DEFAULT_DT,
    ):
        self.kinematics = kinematics_engine or KinematicsEngine()
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.dt = float(dt)

    def compute_duration(self, q_start: np.ndarray, q_goal: np.ndarray) -> float:
        """Calculate minimum time needed without exceeding joint velocity or acceleration limits."""
        delta_q = np.abs(np.asarray(q_goal, dtype=np.float64) - np.asarray(q_start, dtype=np.float64))
        # For quintic minimum jerk: peak velocity = 1.875 * delta_q / T, peak acc = 5.77 * delta_q / T^2
        t_vel = np.max(1.875 * delta_q / self.max_velocity) if np.any(delta_q > 1e-6) else 0.2
        t_acc = np.max(np.sqrt(5.77 * delta_q / self.max_acceleration)) if np.any(delta_q > 1e-6) else 0.2
        return float(max(t_vel, t_acc, 0.3))

    def generate_trajectory(
        self,
        q_start: np.ndarray | list[float],
        q_goal: np.ndarray | list[float],
        duration: Optional[float] = None,
        validate: bool = True,
    ) -> Trajectory:
        """Generate minimum-jerk trajectory from q_start to q_goal."""
        q0 = np.asarray(q_start, dtype=np.float64)
        q1 = np.asarray(q_goal, dtype=np.float64)

        min_T = self.compute_duration(q0, q1)
        T = float(max(duration, min_T) if duration is not None else min_T)
        steps = max(2, int(np.ceil(T / self.dt)) + 1)
        times = np.linspace(0.0, T, steps)

        tau = times / T
        # Quintic polynomial: s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        s_dot = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / T
        s_ddot = (60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)) / (T ** 2)

        delta = (q1 - q0)[:, np.newaxis]
        positions = q0[:, np.newaxis] + delta * s[np.newaxis, :]
        velocities = delta * s_dot[np.newaxis, :]
        accelerations = delta * s_ddot[np.newaxis, :]

        traj = Trajectory(
            times=times,
            positions=positions.T,
            velocities=velocities.T,
            accelerations=accelerations.T,
            duration=T,
            dt=T / (steps - 1),
        )

        if validate:
            self.validate_trajectory(traj, check_table_safety=True)

        return traj

    def validate_trajectory(self, trajectory: Trajectory, check_table_safety: bool = True) -> bool:
        """Validate joint limits, velocity bounds, and table penetration along trajectory."""
        limits = self.kinematics.joint_limits

        # Check joint angle limits
        for i in range(5):
            if np.any(trajectory.positions[:, i] < limits[i, 0] - 1e-4) or np.any(
                trajectory.positions[:, i] > limits[i, 1] + 1e-4
            ):
                raise SafetyLimitError(f"Joint {self.kinematics.arm_joints[i]} exceeds limits in trajectory")

        # Check velocity bounds
        max_v = float(np.max(np.abs(trajectory.velocities)))
        if max_v > self.max_velocity + 0.05:
            raise SafetyLimitError(f"Peak trajectory velocity {max_v:.2f} rad/s exceeds limit {self.max_velocity:.2f}")

        # Check table surface collision margin for intermediate points
        if check_table_safety:
            step_skip = max(1, len(trajectory.times) // 25)
            for pos_q in trajectory.positions[::step_skip]:
                pos_ee, _ = self.kinematics.forward_kinematics(pos_q)
                if pos_ee[2] < TABLE_Z_MIN - 1e-4:
                    raise TableCollisionError(
                        f"Trajectory penetrates table safety boundary at Z = {pos_ee[2]:.4f}m (min: {TABLE_Z_MIN:.4f}m)"
                    )

        return True

    def plan_to_pose(
        self,
        q_current: np.ndarray | list[float],
        pose: Union[str, np.ndarray, list[float]],
        duration: Optional[float] = None,
    ) -> Trajectory:
        """Plan safe trajectory from current state to a target baseline pose or angles."""
        if isinstance(pose, str):
            pose_upper = pose.upper()
            if pose_upper not in BASELINE_POSES:
                raise ValueError(f"Unknown baseline pose: {pose}. Available: {list(BASELINE_POSES.keys())}")
            q_goal = BASELINE_POSES[pose_upper]
        else:
            q_goal = np.asarray(pose, dtype=np.float64)

        return self.generate_trajectory(q_current, q_goal, duration=duration, validate=True)
