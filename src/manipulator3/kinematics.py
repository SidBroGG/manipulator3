import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import mujoco
import ikpy.chain
from manipulator3.model import (
    load_mujoco_scene,
    load_ikpy_chain,
    ARM_JOINTS,
    URDF_PATH,
    MJCF_PATH,
)

logger = logging.getLogger(__name__)

# Software safety boundaries (meters)
TABLE_Z_MIN = 0.015  # 15 mm above table plane (Z=0)
REACH_R_MIN = 0.08   # Minimum radial reach from base
REACH_R_MAX = 0.32   # Maximum radial reach from base
BASE_HEIGHT_WORLD = 0.066536

# Baseline joint poses for 5 arm joints (radians)
POSE_HOME = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
POSE_SCAN = np.array([0.0, -0.1364, -1.2642, 0.9000, 0.0], dtype=np.float64)
POSE_DROP_OFF = np.array([-1.6814, -0.6843, -1.4092, 0.8270, 0.0], dtype=np.float64)

BASELINE_POSES: Dict[str, np.ndarray] = {
    "HOME": POSE_HOME,
    "SCAN": POSE_SCAN,
    "DROP_OFF": POSE_DROP_OFF,
}


class SafetyLimitError(ValueError):
    """Base exception for software safety boundary violations."""
    pass


class TableCollisionError(SafetyLimitError):
    """Raised when target position penetrates or breaches table safety margin."""
    pass


class WorkspaceLimitError(SafetyLimitError):
    """Raised when target position is outside reachable workspace."""
    pass


class KinematicsConvergenceError(RuntimeError):
    """Raised when inverse kinematics solver fails to converge."""
    pass


def world_to_base_pos(pos_world: np.ndarray) -> np.ndarray:
    """Transform 3D position from world frame to base_link frame."""
    p = np.asarray(pos_world, dtype=np.float64)
    return np.array([-p[0], -p[1], p[2] - BASE_HEIGHT_WORLD], dtype=np.float64)


def base_to_world_pos(pos_base: np.ndarray) -> np.ndarray:
    """Transform 3D position from base_link frame to world frame."""
    p = np.asarray(pos_base, dtype=np.float64)
    return np.array([-p[0], -p[1], p[2] + BASE_HEIGHT_WORLD], dtype=np.float64)


class KinematicsEngine:
    """Hybrid IKPy and MuJoCo kinematics engine for 5-DOF arm."""

    def __init__(self, mjcf_path: Optional[str] = None, urdf_path: Optional[str] = None):
        # MuJoCo model & data for forward kinematics and Jacobian calculations
        self.model, self.data = load_mujoco_scene(mjcf_path)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "main__gripperbase")
        if self.body_id < 0:
            raise ValueError("Body 'main__gripperbase' not found in model.")

        self.arm_joints = ARM_JOINTS
        self.arm_dofs = [self.model.joint(j).dofadr[0] for j in self.arm_joints]
        self.q_adrs = [self.model.joint(j).qposadr[0] for j in self.arm_joints]
        self.joint_limits = np.array([self.model.jnt_range[self.model.joint(j).id] for j in self.arm_joints])

        # IKPy chain (5 arm links up to main__gripperbase)
        base_chain = load_ikpy_chain(urdf_path)
        self.ikpy_chain = ikpy.chain.Chain(
            links=base_chain.links[:6],
            active_links_mask=[False, True, True, True, True, True],
        )

    def check_safety(self, pos: np.ndarray, clamp: bool = False) -> np.ndarray:
        """Enforce table collision boundary and workspace reach limits."""
        pos = np.asarray(pos, dtype=np.float64).copy()
        r = float(np.hypot(pos[0], pos[1]))

        if pos[2] < TABLE_Z_MIN:
            if clamp:
                pos[2] = TABLE_Z_MIN
            else:
                raise TableCollisionError(
                    f"Target Z ({pos[2]:.4f}m) penetrates table safety boundary ({TABLE_Z_MIN:.4f}m)"
                )

        if not (REACH_R_MIN <= r <= REACH_R_MAX):
            if clamp:
                scale = np.clip(r, REACH_R_MIN, REACH_R_MAX) / (r + 1e-9)
                pos[0] *= scale
                pos[1] *= scale
            else:
                raise WorkspaceLimitError(
                    f"Target radial distance {r:.4f}m outside reachable bounds [{REACH_R_MIN}, {REACH_R_MAX}]"
                )

        return pos

    def forward_kinematics(self, joint_angles: np.ndarray | list[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector position and 3x3 orientation in world coordinates."""
        q = np.asarray(joint_angles, dtype=np.float64)
        for i, adr in enumerate(self.q_adrs):
            self.data.qpos[adr] = q[i]
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.xpos[self.body_id].copy()
        mat = self.data.xmat[self.body_id].reshape(3, 3).copy()
        return pos, mat

    def inverse_kinematics(
        self,
        target_pos: np.ndarray | list[float],
        target_z_axis: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        max_iter: int = 80,
        tolerance: float = 1e-3,
        enforce_safety: bool = True,
    ) -> np.ndarray:
        """
        Solve inverse kinematics for target 3D world position with optional gripper approach orientation.
        Combines IKPy initial global optimization with MuJoCo Jacobian DLS refinement.
        """
        target = np.asarray(target_pos, dtype=np.float64)
        if enforce_safety:
            target = self.check_safety(target, clamp=False)

        # Base yaw heuristic: revolute_1 rotates around -Z axis
        heuristic_yaw = -float(np.arctan2(target[1], target[0]))
        if initial_guess is None:
            q_cur = np.array([heuristic_yaw, -0.4, -1.0, 0.8, 0.0], dtype=np.float64)
        else:
            q_cur = np.asarray(initial_guess, dtype=np.float64).copy()

        # Step 1: Fast IKPy estimation
        target_base = world_to_base_pos(target)
        try:
            q_ikpy_init = [0.0] + list(q_cur)
            q_ikpy_sol = self.ikpy_chain.inverse_kinematics(
                target_base,
                initial_position=q_ikpy_init,
                optimizer_budget=35,
            )
            q_cand = np.array(q_ikpy_sol[1:], dtype=np.float64)
            pos_cand, _ = self.forward_kinematics(q_cand)
            if np.linalg.norm(target - pos_cand) < tolerance and target_z_axis is None:
                return q_cand
            q_cur = q_cand
        except Exception:
            pass

        # Step 2: MuJoCo Damped Least Squares (DLS) Jacobian refinement
        for i, adr in enumerate(self.q_adrs):
            self.data.qpos[adr] = np.clip(q_cur[i], self.joint_limits[i, 0], self.joint_limits[i, 1])
        mujoco.mj_forward(self.model, self.data)

        has_orientation = target_z_axis is not None
        if has_orientation:
            target_z = np.asarray(target_z_axis, dtype=np.float64)
            target_z = target_z / np.linalg.norm(target_z)

        for step in range(max_iter):
            cur_pos = self.data.xpos[self.body_id].copy()
            err_p = target - cur_pos

            if not has_orientation:
                if np.linalg.norm(err_p) < tolerance:
                    return np.array([self.data.qpos[adr] for adr in self.q_adrs], dtype=np.float64)
            else:
                cur_mat = self.data.xmat[self.body_id].reshape(3, 3)
                cur_z = cur_mat[:, 2]
                err_r = np.cross(cur_z, target_z)
                if np.linalg.norm(err_p) < tolerance and np.linalg.norm(err_r) < 0.02:
                    return np.array([self.data.qpos[adr] for adr in self.q_adrs], dtype=np.float64)

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv)) if has_orientation else None
            mujoco.mj_jac(self.model, self.data, jacp, jacr, cur_pos, self.body_id)

            J_p = jacp[:, self.arm_dofs]
            if not has_orientation:
                J = J_p
                err = err_p
                diag = 1e-4 * np.eye(3)
            else:
                w_rot = 0.25
                J_r = jacr[:, self.arm_dofs]
                J = np.vstack([J_p, w_rot * J_r])
                err = np.concatenate([err_p, w_rot * err_r])
                diag = 1e-4 * np.eye(6)

            dq = J.T @ np.linalg.solve(J @ J.T + diag, err)

            step_limit = 0.2
            max_step = float(np.max(np.abs(dq)))
            if max_step > step_limit:
                dq = dq * (step_limit / max_step)

            for i, adr in enumerate(self.q_adrs):
                self.data.qpos[adr] = np.clip(
                    self.data.qpos[adr] + dq[i],
                    self.joint_limits[i, 0],
                    self.joint_limits[i, 1],
                )
            mujoco.mj_forward(self.model, self.data)

        final_pos = self.data.xpos[self.body_id].copy()
        final_err = float(np.linalg.norm(target - final_pos))
        if final_err > tolerance * 2.0:
            raise KinematicsConvergenceError(
                f"IK failed to converge to target {target.round(4)}. Error: {final_err*1000:.2f} mm"
            )

        return np.array([self.data.qpos[adr] for adr in self.q_adrs], dtype=np.float64)
