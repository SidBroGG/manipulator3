from pathlib import Path
from typing import Tuple, Dict, Optional
import cv2
import numpy as np
import mujoco

from manipulator3.model import (
    SCENE_PATH,
    ALL_JOINTS,
    ARM_JOINTS,
    ACTUATOR_NAMES,
    get_gripper_pose,
    get_tof_reading,
)


class SimulationEnvironment:
    """MuJoCo tabletop simulation environment with camera rendering and ToF sensing."""

    def __init__(
        self,
        scene_path: Optional[Path | str] = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.scene_path = Path(scene_path or SCENE_PATH)
        self.width = width
        self.height = height

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        # Cache joint and actuator indices
        self.actuator_map = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        }
        self.joint_map = {
            name: self.model.joint(name).qposadr[0]
            for name in ALL_JOINTS
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        }

    def step(self, n_steps: int = 1) -> None:
        """Step physics simulation forward."""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def render(self, camera_name: str = "gripper_camera") -> np.ndarray:
        """Render RGB image from specified camera."""
        if camera_name == "gripper_camera":
            scn_opt = mujoco.MjvOption()
            scn_opt.geomgroup[0] = 1
            scn_opt.geomgroup[1] = 1
            scn_opt.geomgroup[2] = 0  # Hide mocap marker spheres from robot camera
            self.renderer.update_scene(self.data, camera=camera_name, scene_option=scn_opt)
        else:
            self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def render_bgr(self, camera_name: str = "gripper_camera") -> np.ndarray:
        """Render camera frame in OpenCV BGR format."""
        rgb = self.render(camera_name)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def get_tof_reading(self) -> float:
        """Return distance in meters (-1.0 if beam is clear or out of range)."""
        return get_tof_reading(self.model, self.data)

    def get_tof_distance_mm(self) -> float:
        """Return distance in millimeters (-1.0 if clear)."""
        reading = self.get_tof_reading()
        return reading * 1000.0 if reading > 0 else -1.0

    def set_actuator_target(self, actuator_name: str, value: float) -> None:
        """Set control target for a single actuator."""
        if actuator_name in self.actuator_map:
            act_id = self.actuator_map[actuator_name]
            ctrlrange = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = float(np.clip(value, ctrlrange[0], ctrlrange[1]))

    def set_actuator_targets(self, targets: Dict[str, float] | np.ndarray) -> None:
        """Set control targets for actuators."""
        if isinstance(targets, dict):
            for name, val in targets.items():
                self.set_actuator_target(name, val)
        else:
            n = min(len(targets), len(self.model.actuator_ctrlrange))
            for i in range(n):
                act_name = self.model.actuator(i).name
                self.set_actuator_target(act_name, targets[i])

    def get_joint_positions(self) -> Dict[str, float]:
        """Return current positions of all tracked joints in radians."""
        return {
            name: float(self.data.qpos[adr])
            for name, adr in self.joint_map.items()
        }

    def get_gripper_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return main__gripperbase position (3,) and rotation matrix (3, 3)."""
        return get_gripper_pose(self.model, self.data)

    def set_mocap_pose(
        self,
        body_name: str,
        pos: np.ndarray,
        quat: Optional[np.ndarray] = None,
    ) -> None:
        """Set position and optional quaternion orientation for a mocap body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            mocap_id = self.model.body_mocapid[body_id]
            if mocap_id >= 0:
                self.data.mocap_pos[mocap_id] = np.asarray(pos, dtype=np.float64)
                if quat is not None:
                    self.data.mocap_quat[mocap_id] = np.asarray(quat, dtype=np.float64)
                mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Reset simulation state to keyframe or initial position."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        """Free renderer resources."""
        if hasattr(self.renderer, "close"):
            self.renderer.close()

    def __enter__(self) -> "SimulationEnvironment":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def draw_sensor_overlay(
    frame: np.ndarray,
    tof_distance_m: float,
    fps: Optional[float] = None,
    camera_name: str = "gripper_camera",
    status_text: Optional[str] = None,
) -> np.ndarray:
    """Draw crosshair reticle, ToF range, and diagnostic HUD on frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2

    # Center reticle
    reticle_color = (0, 255, 255) if tof_distance_m > 0 else (120, 120, 120)
    cv2.circle(canvas, (cx, cy), 14, reticle_color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 2, (0, 0, 255), -1)
    cv2.line(canvas, (cx - 24, cy), (cx - 8, cy), reticle_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx + 8, cy), (cx + 24, cy), reticle_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - 24), (cx, cy - 8), reticle_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy + 8), (cx, cy + 24), reticle_color, 1, cv2.LINE_AA)

    # Top info bar
    cv2.rectangle(canvas, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        f"CAM: {camera_name} ({w}x{h})",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if fps is not None:
        cv2.putText(
            canvas,
            f"FPS: {fps:.1f}",
            (w - 110, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Bottom ToF HUD bar
    cv2.rectangle(canvas, (0, h - 42), (w, h), (20, 20, 20), -1)

    if tof_distance_m > 0:
        tof_mm = tof_distance_m * 1000.0
        # Color transition: Red (<50mm) -> Green (50-250mm) -> Cyan (>250mm)
        if tof_mm < 50.0:
            tof_color = (0, 60, 255)
            tag = "[CLOSE APPROACH]"
        elif tof_mm < 150.0:
            tof_color = (0, 255, 0)
            tag = "[TARGET IN RANGE]"
        else:
            tof_color = (255, 200, 0)
            tag = "[SIGHTLINE HIT]"
        tof_str = f"ToF: {tof_mm:.1f} mm {tag}"
    else:
        tof_color = (160, 160, 160)
        tof_str = "ToF: > 2000 mm (Clear)"

    cv2.putText(
        canvas,
        tof_str,
        (10, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        tof_color,
        2,
        cv2.LINE_AA,
    )

    if status_text:
        cv2.putText(
            canvas,
            status_text,
            (w - 220, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return canvas
