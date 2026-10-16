import argparse
import logging
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from manipulator3.detector import ColorBlobDetector, draw_detection_overlay
from manipulator3.pointing import (
    HandTracker,
    PointingStatus,
    PointingTarget,
    transform_ray_camera_to_world,
    intersect_ray_with_table,
)
from manipulator3.simulation import SimulationEnvironment, draw_sensor_overlay
from manipulator3.state_machine import PickAndPlaceStateMachine, ManipulatorState

logger = logging.getLogger(__name__)

# Tabletop pickable objects in simulation scene
TABLE_OBJECTS = {
    "red_cube": {
        "name": "Red Cube",
        "pos": np.array([0.18, 0.0, 0.015]),
        "color_bgr": (40, 40, 220),
    },
    "green_cylinder": {
        "name": "Green Cylinder",
        "pos": np.array([0.16, 0.12, 0.020]),
        "color_bgr": (40, 210, 40),
    },
    "blue_box": {
        "name": "Blue Box",
        "pos": np.array([0.15, -0.12, 0.018]),
        "color_bgr": (220, 100, 40),
    },
    "yellow_target": {
        "name": "Yellow Target",
        "pos": np.array([0.22, -0.04, 0.015]),
        "color_bgr": (40, 210, 220),
    },
}

# Default camera-to-world extrinsic for desktop webcam looking down towards user desk
DEFAULT_CAM_POS = np.array([0.35, 0.0, 0.42], dtype=np.float64)
# Camera rotation: pitch downward ~45 deg towards table
DEFAULT_CAM_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.7071, 0.0, -0.7071],
        [-0.7071, 0.0, -0.7071],
    ],
    dtype=np.float64,
)


class InteractivePointingDemo:
    """Synchronizes camera hand movements with MuJoCo simulation for gesture pick-and-place."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.width = width
        self.height = height
        self.camera_id = camera_id

        self.env = SimulationEnvironment(width=width, height=height)
        self.hand_tracker = HandTracker()
        self.sm = PickAndPlaceStateMachine(target_name="red_cube")

        # Camera capture
        self.cap = cv2.VideoCapture(self.camera_id) if self.camera_id >= 0 else None
        self.has_camera = bool(self.cap and self.cap.isOpened())
        if not self.has_camera:
            logger.warning("No physical webcam opened at ID %d. Mouse/keyboard mode active.", self.camera_id)
        else:
            logger.info("Opened physical camera device ID %d", self.camera_id)

        # Smooth filter for hand and table target
        self.curr_hand_pos = np.array([0.35, 0.0, 0.28], dtype=np.float64)
        self.curr_table_target = np.array([0.18, 0.0], dtype=np.float64)

        # Object selection tracking
        self.hovered_object: Optional[str] = "red_cube"
        self.hover_start_time: float = 0.0
        self.dwell_time_required: float = 0.90  # 900 ms hover locks target
        self.locked_object: Optional[str] = None

        # Mouse tracking fallback
        self.mouse_table_pos = np.array([0.18, 0.0], dtype=np.float64)
        self.mouse_active = False

    def on_mouse_event(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse movement over the simulator overview window to point virtually."""
        # Right half of the 1280x480 canvas is the simulation render
        if x >= self.width:
            sim_x = x - self.width
            sim_y = y
            # Linear mapping from simulation overview screen to table workspace [0.08..0.30, -0.20..0.20]
            norm_x = (sim_x / self.width - 0.5) * 2.0
            norm_y = (sim_y / self.height - 0.5) * 2.0
            table_x = 0.20 - norm_y * 0.15
            table_y = norm_x * 0.22
            self.mouse_table_pos = np.array([np.clip(table_x, 0.08, 0.32), np.clip(table_y, -0.22, 0.22)])
            self.mouse_active = True

            if event == cv2.EVENT_LBUTTONDOWN:
                self.trigger_selection()

    def trigger_selection(self) -> None:
        """Trigger autonomous pick-and-place for currently hovered object."""
        if self.sm.state != ManipulatorState.IDLE:
            logger.warning("Manipulator is currently busy in state: %s", self.sm.state.value)
            return

        target_key = self.hovered_object or "red_cube"
        obj_info = TABLE_OBJECTS[target_key]
        target_pos = obj_info["pos"]
        azimuth = float(np.arctan2(target_pos[1], target_pos[0]))

        target = PointingTarget(
            status=PointingStatus.VALID,
            point_2d=target_pos[:2],
            point_3d=target_pos,
            azimuth=azimuth,
            is_reachable=True,
        )

        logger.info("LOCKED ON TARGET: %s at %s (azimuth: %.2f rad)", obj_info["name"], target_pos[:2].round(3), azimuth)
        self.sm.set_target(target_key)
        self.sm.trigger_pointing(target)
        self.locked_object = target_key

    def process_camera(self) -> Tuple[np.ndarray, Optional[PointingTarget]]:
        """Read webcam frame, estimate hand landmarks and pointing ray."""
        if self.has_camera and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Mirror camera horizontally for natural interaction
                frame = cv2.flip(frame, 1)
                target, landmarks = self.hand_tracker.estimate_pointing_target(
                    frame,
                    cam_pos=DEFAULT_CAM_POS,
                    cam_rot=DEFAULT_CAM_ROT,
                )
                annotated = self.hand_tracker.draw_overlay(frame, target, landmarks)

                if landmarks is not None and landmarks.index_tip is not None:
                    # Update smoothed 3D hand position from real skeleton
                    tip_cam = landmarks.world_landmarks[8]
                    hand_world = DEFAULT_CAM_POS + DEFAULT_CAM_ROT @ tip_cam
                    self.curr_hand_pos = 0.75 * self.curr_hand_pos + 0.25 * hand_world

                if target.status == PointingStatus.VALID and target.point_2d is not None:
                    self.curr_table_target = 0.70 * self.curr_table_target + 0.30 * target.point_2d
                    return annotated, target

                return annotated, None

        # Fallback synthetic panel if camera is absent or inactive
        canvas = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        cv2.putText(canvas, "INTERACTIVE HAND POINTING (DESKTOP MODE)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(canvas, "Point with Mouse or Press [1-4] to select object:", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        y_offset = 120
        for idx, (key, info) in enumerate(TABLE_OBJECTS.items()):
            is_hover = (key == self.hovered_object)
            color = info["color_bgr"]
            bg_col = (60, 60, 60) if not is_hover else (100, 100, 40)
            cv2.rectangle(canvas, (20, y_offset), (300, y_offset + 38), bg_col, -1)
            cv2.circle(canvas, (40, y_offset + 19), 10, color, -1)
            cv2.putText(canvas, f"[{idx+1}] {info['name']}", (65, y_offset + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_offset += 48

        cv2.putText(canvas, "Controls:", (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
        cv2.putText(canvas, "- [SPACE] or Left Click: Pick & Place target", (30, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(canvas, "- [R]: Reset scene objects", (30, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(canvas, "- [Q / ESC]: Exit", (30, y_offset + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Virtual pointer target
        if self.mouse_active:
            self.curr_table_target = self.mouse_table_pos.copy()
            self.curr_hand_pos = np.array([self.curr_table_target[0] + 0.08, self.curr_table_target[1], 0.25])

        azimuth = float(np.arctan2(self.curr_table_target[1], self.curr_table_target[0]))
        target = PointingTarget(
            status=PointingStatus.VALID,
            point_2d=self.curr_table_target.copy(),
            azimuth=azimuth,
            is_reachable=True,
        )
        return canvas, target

    def update_hover_and_dwell(self, table_pos: np.ndarray) -> None:
        """Check proximity of pointer to tabletop objects and update dwell lock progress."""
        min_dist = 999.0
        closest_key = None
        for key, info in TABLE_OBJECTS.items():
            dist = float(np.linalg.norm(table_pos - info["pos"][:2]))
            if dist < min_dist:
                min_dist = dist
                closest_key = key

        if min_dist < 0.065 and closest_key is not None:
            if self.hovered_object != closest_key:
                self.hovered_object = closest_key
                self.hover_start_time = time.time()
            else:
                dwell_elapsed = time.time() - self.hover_start_time
                if dwell_elapsed >= self.dwell_time_required and self.sm.state == ManipulatorState.IDLE:
                    self.trigger_selection()
        else:
            self.hovered_object = closest_key if min_dist < 0.12 else None
            self.hover_start_time = time.time()

    def run(self) -> None:
        """Run interactive loop connecting camera tracking with MuJoCo simulation."""
        window_name = "Manipulator3 - Interactive Hand Pointing & Pick-and-Place"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.on_mouse_event)

        logger.info("Interactive demo window started. Move hand in front of camera or use mouse/keys [1-4].")

        prev_time = time.time()
        try:
            while True:
                fps = 1.0 / max(1e-4, time.time() - prev_time)
                prev_time = time.time()

                # 1. Process webcam / hand tracking
                cam_panel, pointing_target = self.process_camera()

                if pointing_target is not None and pointing_target.point_2d is not None:
                    self.update_hover_and_dwell(pointing_target.point_2d)

                # 2. Transfer hand and laser marker into MuJoCo simulation
                self.env.set_mocap_pose("human_hand", self.curr_hand_pos)
                self.env.set_mocap_pose(
                    "pointing_marker",
                    np.array([self.curr_table_target[0], self.curr_table_target[1], 0.002]),
                )

                # 3. Step state machine
                gripper_frame = self.env.render_bgr("gripper_camera")
                tof_mm = self.env.get_tof_distance_mm()
                jpos = self.env.get_joint_positions()
                actual_q = np.array([jpos[f"revolute_{i+1}"] for i in range(5)], dtype=np.float64)
                grip_pos = jpos.get("revolute_7", 0.40)

                output = self.sm.step(
                    frame=gripper_frame,
                    tof_mm=tof_mm,
                    actual_q=actual_q,
                    gripper_joint_pos=grip_pos,
                )

                # Apply controls
                targets = list(output.commanded_q) + [output.commanded_gripper]
                self.env.set_actuator_targets(targets)
                for _ in range(12):
                    self.env.step()

                # 4. Render 3D simulation overview
                sim_overview = self.env.render_bgr("overview_camera")

                # Picture-in-Picture: Gripper camera in top-right corner
                pip_h, pip_w = 120, 160
                pip_frame = cv2.resize(gripper_frame, (pip_w, pip_h))
                cv2.rectangle(pip_frame, (0, 0), (pip_w - 1, pip_h - 1), (0, 255, 0), 2)
                tof_str = f"ToF: {tof_mm:.0f} mm" if tof_mm > 0 else "ToF: Clear"
                cv2.putText(pip_frame, tof_str, (8, pip_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                sim_overview[10 : 10 + pip_h, self.width - pip_w - 10 : self.width - 10] = pip_frame

                # Overlay status in simulation view
                state_color = (0, 255, 0) if output.state != ManipulatorState.IDLE else (200, 200, 200)
                cv2.putText(
                    sim_overview,
                    f"FSM: {output.state.value}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    state_color,
                    2,
                )
                cv2.putText(
                    sim_overview,
                    f"{output.message}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                # Dwell progress indicator
                if self.hovered_object and self.sm.state == ManipulatorState.IDLE:
                    dwell_pct = min(1.0, (time.time() - self.hover_start_time) / self.dwell_time_required)
                    bar_w = int(dwell_pct * 140)
                    cv2.rectangle(sim_overview, (20, 85), (160, 97), (60, 60, 60), -1)
                    cv2.rectangle(sim_overview, (20, 85), (20 + bar_w, 97), (0, 255, 0), -1)
                    cv2.putText(
                        sim_overview,
                        f"LOCK: {TABLE_OBJECTS[self.hovered_object]['name']}",
                        (170, 96),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )

                # Combine left (webcam) and right (simulation) panels into side-by-side display
                combined = np.hstack([cam_panel, sim_overview])
                cv2.imshow(window_name, combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    self.trigger_selection()
                elif key == ord("1"):
                    self.hovered_object = "red_cube"
                    self.curr_table_target = TABLE_OBJECTS["red_cube"]["pos"][:2].copy()
                    self.trigger_selection()
                elif key == ord("2"):
                    self.hovered_object = "green_cylinder"
                    self.curr_table_target = TABLE_OBJECTS["green_cylinder"]["pos"][:2].copy()
                    self.trigger_selection()
                elif key == ord("3"):
                    self.hovered_object = "blue_box"
                    self.curr_table_target = TABLE_OBJECTS["blue_box"]["pos"][:2].copy()
                    self.trigger_selection()
                elif key == ord("4"):
                    self.hovered_object = "yellow_target"
                    self.curr_table_target = TABLE_OBJECTS["yellow_target"]["pos"][:2].copy()
                    self.trigger_selection()
                elif key == ord("r"):
                    self.env.reset()
                    self.sm.reset()
                    logger.info("Simulation scene reset.")
        finally:
            if self.cap:
                self.cap.release()
            self.env.close()
            cv2.destroyAllWindows()
