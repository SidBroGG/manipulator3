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
    is_pointing_gesture,
)
from manipulator3.simulation import SimulationEnvironment, draw_sensor_overlay
from manipulator3.state_machine import PickAndPlaceStateMachine, ManipulatorState

logger = logging.getLogger(__name__)

# Tabletop pickable objects in simulation scene (World coordinates: X forward, Y left, Z up)
TABLE_OBJECTS = {
    "red_cube": {
        "name": "Red Cube",
        "pos": np.array([0.18, 0.0, 0.015]),
        "color_bgr": (40, 40, 230),
        "key": "1",
    },
    "green_cylinder": {
        "name": "Green Cylinder",
        "pos": np.array([0.16, 0.12, 0.020]),
        "color_bgr": (40, 210, 40),
        "key": "2",
    },
    "blue_box": {
        "name": "Blue Box",
        "pos": np.array([0.15, -0.12, 0.018]),
        "color_bgr": (230, 110, 40),
        "key": "3",
    },
    "yellow_target": {
        "name": "Yellow Target",
        "pos": np.array([0.22, -0.04, 0.015]),
        "color_bgr": (40, 220, 230),
        "key": "4",
    },
}


class OneEuroFilter:
    """Speed-based adaptive low-pass filter (CHI 2012) for smooth, jitter-free hand tracking."""

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        x_val = np.asarray(x, dtype=np.float64)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x_val.copy()
            self.dx_prev = np.zeros_like(x_val)
            self.t_prev = t
            return x_val.copy()

        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        # Filter velocity estimate
        dx = (x_val - self.x_prev) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        edx = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        self.dx_prev = edx

        # Compute adaptive cutoff frequency
        speed = float(np.linalg.norm(edx))
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(dt, cutoff)

        x_filt = alpha * x_val + (1.0 - alpha) * self.x_prev
        self.x_prev = x_filt
        return x_filt

    def _alpha(self, dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / dt)


def table_to_screen_u_v(table_x: float, table_y: float) -> Tuple[float, float]:
    """Map tabletop coordinates (X, Y) to normalized screen position (u, v) in [0, 1]."""
    # X in [0.10, 0.26] -> v in [0.78, 0.22] (forward on table = higher on screen)
    # Y in [-0.20, +0.20] -> u in [0.18, 0.82] (left/right mirrored on screen)
    u = 0.50 + float(table_y) / 0.44
    v = 0.76 - (float(table_x) - 0.10) / 0.25
    return float(np.clip(u, 0.08, 0.92)), float(np.clip(v, 0.12, 0.88))


def screen_u_v_to_table(u: float, v: float) -> Tuple[float, float]:
    """Map normalized screen position (u, v) back to tabletop coordinates (X, Y)."""
    table_y = (u - 0.50) * 0.44
    table_x = 0.10 + (0.76 - v) * 0.25
    return float(np.clip(table_x, 0.08, 0.32)), float(np.clip(table_y, -0.22, 0.22))


class InteractivePointingDemo:
    """Interactive gesture bridge connecting webcam pointing directly to MuJoCo simulation."""

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

        # Camera setup
        self.cap = cv2.VideoCapture(self.camera_id) if self.camera_id >= 0 else None
        self.has_camera = bool(self.cap and self.cap.isOpened())
        if not self.has_camera:
            logger.info("Webcam ID %d unavailable. Operating in mouse/keyboard mode.", self.camera_id)
        else:
            logger.info("Opened physical camera device ID %d", self.camera_id)

        # Filters for rock-solid stability (eliminates hand shake & jitter)
        self.table_filter = OneEuroFilter(min_cutoff=0.6, beta=0.008)
        self.hand_pos_filter = OneEuroFilter(min_cutoff=0.8, beta=0.010)
        self.screen_crosshair_filter = OneEuroFilter(min_cutoff=0.6, beta=0.008)

        # State tracking
        self.curr_hand_pos = np.array([0.30, 0.0, 0.28], dtype=np.float64)
        self.curr_table_target = np.array([0.18, 0.0], dtype=np.float64)
        self.screen_cursor = np.array([0.5, 0.5], dtype=np.float64)

        # Hover & lock
        self.hovered_object: Optional[str] = "red_cube"
        self.hover_start_time: float = time.time()
        self.dwell_time_required: float = 0.85  # 850 ms hover to lock and trigger
        self.is_gesture_active: bool = False

        # Mouse input fallback
        self.mouse_table_pos = np.array([0.18, 0.0], dtype=np.float64)
        self.mouse_active: bool = False

    def on_mouse_event(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse movement and clicks on the simulation overview window."""
        if x >= self.width:
            sim_x = x - self.width
            sim_y = y
            u = sim_x / self.width
            v = sim_y / self.height
            tx, ty = screen_u_v_to_table(u, v)
            self.mouse_table_pos = np.array([tx, ty], dtype=np.float64)
            self.mouse_active = True

            if event == cv2.EVENT_LBUTTONDOWN:
                self.trigger_selection()

    def trigger_selection(self) -> None:
        """Trigger autonomous pick-and-place for currently targeted object."""
        if self.sm.state != ManipulatorState.IDLE:
            logger.warning("Manipulator is busy in state: %s", self.sm.state.value)
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

    def update_hover_and_snap(self, raw_table_pos: np.ndarray) -> np.ndarray:
        """Find closest object, apply magnetic snapping when close, and update dwell timer."""
        min_dist = 999.0
        closest_key = None
        for key, info in TABLE_OBJECTS.items():
            dist = float(np.linalg.norm(raw_table_pos - info["pos"][:2]))
            if dist < min_dist:
                min_dist = dist
                closest_key = key

        result_pos = raw_table_pos.copy()

        # Magnetic Snap: within 55 mm, snap directly to object center for easy targeting
        if min_dist < 0.055 and closest_key is not None:
            obj_target = TABLE_OBJECTS[closest_key]["pos"][:2]
            alpha_snap = float(np.clip(1.0 - (min_dist / 0.055), 0.0, 1.0))
            result_pos = (1.0 - alpha_snap * 0.85) * raw_table_pos + (alpha_snap * 0.85) * obj_target

            if self.hovered_object != closest_key:
                self.hovered_object = closest_key
                self.hover_start_time = time.time()
            else:
                elapsed = time.time() - self.hover_start_time
                if elapsed >= self.dwell_time_required and self.sm.state == ManipulatorState.IDLE and self.is_gesture_active:
                    self.trigger_selection()
        elif min_dist < 0.10 and closest_key is not None:
            self.hovered_object = closest_key
        else:
            self.hovered_object = None
            self.hover_start_time = time.time()

        return result_pos

    def update_hover_and_dwell(self, table_pos: np.ndarray) -> np.ndarray:
        """Update hover state and snap for table coordinates (test compatibility)."""
        return self.update_hover_and_snap(table_pos)

    def draw_augmented_reality_targets(
        self,
        panel: np.ndarray,
        cursor_px: Tuple[int, int],
        is_pointing: bool,
    ) -> None:
        """Render clear on-screen target circles for all 4 objects directly on webcam feed."""
        h, w = panel.shape[:2]

        # Draw header banner with instructions in Russian & English
        cv2.rectangle(panel, (0, 0), (w, 52), (20, 20, 20), -1)
        status_txt = "POINT FINGER AT AN OBJECT TO PICK IT UP / НАВЕДИТЕ ПАЛЕЦ" if self.sm.state == ManipulatorState.IDLE else f"ACTIVE: {self.sm.state.value}"
        status_col = (0, 255, 255) if self.sm.state == ManipulatorState.IDLE else (0, 255, 0)
        cv2.putText(panel, status_txt, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_col, 1)
        cv2.putText(panel, "Hold 0.8s on target or press [SPACE] to grasp", (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Draw each object's target disc on camera view
        for key, info in TABLE_OBJECTS.items():
            tx, ty = info["pos"][:2]
            u, v = table_to_screen_u_v(tx, ty)
            cx, cy = int(u * w), int(v * h)

            is_hover = (key == self.hovered_object)
            col = info["color_bgr"]
            radius = 28 if is_hover else 20

            # Target ring
            cv2.circle(panel, (cx, cy), radius, col, 2 if not is_hover else 3)
            cv2.circle(panel, (cx, cy), 4, col, -1)

            # Label banner
            label = f"[{info['key']}] {info['name']}"
            cv2.putText(panel, label, (cx - 45, cy + radius + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # If hovered, draw circular progress ring for dwell lock
            if is_hover and self.sm.state == ManipulatorState.IDLE and is_pointing:
                dwell_frac = min(1.0, (time.time() - self.hover_start_time) / self.dwell_time_required)
                end_angle = int(dwell_frac * 360)
                cv2.ellipse(panel, (cx, cy), (radius + 6, radius + 6), -90, 0, end_angle, (0, 255, 0), 3)

        # Draw laser cursor at fingertip
        cursor_col = (0, 255, 0) if is_pointing else (0, 160, 255)
        cv2.drawMarker(panel, cursor_px, cursor_col, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(panel, cursor_px, 12, cursor_col, 2)

        # Draw laser ray line from cursor to hovered target
        if self.hovered_object:
            tx, ty = TABLE_OBJECTS[self.hovered_object]["pos"][:2]
            u, v = table_to_screen_u_v(tx, ty)
            tx_px, ty_px = int(u * w), int(v * h)
            cv2.line(panel, cursor_px, (tx_px, ty_px), cursor_col, 1, cv2.LINE_AA)

    def process_camera(self) -> Tuple[np.ndarray, Optional[PointingTarget]]:
        """Read camera frame, track hand with anti-jitter filter, and project to tabletop."""
        t_now = time.time()
        is_pointing = False
        cursor_px = (self.width // 2, self.height // 2)

        if self.has_camera and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Mirror camera horizontally
                frame = cv2.flip(frame, 1)

                landmarks = self.hand_tracker.detect_landmarks(frame)
                annotated = frame.copy()

                if landmarks is not None:
                    is_pointing, _ = self.hand_tracker.is_pointing_gesture(landmarks)
                    self.is_gesture_active = is_pointing

                    # Draw hand skeleton
                    for idx in [0, 5, 6, 7, 8, 9, 12, 16, 20]:
                        pt = landmarks.landmarks[idx]
                        px, py = int(pt[0] * self.width), int(pt[1] * self.height)
                        cv2.circle(annotated, (px, py), 4, (0, 255, 255), -1)
                    # Index finger ray line
                    p_mcp = (int(landmarks.landmarks[5][0] * self.width), int(landmarks.landmarks[5][1] * self.height))
                    p_tip = (int(landmarks.landmarks[8][0] * self.width), int(landmarks.landmarks[8][1] * self.height))
                    cv2.line(annotated, p_mcp, p_tip, (0, 255, 0) if is_pointing else (0, 140, 255), 3)

                    # Fingertip position in screen space
                    tip_u = float(landmarks.landmarks[8][0])
                    tip_v = float(landmarks.landmarks[8][1])

                    # Steer aiming ray along index finger vector
                    dx = landmarks.landmarks[8][0] - landmarks.landmarks[5][0]
                    dy = landmarks.landmarks[8][1] - landmarks.landmarks[5][1]
                    d_norm = float(np.hypot(dx, dy)) + 1e-6
                    aim_u = float(np.clip(tip_u + (dx / d_norm) * 0.08, 0.05, 0.95))
                    aim_v = float(np.clip(tip_v + (dy / d_norm) * 0.08, 0.05, 0.95))

                    # Apply OneEuroFilter for ultra-smooth anti-jitter screen tracking
                    filtered_screen = self.screen_crosshair_filter.filter(np.array([aim_u, aim_v]), t_now)
                    cursor_px = (int(filtered_screen[0] * self.width), int(filtered_screen[1] * self.height))

                    # Convert filtered screen cursor to tabletop workspace coordinates
                    raw_tx, raw_ty = screen_u_v_to_table(filtered_screen[0], filtered_screen[1])

                    # Snap & dwell
                    snapped_target = self.update_hover_and_snap(np.array([raw_tx, raw_ty]))
                    filt_table = self.table_filter.filter(snapped_target, t_now)
                    self.curr_table_target = filt_table

                    # Smooth 3D hand position for MuJoCo visualization
                    hand_world = np.array([filt_table[0] + 0.10, filt_table[1], 0.26])
                    self.curr_hand_pos = self.hand_pos_filter.filter(hand_world, t_now)

                    # Draw on-screen targets and crosshairs
                    self.draw_augmented_reality_targets(annotated, cursor_px, is_pointing)

                    azimuth = float(np.arctan2(self.curr_table_target[1], self.curr_table_target[0]))
                    target = PointingTarget(
                        status=PointingStatus.VALID,
                        point_2d=self.curr_table_target.copy(),
                        azimuth=azimuth,
                        is_reachable=True,
                    )
                    return annotated, target

                # No hand detected in frame
                self.is_gesture_active = False
                self.draw_augmented_reality_targets(annotated, cursor_px, False)
                return annotated, None

        # Desktop fallback canvas (Mouse / Key mode)
        canvas = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        self.is_gesture_active = True

        if self.mouse_active:
            raw_tx, raw_ty = self.mouse_table_pos
            u, v = table_to_screen_u_v(raw_tx, raw_ty)
            cursor_px = (int(u * self.width), int(v * self.height))
            snapped = self.update_hover_and_snap(np.array([raw_tx, raw_ty]))
            self.curr_table_target = self.table_filter.filter(snapped, t_now)
            hand_world = np.array([self.curr_table_target[0] + 0.08, self.curr_table_target[1], 0.25])
            self.curr_hand_pos = self.hand_pos_filter.filter(hand_world, t_now)

        self.draw_augmented_reality_targets(canvas, cursor_px, True)

        azimuth = float(np.arctan2(self.curr_table_target[1], self.curr_table_target[0]))
        target = PointingTarget(
            status=PointingStatus.VALID,
            point_2d=self.curr_table_target.copy(),
            azimuth=azimuth,
            is_reachable=True,
        )
        return canvas, target

    def run(self) -> None:
        """Run interactive loop connecting camera tracking with MuJoCo simulation."""
        window_name = "Manipulator3 - Interactive Hand Pointing & Pick-and-Place"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.on_mouse_event)

        logger.info("Interactive demo window started. Move hand in front of camera or press keys [1-4].")

        prev_time = time.time()
        try:
            while True:
                fps = 1.0 / max(1e-4, time.time() - prev_time)
                prev_time = time.time()

                # 1. Process camera with anti-jitter filter and AR target rings
                cam_panel, pointing_target = self.process_camera()

                # 2. Transfer smoothed hand & laser marker into MuJoCo simulation
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

                # Target lock banner in simulation overview
                if self.hovered_object and self.sm.state == ManipulatorState.IDLE:
                    obj_name = TABLE_OBJECTS[self.hovered_object]["name"]
                    dwell_pct = min(1.0, (time.time() - self.hover_start_time) / self.dwell_time_required)
                    bar_w = int(dwell_pct * 140)
                    cv2.rectangle(sim_overview, (20, 85), (160, 97), (60, 60, 60), -1)
                    cv2.rectangle(sim_overview, (20, 85), (20 + bar_w, 97), (0, 255, 0), -1)
                    cv2.putText(
                        sim_overview,
                        f"AIMED: {obj_name}",
                        (170, 96),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )

                # Combine left (webcam + AR targets) and right (simulation) panels
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
