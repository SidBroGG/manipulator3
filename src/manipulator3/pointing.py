import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import cv2
import numpy as np

from manipulator3.kinematics import REACH_R_MIN, REACH_R_MAX

logger = logging.getLogger(__name__)

MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "hand_landmarker.task"

DEFAULT_SECTOR_SPAN_RAD = float(np.deg2rad(40.0))
DEFAULT_CONFIDENCE_RADIUS = 0.06  # 60 mm


class PointingStatus(str, Enum):
    VALID = "VALID"
    NO_HAND = "NO_HAND"
    NOT_POINTING = "NOT_POINTING"
    PARALLEL_OR_UPWARD = "PARALLEL_OR_UPWARD"
    OUT_OF_TABLE = "OUT_OF_TABLE"
    OUT_OF_REACH = "OUT_OF_REACH"


@dataclass
class TableBounds:
    x_min: float = -0.20
    x_max: float = 0.60
    y_min: float = -0.35
    y_max: float = 0.35
    z_table: float = 0.0


DEFAULT_TABLE_BOUNDS = TableBounds()


@dataclass
class HandLandmarks:
    landmarks: np.ndarray  # (21, 3) normalized screen coords
    world_landmarks: np.ndarray  # (21, 3) metric camera coords in meters
    handedness: str = "Right"
    score: float = 1.0

    @property
    def wrist(self) -> np.ndarray:
        return self.world_landmarks[0]

    @property
    def index_mcp(self) -> np.ndarray:
        return self.world_landmarks[5]

    @property
    def index_pip(self) -> np.ndarray:
        return self.world_landmarks[6]

    @property
    def index_dip(self) -> np.ndarray:
        return self.world_landmarks[7]

    @property
    def index_tip(self) -> np.ndarray:
        return self.world_landmarks[8]

    @property
    def middle_mcp(self) -> np.ndarray:
        return self.world_landmarks[9]

    @property
    def middle_pip(self) -> np.ndarray:
        return self.world_landmarks[10]

    @property
    def middle_tip(self) -> np.ndarray:
        return self.world_landmarks[12]

    @property
    def ring_tip(self) -> np.ndarray:
        return self.world_landmarks[16]

    @property
    def pinky_tip(self) -> np.ndarray:
        return self.world_landmarks[20]


@dataclass
class PointingTarget:
    status: PointingStatus
    point_2d: Optional[np.ndarray] = None
    point_3d: Optional[np.ndarray] = None
    ray_origin: Optional[np.ndarray] = None
    ray_direction: Optional[np.ndarray] = None
    azimuth: Optional[float] = None
    distance: Optional[float] = None
    search_sector: Optional[Tuple[float, float]] = None
    confidence_radius: float = DEFAULT_CONFIDENCE_RADIUS
    is_reachable: bool = False


def ensure_model_asset(model_path: Optional[str | Path] = None) -> Path:
    """Ensure hand landmarker model asset is downloaded and available locally."""
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading hand landmarker model to %s", target_path)
    urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, str(target_path))
    return target_path


def compute_pointing_vector(mcp: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Compute normalized 3D pointing vector from index MCP to index TIP."""
    v = np.asarray(tip, dtype=np.float64) - np.asarray(mcp, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return v / norm


def intersect_ray_with_table(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    table_bounds: TableBounds = DEFAULT_TABLE_BOUNDS,
    r_min: float = REACH_R_MIN,
    r_max: float = REACH_R_MAX,
    sector_span_rad: float = DEFAULT_SECTOR_SPAN_RAD,
    confidence_radius: float = DEFAULT_CONFIDENCE_RADIUS,
) -> PointingTarget:
    """Project a 3D ray onto the tabletop plane Z = table_bounds.z_table and evaluate reachability."""
    origin = np.asarray(ray_origin, dtype=np.float64)
    direction = np.asarray(ray_direction, dtype=np.float64)
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-6:
        return PointingTarget(status=PointingStatus.PARALLEL_OR_UPWARD)
    direction = direction / dir_norm

    # Ray must point downward towards the table plane
    dz = direction[2]
    z_diff = table_bounds.z_table - origin[2]

    # If ray is horizontal or points away from the table plane
    if abs(dz) < 1e-5 or (dz >= 0 and z_diff <= 0) or (dz <= 0 and z_diff >= 0):
        return PointingTarget(
            status=PointingStatus.PARALLEL_OR_UPWARD,
            ray_origin=origin,
            ray_direction=direction,
        )

    t = z_diff / dz
    if t <= 0:
        return PointingTarget(
            status=PointingStatus.PARALLEL_OR_UPWARD,
            ray_origin=origin,
            ray_direction=direction,
        )

    int_x = origin[0] + t * direction[0]
    int_y = origin[1] + t * direction[1]
    int_z = table_bounds.z_table
    p_2d = np.array([int_x, int_y], dtype=np.float64)
    p_3d = np.array([int_x, int_y, int_z], dtype=np.float64)

    # Check table surface boundaries
    in_table = (
        table_bounds.x_min <= int_x <= table_bounds.x_max
        and table_bounds.y_min <= int_y <= table_bounds.y_max
    )
    if not in_table:
        return PointingTarget(
            status=PointingStatus.OUT_OF_TABLE,
            point_2d=p_2d,
            point_3d=p_3d,
            ray_origin=origin,
            ray_direction=direction,
        )

    # Check manipulator reachable workspace
    r = float(np.hypot(int_x, int_y))
    azimuth = float(np.arctan2(int_y, int_x))
    sector = (azimuth - sector_span_rad / 2.0, azimuth + sector_span_rad / 2.0)

    # Robot base is at origin, workspace operates in front of base (x >= -0.05)
    is_reachable = (r_min <= r <= r_max) and (int_x >= -0.05)

    if not is_reachable:
        return PointingTarget(
            status=PointingStatus.OUT_OF_REACH,
            point_2d=p_2d,
            point_3d=p_3d,
            ray_origin=origin,
            ray_direction=direction,
            azimuth=azimuth,
            distance=r,
            search_sector=sector,
            confidence_radius=confidence_radius,
            is_reachable=False,
        )

    return PointingTarget(
        status=PointingStatus.VALID,
        point_2d=p_2d,
        point_3d=p_3d,
        ray_origin=origin,
        ray_direction=direction,
        azimuth=azimuth,
        distance=r,
        search_sector=sector,
        confidence_radius=confidence_radius,
        is_reachable=True,
    )


def transform_ray_camera_to_world(
    origin_cam: np.ndarray,
    direction_cam: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    is_opengl: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform ray from camera coordinates to world coordinates.
    If is_opengl is True, cam_rot is assumed to use OpenGL convention (MuJoCo cam_xmat)
    and is converted to OpenCV convention (+Y down, +Z forward).
    """
    if is_opengl:
        r_cv = np.column_stack([cam_rot[:, 0], -cam_rot[:, 1], -cam_rot[:, 2]])
    else:
        r_cv = np.asarray(cam_rot, dtype=np.float64)

    dir_cam = np.asarray(direction_cam, dtype=np.float64)
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    dir_world = r_cv @ dir_cam
    dir_world = dir_world / np.linalg.norm(dir_world)

    orig_cam = np.asarray(origin_cam, dtype=np.float64)
    orig_world = np.asarray(cam_pos, dtype=np.float64) + r_cv @ orig_cam
    return orig_world, dir_world


def is_pointing_gesture(
    landmarks: HandLandmarks,
    angle_threshold_deg: float = 48.0,
) -> Tuple[bool, float]:
    """
    Determine if hand landmarks represent an intentional index-finger pointing gesture.
    Robustly checks index extension and straightness across both 3D metric and 2D normalized space.
    """
    # 1. Straightness of index finger (MCP -> PIP vs PIP -> TIP)
    v1 = landmarks.index_pip - landmarks.index_mcp
    v2 = landmarks.index_tip - landmarks.index_pip
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-5 or n2 < 1e-5:
        return False, 0.0

    cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    angle_deg = float(np.rad2deg(np.arccos(cos_angle)))
    if angle_deg > angle_threshold_deg:
        return False, 0.0

    # 2. Extension check: index finger compared to middle and pinky fingers (both 3D and 2D)
    d_wrist_index_tip = float(np.linalg.norm(landmarks.index_tip - landmarks.wrist))
    d_wrist_middle_tip = float(np.linalg.norm(landmarks.middle_tip - landmarks.wrist))
    d_wrist_pinky_tip = float(np.linalg.norm(landmarks.pinky_tip - landmarks.wrist))

    # Normalized 2D screen coordinates
    s_idx = float(np.linalg.norm(landmarks.landmarks[8][:2] - landmarks.landmarks[5][:2]))
    s_mid = float(np.linalg.norm(landmarks.landmarks[12][:2] - landmarks.landmarks[9][:2]))
    s_pinky = float(np.linalg.norm(landmarks.landmarks[20][:2] - landmarks.landmarks[17][:2]))

    is_longer_than_middle = (d_wrist_index_tip > d_wrist_middle_tip * 1.04) or (s_idx > s_mid * 1.06)
    is_longer_than_pinky = (d_wrist_index_tip > d_wrist_pinky_tip * 1.08) or (s_idx > s_pinky * 1.10)

    is_pointing = bool(is_longer_than_middle and is_longer_than_pinky)
    confidence = float(np.clip((cos_angle + 1.0) / 2.0, 0.0, 1.0))
    return is_pointing, confidence


class HandTracker:
    """Perception pipeline for hand tracking and pointing ray estimation via MediaPipe."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
        except ImportError as e:
            raise ImportError(
                "mediapipe is not installed. It is only required on the compute host for HandTracker, not on the thin client."
            ) from e

        self._mp = mp
        self.model_path = ensure_model_asset(model_path)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect_landmarks(self, image: np.ndarray) -> Optional[HandLandmarks]:
        """Detect hand landmarks from a BGR or RGB image frame."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR or RGB numpy array.")

        # MediaPipe expects RGB format
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None

        lm_list = result.hand_landmarks[0]
        world_list = (
            result.hand_world_landmarks[0]
            if result.hand_world_landmarks
            else result.hand_landmarks[0]
        )

        norm_coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float64)
        world_coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in world_list], dtype=np.float64
        )

        handedness = "Right"
        score = 1.0
        if result.handedness and len(result.handedness) > 0:
            cat = result.handedness[0][0]
            handedness = cat.category_name
            score = cat.score

        return HandLandmarks(
            landmarks=norm_coords,
            world_landmarks=world_coords,
            handedness=handedness,
            score=score,
        )

    def is_pointing_gesture(
        self,
        landmarks: HandLandmarks,
        angle_threshold_deg: float = 48.0,
    ) -> Tuple[bool, float]:
        """Determine if landmarks represent an intentional index pointing gesture."""
        return is_pointing_gesture(landmarks, angle_threshold_deg=angle_threshold_deg)

    def estimate_pointing_target(
        self,
        image: np.ndarray,
        cam_pos: Optional[np.ndarray] = None,
        cam_rot: Optional[np.ndarray] = None,
        hand_distance: float = 0.45,
        is_opengl: bool = False,
        table_bounds: TableBounds = DEFAULT_TABLE_BOUNDS,
    ) -> Tuple[PointingTarget, Optional[HandLandmarks]]:
        """
        Full pipeline: detect hand, verify gesture, transform ray to world frame,
        and project onto tabletop.
        """
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            return PointingTarget(status=PointingStatus.NO_HAND), None

        pointing_valid, _ = is_pointing_gesture(landmarks)
        if not pointing_valid:
            return PointingTarget(status=PointingStatus.NOT_POINTING), landmarks

        # Pointing vector in camera frame (OpenCV convention: +X right, +Y down, +Z forward)
        dir_cam = compute_pointing_vector(landmarks.index_mcp, landmarks.index_tip)

        # Origin of ray in camera frame
        tip_norm = landmarks.landmarks[8]
        orig_cam = np.array(
            [
                (tip_norm[0] - 0.5) * hand_distance,
                (tip_norm[1] - 0.5) * hand_distance,
                hand_distance,
            ],
            dtype=np.float64,
        )

        if cam_pos is not None and cam_rot is not None:
            orig_world, dir_world = transform_ray_camera_to_world(
                orig_cam, dir_cam, cam_pos, cam_rot, is_opengl=is_opengl
            )
        else:
            orig_world, dir_world = orig_cam, dir_cam

        target = intersect_ray_with_table(
            orig_world,
            dir_world,
            table_bounds=table_bounds,
        )
        return target, landmarks

    def draw_overlay(
        self,
        image: np.ndarray,
        target: PointingTarget,
        landmarks: Optional[HandLandmarks] = None,
    ) -> np.ndarray:
        """Render HUD overlay with pointing line, target coordinates, and status."""
        annotated = image.copy()
        h, w = annotated.shape[:2]

        if landmarks is not None:
            # Draw key joints
            for idx in [0, 5, 6, 7, 8, 9, 12, 16, 20]:
                pt = landmarks.landmarks[idx]
                px, py = int(pt[0] * w), int(pt[1] * h)
                cv2.circle(annotated, (px, py), 4, (0, 255, 255), -1)

            # Draw index finger vector
            mcp = landmarks.landmarks[5]
            tip = landmarks.landmarks[8]
            p_mcp = (int(mcp[0] * w), int(mcp[1] * h))
            p_tip = (int(tip[0] * w), int(tip[1] * h))
            cv2.line(annotated, p_mcp, p_tip, (0, 255, 0), 3)

            # Draw extended pointing ray
            ray_dx = p_tip[0] - p_mcp[0]
            ray_dy = p_tip[1] - p_mcp[1]
            p_ext = (p_tip[0] + int(ray_dx * 2.5), p_tip[1] + int(ray_dy * 2.5))
            cv2.arrowedLine(annotated, p_tip, p_ext, (0, 165, 255), 2, tipLength=0.2)

        # Status text & color
        status_colors = {
            PointingStatus.VALID: (0, 255, 0),
            PointingStatus.NO_HAND: (128, 128, 128),
            PointingStatus.NOT_POINTING: (255, 200, 0),
            PointingStatus.PARALLEL_OR_UPWARD: (0, 165, 255),
            PointingStatus.OUT_OF_TABLE: (0, 0, 255),
            PointingStatus.OUT_OF_REACH: (0, 140, 255),
        }
        color = status_colors.get(target.status, (255, 255, 255))

        # Status banner
        cv2.rectangle(annotated, (10, 10), (360, 110), (25, 25, 25), -1)
        cv2.rectangle(annotated, (10, 10), (360, 110), color, 1)

        cv2.putText(
            annotated,
            f"STATUS: {target.status.value}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        if target.point_2d is not None:
            coord_str = f"Target (X, Y): ({target.point_2d[0]:.3f}, {target.point_2d[1]:.3f}) m"
            cv2.putText(
                annotated,
                coord_str,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )

        if target.azimuth is not None and target.distance is not None:
            reach_str = f"Yaw: {np.rad2deg(target.azimuth):.1f} deg | Dist: {target.distance*1000:.0f} mm"
            cv2.putText(
                annotated,
                reach_str,
                (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )

        cv2.putText(
            annotated,
            f"Reachable: {target.is_reachable}",
            (20, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if target.is_reachable else (0, 0, 255),
            1,
        )

        return annotated
