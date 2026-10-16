import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "models"
DEFAULT_YOLO_PATH = DEFAULT_MODEL_DIR / "yolo11n.pt"

# Common pickable tabletop objects from COCO dataset
DEFAULT_PICKABLE_CLASSES = {
    "bottle", "cup", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "box", "book", "clock", "vase", "scissors", "teddy bear",
    "cell phone", "mouse", "remote", "keyboard", "toothbrush",
}


@dataclass
class DetectedObject:
    box_xyxy: Tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    center_xy: Tuple[float, float]
    norm_center: Tuple[float, float]  # (u_norm, v_norm) in [-1, 1], (0, 0) is frame center
    width: float
    height: float
    area: float
    aspect_ratio: float


def compute_detected_object(
    box: Tuple[float, float, float, float],
    class_id: int,
    class_name: str,
    confidence: float,
    frame_width: int,
    frame_height: int,
) -> DetectedObject:
    x1, y1, x2, y2 = box
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    cx = float(x1 + w / 2.0)
    cy = float(y1 + h / 2.0)

    # Normalized error: -1 (left/top) to +1 (right/bottom), (0, 0) is center
    u_norm = float(2.0 * (cx / frame_width) - 1.0)
    v_norm = float(2.0 * (cy / frame_height) - 1.0)
    aspect_ratio = float(w / h) if h > 1e-6 else 1.0

    return DetectedObject(
        box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        class_id=int(class_id),
        class_name=str(class_name),
        confidence=float(confidence),
        center_xy=(cx, cy),
        norm_center=(u_norm, v_norm),
        width=w,
        height=h,
        area=w * h,
        aspect_ratio=aspect_ratio,
    )


class YOLOObjectDetector:
    """YOLO11n object detector for robotic camera feed."""

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[Iterable[str]] = None,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = set(target_classes) if target_classes is not None else None
        self.device = device

        resolved_path = self._resolve_model_path(model_path)
        from ultralytics import YOLO
        logger.info("Loading YOLO11n weights from %s", resolved_path)
        self.model = YOLO(str(resolved_path))

    @staticmethod
    def _resolve_model_path(model_path: Optional[Union[Path, str]]) -> Path:
        if model_path is not None:
            p = Path(model_path)
            if p.exists():
                return p
        if DEFAULT_YOLO_PATH.exists():
            return DEFAULT_YOLO_PATH
        local_cwd = Path("yolo11n.pt")
        if local_cwd.exists():
            return local_cwd
        return Path("yolo11n.pt")

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        exclude_border_px: int = 15,
    ) -> List[DetectedObject]:
        """Detect objects in BGR or RGB frame."""
        h, w = frame.shape[:2]
        thresh = conf if conf is not None else self.conf_threshold

        results = self.model(
            frame,
            conf=thresh,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectedObject] = []
        if not results:
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        names = res.names
        for box_item in res.boxes:
            c_id = int(box_item.cls[0].item())
            c_name = names.get(c_id, str(c_id))
            confidence = float(box_item.conf[0].item())

            if self.target_classes is not None and c_name not in self.target_classes:
                continue

            xyxy = box_item.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy

            # Skip edge border artifacts (e.g. gripper fingers at extreme edges)
            if exclude_border_px > 0:
                if x1 < exclude_border_px and x2 < exclude_border_px * 2:
                    continue
                if x2 > (w - exclude_border_px) and x1 > (w - exclude_border_px * 2):
                    continue

            det = compute_detected_object(
                box=(x1, y1, x2, y2),
                class_id=c_id,
                class_name=c_name,
                confidence=confidence,
                frame_width=w,
                frame_height=h,
            )
            detections.append(det)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def get_best_target(
        self,
        detections: List[DetectedObject],
        strategy: str = "confidence",
    ) -> Optional[DetectedObject]:
        """Select best candidate target according to strategy ('confidence', 'center', 'area')."""
        if not detections:
            return None
        if strategy == "center":
            return min(detections, key=lambda d: d.norm_center[0] ** 2 + d.norm_center[1] ** 2)
        if strategy == "area":
            return max(detections, key=lambda d: d.area)
        return detections[0]

    def benchmark_inference(
        self,
        iterations: int = 15,
        img_size: Tuple[int, int] = (640, 480),
    ) -> Dict[str, float]:
        """Benchmark detector inference latency on dummy image."""
        w, h = img_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)

        # Warmup
        self.model(dummy, verbose=False, device=self.device)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.model(dummy, verbose=False, device=self.device)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        latencies = np.array(latencies)
        mean_ms = float(np.mean(latencies))
        fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

        logger.info("YOLO11n benchmark: %.2f ms/frame (%.1f FPS)", mean_ms, fps)
        return {
            "mean_ms": mean_ms,
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "std_ms": float(np.std(latencies)),
            "fps": fps,
        }


class ColorBlobDetector:
    """Fast color blob detector for simulated objects or distinct colored markers."""

    # HSV color ranges for tabletop simulation objects
    COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
        "red_cube": [
            ((0, 100, 50), (10, 255, 255)),
            ((170, 100, 50), (180, 255, 255)),
        ],
        "green_cylinder": [
            ((35, 80, 50), (85, 255, 255)),
        ],
        "blue_box": [
            ((95, 80, 50), (135, 255, 255)),
        ],
        "yellow_target": [
            ((18, 100, 50), (35, 255, 255)),
        ],
    }

    def __init__(
        self,
        target_colors: Optional[Iterable[str]] = None,
        min_area: float = 120.0,
    ) -> None:
        self.target_colors = set(target_colors) if target_colors else set(self.COLOR_RANGES.keys())
        self.min_area = min_area

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedObject]:
        """Detect colored objects on table via HSV color segmentation."""
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        detections: List[DetectedObject] = []

        class_id = 0
        for color_name, ranges in self.COLOR_RANGES.items():
            if color_name not in self.target_colors:
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                sub_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.bitwise_or(mask, sub_mask)

            # Morphological noise cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                # Compute confidence based on fill ratio of bounding box
                fill_ratio = float(area / (bw * bh + 1e-6))
                confidence = float(np.clip(fill_ratio * 0.95 + 0.05, 0.1, 0.99))

                det = compute_detected_object(
                    box=(x, y, x + bw, y + bh),
                    class_id=class_id,
                    class_name=color_name,
                    confidence=confidence,
                    frame_width=w,
                    frame_height=h,
                )
                detections.append(det)

            class_id += 1

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections


def draw_detection_overlay(
    frame: np.ndarray,
    detections: List[DetectedObject],
    best_target: Optional[DetectedObject] = None,
) -> np.ndarray:
    """Render bounding boxes, class labels, reticles, and normalized crosshairs."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    cx_frame, cy_frame = w // 2, h // 2

    # Draw central crosshair
    cv2.line(canvas, (cx_frame - 20, cy_frame), (cx_frame + 20, cy_frame), (0, 255, 255), 1)
    cv2.line(canvas, (cx_frame, cy_frame - 20), (cx_frame, cy_frame + 20), (0, 255, 255), 1)

    for det in detections:
        is_best = (best_target is not None and det is best_target)
        color = (0, 255, 0) if is_best else (220, 180, 40)
        thick = 2 if is_best else 1

        x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)

        # Center dot
        dcx, dcy = int(det.center_xy[0]), int(det.center_xy[1])
        cv2.circle(canvas, (dcx, dcy), 4, color, -1)

        # Text label
        label = f"{det.class_name} {det.confidence * 100:.0f}%"
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        # Vector line from frame center to target center
        if is_best:
            cv2.line(canvas, (cx_frame, cy_frame), (dcx, dcy), (0, 220, 255), 1, cv2.LINE_AA)
            err_label = f"err: u={det.norm_center[0]:+.2f}, v={det.norm_center[1]:+.2f}"
            cv2.putText(
                canvas,
                err_label,
                (dcx + 8, dcy + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas
