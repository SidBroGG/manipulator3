from dataclasses import dataclass, field
import json
import time
from typing import Dict, List, Union, Optional
import cv2
import numpy as np

TOPIC_CAMERA_RAW = "manipulator/camera/raw"
TOPIC_TOF_RANGE = "manipulator/tof/range"
TOPIC_JOINTS_COMMAND = "manipulator/joints/command"
TOPIC_JOINTS_STATE = "manipulator/joints/state"
TOPIC_STATUS = "manipulator/status"


@dataclass
class ToFRange:
    distance_mm: float
    valid: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "distance_mm": float(self.distance_mm),
            "valid": bool(self.valid),
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToFRange":
        return cls(
            distance_mm=float(data["distance_mm"]),
            valid=bool(data["valid"]),
            timestamp=float(data.get("timestamp", 0.0)),
        )

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ToFRange":
        return cls.from_dict(json.loads(data.decode("utf-8")))


@dataclass
class JointsCommand:
    angles: Union[Dict[str, float], List[float]]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "angles": self.angles,
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JointsCommand":
        return cls(
            angles=data["angles"],
            timestamp=float(data.get("timestamp", 0.0)),
        )

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "JointsCommand":
        return cls.from_dict(json.loads(data.decode("utf-8")))


@dataclass
class JointsState:
    positions: Dict[str, float]
    velocities: Optional[Dict[str, float]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "positions": self.positions,
            "velocities": self.velocities,
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JointsState":
        return cls(
            positions=data["positions"],
            velocities=data.get("velocities"),
            timestamp=float(data.get("timestamp", 0.0)),
        )

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "JointsState":
        return cls.from_dict(json.loads(data.decode("utf-8")))


@dataclass
class RobotStatus:
    state: str = "IDLE"
    connected: bool = True
    fps: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "connected": bool(self.connected),
            "fps": float(self.fps),
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RobotStatus":
        return cls(
            state=data.get("state", "IDLE"),
            connected=bool(data.get("connected", True)),
            fps=float(data.get("fps", 0.0)),
            timestamp=float(data.get("timestamp", 0.0)),
        )

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "RobotStatus":
        return cls.from_dict(json.loads(data.decode("utf-8")))


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    """Encode OpenCV BGR image to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode frame to JPEG.")
    return buffer.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to OpenCV BGR image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG bytes to image.")
    return img
