import importlib
from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Union, Optional


def _load_optional_module(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


_cv2: Optional[Any] = None
_np: Optional[Any] = None


def _get_cv2() -> Optional[Any]:
    global _cv2
    if _cv2 is None:
        _cv2 = _load_optional_module("cv2")
    return _cv2


def _get_np() -> Optional[Any]:
    global _np
    if _np is None:
        _np = _load_optional_module("numpy")
    return _np


def __getattr__(name: str) -> Any:
    if name == "cv2":
        return _get_cv2()
    if name == "np":
        return _get_np()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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


def encode_jpeg(image: Any, quality: int = 80) -> bytes:
    """Encode OpenCV BGR image to JPEG bytes."""
    cv2_mod = _get_cv2()
    if cv2_mod is None:
        raise ImportError("OpenCV (cv2) is required to encode JPEG frames.")
    success, buffer = cv2_mod.imencode(".jpg", image, [int(cv2_mod.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode frame to JPEG.")
    return buffer.tobytes()


def decode_jpeg(data: bytes) -> Any:
    """Decode JPEG bytes to OpenCV BGR image."""
    cv2_mod = _get_cv2()
    np_mod = _get_np()
    if cv2_mod is None or np_mod is None:
        raise ImportError("OpenCV (cv2) and NumPy are required to decode JPEG frames.")
    arr = np_mod.frombuffer(data, dtype=np_mod.uint8)
    img = cv2_mod.imdecode(arr, cv2_mod.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG bytes to image.")
    return img
