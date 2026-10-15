import threading
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import zenoh

from manipulator3.protocol import (
    TOPIC_CAMERA_RAW,
    TOPIC_TOF_RANGE,
    TOPIC_JOINTS_COMMAND,
    TOPIC_JOINTS_STATE,
    TOPIC_STATUS,
    ToFRange,
    JointsCommand,
    JointsState,
    RobotStatus,
    decode_jpeg,
)


class HostClient:
    """Zenoh client for Laptop Compute Host."""

    def __init__(self, config: Optional[zenoh.Config] = None) -> None:
        self.config = config or zenoh.Config()
        self.session = zenoh.open(self.config)

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_jpeg_time: float = 0.0
        self._frame_count: int = 0
        self._last_fps_calc_time: float = time.time()
        self._fps: float = 0.0

        self._latest_tof: Optional[ToFRange] = None
        self._latest_joints_state: Optional[JointsState] = None
        self._latest_status: Optional[RobotStatus] = None

        # Publishers
        self.pub_joints_cmd = self.session.declare_publisher(TOPIC_JOINTS_COMMAND)
        self.pub_status = self.session.declare_publisher(TOPIC_STATUS)

        # Subscribers
        self.sub_cam = self.session.declare_subscriber(TOPIC_CAMERA_RAW, self._on_camera_sample)
        self.sub_tof = self.session.declare_subscriber(TOPIC_TOF_RANGE, self._on_tof_sample)
        self.sub_state = self.session.declare_subscriber(TOPIC_JOINTS_STATE, self._on_joints_state_sample)
        self.sub_status = self.session.declare_subscriber(TOPIC_STATUS, self._on_status_sample)

    def _on_camera_sample(self, sample: zenoh.Sample) -> None:
        now = time.time()
        payload = sample.payload.to_bytes()
        with self._lock:
            self._latest_jpeg = payload
            self._latest_jpeg_time = now
            self._frame_count += 1
            dt = now - self._last_fps_calc_time
            if dt >= 1.0:
                self._fps = self._frame_count / dt
                self._frame_count = 0
                self._last_fps_calc_time = now

    def _on_tof_sample(self, sample: zenoh.Sample) -> None:
        try:
            tof = ToFRange.from_bytes(sample.payload.to_bytes())
            with self._lock:
                self._latest_tof = tof
        except Exception:
            pass

    def _on_joints_state_sample(self, sample: zenoh.Sample) -> None:
        try:
            state = JointsState.from_bytes(sample.payload.to_bytes())
            with self._lock:
                self._latest_joints_state = state
        except Exception:
            pass

    def _on_status_sample(self, sample: zenoh.Sample) -> None:
        try:
            status = RobotStatus.from_bytes(sample.payload.to_bytes())
            with self._lock:
                self._latest_status = status
        except Exception:
            pass

    def send_joints_command(self, angles: Union[Dict[str, float], List[float]]) -> None:
        """Publish joint target angles to manipulator/joints/command."""
        cmd = JointsCommand(angles=angles, timestamp=time.time())
        self.pub_joints_cmd.put(cmd.to_bytes())

    def send_status(self, status: RobotStatus) -> None:
        """Publish host status to manipulator/status."""
        self.pub_status.put(status.to_bytes())

    def get_camera_frame(self, decode: bool = True) -> Optional[Tuple[Union[np.ndarray, bytes], float]]:
        """Return latest camera frame (BGR np.ndarray or raw JPEG bytes) and timestamp."""
        with self._lock:
            raw = self._latest_jpeg
            ts = self._latest_jpeg_time

        if raw is None:
            return None
        if decode:
            return decode_jpeg(raw), ts
        return raw, ts

    def get_tof_range(self) -> Optional[ToFRange]:
        """Return latest ToFRange sensor reading."""
        with self._lock:
            return self._latest_tof

    def get_joints_state(self) -> Optional[JointsState]:
        """Return latest JointsState."""
        with self._lock:
            return self._latest_joints_state

    def get_status(self) -> Optional[RobotStatus]:
        """Return latest RobotStatus."""
        with self._lock:
            return self._latest_status

    def get_camera_fps(self) -> float:
        """Return received camera FPS."""
        with self._lock:
            return self._fps

    def wait_for_data(self, timeout: float = 3.0) -> bool:
        """Wait until at least one frame or sensor reading is received."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._latest_jpeg is not None or self._latest_tof is not None:
                    return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        """Close Zenoh session and release subscribers."""
        if not self.session.is_closed():
            self.session.close()

    def __enter__(self) -> "HostClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
