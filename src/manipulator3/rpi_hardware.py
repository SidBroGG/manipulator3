import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class MockCamera:
    """Synthetic test camera generating animated test patterns."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._start_time = time.time()

    def read(self) -> Tuple[bool, np.ndarray]:
        t = time.time() - self._start_time
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        y = np.linspace(30, 80, self.height, dtype=np.uint8)[:, None]
        img[:, :, 0] = y
        img[:, :, 1] = y
        img[:, :, 2] = y

        # Animated circle simulating object in FOV
        cx = int(self.width / 2 + 100 * np.sin(t * 1.5))
        cy = int(self.height / 2 + 60 * np.cos(t * 1.5))
        cv2.circle(img, (cx, cy), 28, (0, 0, 220), -1)
        cv2.putText(
            img,
            f"Mock OV5647 | T={t:.1f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return True, img

    def release(self) -> None:
        pass


class CameraStreamer:
    """OV5647 camera streamer supporting Picamera2, OpenCV VideoCapture, and Mock."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        mock: bool = False,
        device_id: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.mock = mock
        self.device_id = device_id

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_time: float = 0.0

        self._picam2 = None
        self._cap = None
        self._mock_cam = None

    def _init_source(self) -> None:
        if self.mock:
            self._mock_cam = MockCamera(self.width, self.height)
            return

        # 1. Try Picamera2 (RPi hardware)
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            picam2.configure(config)
            picam2.start()
            self._picam2 = picam2
            return
        except Exception:
            pass

        # 2. Try OpenCV VideoCapture
        try:
            cap = cv2.VideoCapture(self.device_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self._cap = cap
                    return
                cap.release()
        except Exception:
            pass

        # 3. Fallback to mock
        self._mock_cam = MockCamera(self.width, self.height)
        self.mock = True

    def _capture_loop(self) -> None:
        period = 1.0 / max(1.0, self.fps)
        while self._running:
            t0 = time.time()
            frame = None

            if self._picam2 is not None:
                try:
                    frame = self._picam2.capture_array()
                except Exception:
                    pass
            elif self._cap is not None:
                ret, img = self._cap.read()
                if ret:
                    frame = img
            elif self._mock_cam is not None:
                _, frame = self._mock_cam.read()

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_time = time.time()

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def start(self) -> None:
        if self._running:
            return
        self._init_source()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy(), self._latest_time

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            self._picam2 = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._mock_cam is not None:
            self._mock_cam.release()
            self._mock_cam = None


class VL53L0XReader:
    """VL53L0X Time-of-Flight sensor reader over I2C."""

    def __init__(
        self,
        i2c_bus: int = 1,
        i2c_address: int = 0x29,
        poll_rate_hz: float = 30.0,
        mock: bool = False,
    ) -> None:
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.poll_rate_hz = poll_rate_hz
        self.mock = mock

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_dist_mm: float = -1.0
        self._latest_valid: bool = False
        self._latest_time: float = 0.0

        self._sensor = None

    def _init_sensor(self) -> None:
        if self.mock:
            return

        # 1. Try PyVL53L0X
        try:
            import VL53L0X
            tof = VL53L0X.VL53L0X(i2c_bus=self.i2c_bus, i2c_address=self.i2c_address)
            tof.open()
            tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
            self._sensor = tof
            return
        except Exception:
            pass

        # 2. Try Adafruit CircuitPython VL53L0X
        try:
            import board
            import busio
            import adafruit_vl53l0x
            i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = adafruit_vl53l0x.VL53L0X(i2c, address=self.i2c_address)
            return
        except Exception:
            pass

        # Fallback to mock
        self.mock = True

    def set_mock_distance(self, distance_mm: float, valid: bool = True) -> None:
        """Set simulated distance for testing."""
        with self._lock:
            self._latest_dist_mm = float(distance_mm)
            self._latest_valid = valid
            self._latest_time = time.time()

    def _poll_loop(self) -> None:
        period = 1.0 / max(1.0, self.poll_rate_hz)
        start_time = time.time()

        while self._running:
            t0 = time.time()
            dist_mm = -1.0
            valid = False

            if self.mock:
                # Oscillating reading for simulation testing if not explicitly set
                with self._lock:
                    if self._latest_dist_mm >= 0.0:
                        dist_mm = self._latest_dist_mm
                        valid = self._latest_valid
                    else:
                        elapsed = t0 - start_time
                        dist_mm = 50.0 + 30.0 * np.sin(elapsed * 2.0)
                        valid = True
            elif self._sensor is not None:
                try:
                    if hasattr(self._sensor, "get_distance"):
                        d = self._sensor.get_distance()
                        if 20 <= d <= 2000:
                            dist_mm = float(d)
                            valid = True
                    elif hasattr(self._sensor, "range"):
                        d = self._sensor.range
                        if 20 <= d <= 2000:
                            dist_mm = float(d)
                            valid = True
                except Exception:
                    pass

            with self._lock:
                self._latest_dist_mm = dist_mm
                self._latest_valid = valid
                self._latest_time = time.time()

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def start(self) -> None:
        if self._running:
            return
        self._init_sensor()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def get_latest_range(self) -> Tuple[float, bool, float]:
        with self._lock:
            return self._latest_dist_mm, self._latest_valid, self._latest_time

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._sensor is not None:
            try:
                if hasattr(self._sensor, "stop_ranging"):
                    self._sensor.stop_ranging()
                if hasattr(self._sensor, "close"):
                    self._sensor.close()
            except Exception:
                pass
            self._sensor = None
