import argparse
import logging
import signal
import threading
import time
from typing import Dict, List, Optional, Union
import zenoh

from manipulator3.hardware import (
    GPIOBackend,
    MockGPIOBackend,
    get_default_gpio_backend,
    StepperDriverA4988,
    ServoController,
    ToFReaderVL53L0X,
    CameraStreamerOV5647,
)
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
    encode_jpeg,
)

logger = logging.getLogger(__name__)


class ThinClient:
    """Zenoh Thin Client for Raspberry Pi Zero 2 W controlling arm hardware."""

    def __init__(
        self,
        config: Optional[zenoh.Config] = None,
        gpio: Optional[GPIOBackend] = None,
        force_mock_hardware: bool = False,
        camera_fps: int = 20,
        tof_rate_hz: float = 30.0,
        jpeg_quality: int = 75,
        watchdog_timeout_s: float = 2.0,
    ) -> None:
        self.camera_fps = camera_fps
        self.tof_rate_hz = tof_rate_hz
        self.jpeg_quality = jpeg_quality
        self.watchdog_timeout_s = watchdog_timeout_s
        self.force_mock = force_mock_hardware

        # Hardware drivers
        self.gpio = gpio or get_default_gpio_backend(force_mock=self.force_mock)
        self.stepper = StepperDriverA4988(self.gpio)
        self.servos = ServoController(self.gpio)
        self.tof = ToFReaderVL53L0X(mock=self.force_mock)
        self.camera = CameraStreamerOV5647(fps=self.camera_fps, mock=self.force_mock)

        # Zenoh session & endpoints
        self.config = config or zenoh.Config()
        self.session = zenoh.open(self.config)

        self.pub_camera = self.session.declare_publisher(TOPIC_CAMERA_RAW)
        self.pub_tof = self.session.declare_publisher(TOPIC_TOF_RANGE)
        self.pub_state = self.session.declare_publisher(TOPIC_JOINTS_STATE)
        self.pub_status = self.session.declare_publisher(TOPIC_STATUS)

        self.sub_cmd = self.session.declare_subscriber(
            TOPIC_JOINTS_COMMAND, self._on_joint_command
        )

        # State tracking
        self._lock = threading.Lock()
        self._running = False
        self._emergency_stop = False
        self._last_cmd_time = time.time()
        self._threads: List[threading.Thread] = []

    def _on_joint_command(self, sample: zenoh.Sample) -> None:
        try:
            cmd = JointsCommand.from_bytes(sample.payload.to_bytes())
            self.apply_command(cmd.angles)
        except Exception:
            pass

    def apply_command(self, angles: Union[Dict[str, float], List[float]]) -> None:
        """Apply joint angles to stepper base and servo joints."""
        with self._lock:
            self._last_cmd_time = time.time()
            if self._emergency_stop:
                logger.info("[INFO] Command received. Clearing emergency stop.")
                self._emergency_stop = False

        # Base yaw via stepper
        if isinstance(angles, (list, tuple)) and len(angles) > 0:
            self.stepper.set_target_angle(float(angles[0]))
        elif isinstance(angles, dict):
            if "revolute_1" in angles:
                self.stepper.set_target_angle(float(angles["revolute_1"]))
            elif "base" in angles:
                self.stepper.set_target_angle(float(angles["base"]))

        # Arm & gripper joints via servos
        self.servos.set_joint_angles(angles)

    def trigger_emergency_stop(self) -> None:
        """Execute emergency cutoff: halt stepper and cut servo PWM drive."""
        with self._lock:
            self._emergency_stop = True

        self.stepper.emergency_stop()
        self.servos.emergency_stop()
        status = RobotStatus(state="EMERGENCY_STOP", connected=True, timestamp=time.time())
        self.pub_status.put(status.to_bytes())
        logger.warning("[WARNING] Emergency stop triggered: Watchdog timeout.")

    def get_joint_positions(self) -> Dict[str, float]:
        """Return current estimated joint positions in radians."""
        positions = {
            "revolute_1": self.stepper.current_angle_rad,
            "revolute_2": self.servos.current_angles.get("revolute_2", 0.0),
            "revolute_3": self.servos.current_angles.get("revolute_3", 0.0),
            "revolute_4": self.servos.current_angles.get("revolute_4", 0.0),
            "revolute_5": self.servos.current_angles.get("revolute_5", 0.0),
            "revolute_6": -self.servos.current_angles.get("revolute_7", 0.0),
            "revolute_7": self.servos.current_angles.get("revolute_7", 0.0),
        }
        return positions

    def _camera_worker(self) -> None:
        period = 1.0 / self.camera_fps
        while self._running:
            t0 = time.time()
            try:
                frame = self.camera.capture_frame()
                jpeg = encode_jpeg(frame, quality=self.jpeg_quality)
                self.pub_camera.put(jpeg)
            except Exception:
                pass

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def _tof_worker(self) -> None:
        period = 1.0 / self.tof_rate_hz
        while self._running:
            t0 = time.time()
            try:
                dist_mm, valid = self.tof.read_distance_mm()
                msg = ToFRange(distance_mm=dist_mm, valid=valid, timestamp=time.time())
                self.pub_tof.put(msg.to_bytes())
            except Exception:
                pass

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def _stepper_worker(self) -> None:
        while self._running:
            with self._lock:
                is_estop = self._emergency_stop

            if not is_estop:
                stepped = self.stepper.step_once()
                if not stepped:
                    time.sleep(0.005)
                else:
                    time.sleep(0.0005)
            else:
                time.sleep(0.02)

    def _watchdog_and_state_worker(self) -> None:
        period = 0.05  # 20 Hz
        while self._running:
            now = time.time()

            # Watchdog check
            with self._lock:
                last_time = self._last_cmd_time
                is_estop = self._emergency_stop

            if not is_estop and (now - last_time > self.watchdog_timeout_s):
                self.trigger_emergency_stop()

            # State & Status publishing
            positions = self.get_joint_positions()
            state_msg = JointsState(positions=positions, velocities=None, timestamp=now)
            self.pub_state.put(state_msg.to_bytes())

            current_state = "EMERGENCY_STOP" if self._emergency_stop else "READY"
            status = RobotStatus(state=current_state, connected=True, fps=self.camera_fps, timestamp=now)
            self.pub_status.put(status.to_bytes())

            time.sleep(period)

    def start(self) -> None:
        """Start hardware client background threads."""
        if self._running:
            return
        self._running = True
        self._last_cmd_time = time.time()

        self._threads = [
            threading.Thread(target=self._camera_worker, daemon=True, name="rpi-camera"),
            threading.Thread(target=self._tof_worker, daemon=True, name="rpi-tof"),
            threading.Thread(target=self._stepper_worker, daemon=True, name="rpi-stepper"),
            threading.Thread(target=self._watchdog_and_state_worker, daemon=True, name="rpi-watchdog"),
        ]

        for t in self._threads:
            t.start()

        logger.info("[INFO] ThinClient started.")

    def stop(self) -> None:
        """Stop background threads and release hardware."""
        self._running = False
        for t in self._threads:
            t.join(timeout=0.5)
        self._threads.clear()

        self.stepper.emergency_stop()
        self.servos.emergency_stop()
        self.camera.close()
        self.tof.close()
        self.gpio.cleanup()
        logger.info("[INFO] ThinClient stopped.")

    def close(self) -> None:
        self.stop()
        if not self.session.is_closed():
            self.session.close()

    def __enter__(self) -> "ThinClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="RPi Zero 2 W Thin Client Node")
    parser.add_argument("--fps", type=int, default=20, help="Camera frame rate")
    parser.add_argument("--quality", type=int, default=75, help="JPEG quality")
    parser.add_argument("--watchdog", type=float, default=2.0, help="Watchdog timeout in seconds")
    parser.add_argument("--mock", action="store_true", help="Force mock hardware backend")
    args = parser.parse_args()

    client = ThinClient(
        camera_fps=args.fps,
        jpeg_quality=args.quality,
        watchdog_timeout_s=args.watchdog,
        force_mock_hardware=args.mock,
    )

    def handle_signal(sig, frame):
        client.stop()
        client.close()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    client.start()
    try:
        while client._running:
            time.sleep(1.0)
    finally:
        client.close()


if __name__ == "__main__":
    main()
