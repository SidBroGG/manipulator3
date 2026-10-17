import argparse
import logging
import threading
import time
from typing import Optional, Union, Dict, List
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
    encode_jpeg,
)
from manipulator3.servo_controller import ServoController, DEFAULT_SERVO_CONFIGS
from manipulator3.rpi_hardware import CameraStreamer, VL53L0XReader

logger = logging.getLogger(__name__)


class RPiThinClient:
    """Thin client running on Raspberry Pi Zero 2 W managing camera, ToF, and servos over Zenoh."""

    def __init__(
        self,
        config: Optional[zenoh.Config] = None,
        mock: bool = False,
        camera_fps: float = 25.0,
        jpeg_quality: int = 70,
        tof_rate_hz: float = 30.0,
        watchdog_timeout: float = 1.5,
        i2c_bus: int = 1,
        pca_address: int = 0x40,
        tof_address: int = 0x29,
    ) -> None:
        self.mock = mock
        self.camera_fps = camera_fps
        self.jpeg_quality = jpeg_quality
        self.tof_rate_hz = tof_rate_hz

        # Zenoh session
        self.config = config or zenoh.Config()
        self.session = zenoh.open(self.config)

        # Hardware modules
        self.servos = ServoController(
            configs=DEFAULT_SERVO_CONFIGS,
            i2c_bus=i2c_bus,
            pca_address=pca_address,
            mock=mock,
            watchdog_timeout_sec=watchdog_timeout,
        )
        self.camera = CameraStreamer(
            fps=camera_fps,
            mock=mock,
        )
        self.tof = VL53L0XReader(
            i2c_bus=i2c_bus,
            i2c_address=tof_address,
            poll_rate_hz=tof_rate_hz,
            mock=mock,
        )

        # Publishers
        self.pub_camera = self.session.declare_publisher(TOPIC_CAMERA_RAW)
        self.pub_tof = self.session.declare_publisher(TOPIC_TOF_RANGE)
        self.pub_state = self.session.declare_publisher(TOPIC_JOINTS_STATE)
        self.pub_status = self.session.declare_publisher(TOPIC_STATUS)

        # Subscriber for host joint commands
        self.sub_cmd = self.session.declare_subscriber(
            TOPIC_JOINTS_COMMAND, self._on_joint_command
        )

        self._running = False
        self._threads: List[threading.Thread] = []
        self._measured_fps: float = 0.0

    def _on_joint_command(self, sample: zenoh.Sample) -> None:
        try:
            cmd = JointsCommand.from_bytes(sample.payload.to_bytes())
            updated_angles = self.servos.apply_angles(cmd.angles)
            # Publish updated joint state
            state_msg = JointsState(positions=updated_angles, timestamp=time.time())
            self.pub_state.put(state_msg.to_bytes())
        except Exception:
            pass

    def _camera_publisher_loop(self) -> None:
        period = 1.0 / max(1.0, self.camera_fps)
        last_frame_time = 0.0
        frame_counter = 0
        fps_timer = time.time()

        while self._running:
            t0 = time.time()
            frame_info = self.camera.get_latest_frame()
            if frame_info is not None:
                img, frame_ts = frame_info
                if frame_ts > last_frame_time:
                    last_frame_time = frame_ts
                    try:
                        jpeg_bytes = encode_jpeg(img, quality=self.jpeg_quality)
                        self.pub_camera.put(jpeg_bytes)
                        frame_counter += 1
                    except Exception:
                        pass

            # Update FPS measurement
            if t0 - fps_timer >= 1.0:
                self._measured_fps = frame_counter / (t0 - fps_timer)
                frame_counter = 0
                fps_timer = t0

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def _tof_publisher_loop(self) -> None:
        period = 1.0 / max(1.0, self.tof_rate_hz)
        last_tof_time = 0.0

        while self._running:
            t0 = time.time()
            dist_mm, valid, ts = self.tof.get_latest_range()
            if ts > last_tof_time:
                last_tof_time = ts
                msg = ToFRange(distance_mm=dist_mm, valid=valid, timestamp=ts)
                self.pub_tof.put(msg.to_bytes())

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def _watchdog_and_status_loop(self) -> None:
        last_status_pub = 0.0
        while self._running:
            t0 = time.time()
            cutoff = self.servos.check_watchdog()
            if t0 - last_status_pub >= 0.5:
                state_str = "WATCHDOG_STOP" if cutoff else "RUNNING"
                status = RobotStatus(
                    state=state_str,
                    connected=True,
                    fps=self._measured_fps,
                    timestamp=t0,
                )
                self.pub_status.put(status.to_bytes())
                last_status_pub = t0
            time.sleep(0.05)

    def start(self) -> None:
        """Start hardware acquisition and Zenoh publishing loops."""
        if self._running:
            return
        self._running = True
        self.camera.start()
        self.tof.start()

        self._threads = [
            threading.Thread(target=self._camera_publisher_loop, daemon=True),
            threading.Thread(target=self._tof_publisher_loop, daemon=True),
            threading.Thread(target=self._watchdog_and_status_loop, daemon=True),
        ]
        for th in self._threads:
            th.start()
        print(f"[INFO] RPi Thin Client started (Mock={self.mock}, FPS={self.camera_fps}).")

    def run(self, duration: float = 0.0) -> None:
        """Run blocking client loop until interrupted or duration expires."""
        self.start()
        start_time = time.time()
        last_log = start_time

        try:
            while self._running:
                now = time.time()
                if duration > 0 and now - start_time >= duration:
                    break

                if now - last_log >= 2.0:
                    dist_mm, valid, _ = self.tof.get_latest_range()
                    tof_str = f"{dist_mm:.1f} mm" if valid else "Out of Range"
                    cutoff_str = "CUTOFF" if self.servos.is_cutoff else "OK"
                    print(
                        f"[INFO] RPi Status | FPS: {self._measured_fps:.1f} | ToF: {tof_str} | Watchdog: {cutoff_str}"
                    )
                    last_log = now

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[INFO] Interrupt received, stopping thin client...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all background workers and release hardware."""
        self._running = False
        for th in self._threads:
            if th.is_alive():
                th.join(timeout=0.5)
        self._threads.clear()

        self.camera.stop()
        self.tof.stop()
        self.servos.close()

        if not self.session.is_closed():
            self.session.close()
        print("[INFO] RPi Thin Client stopped cleanly.")

    def __enter__(self) -> "RPiThinClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="RPi Zero 2 W Thin Client Node")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock sensor and servo mode")
    parser.add_argument("--fps", type=float, default=25.0, help="Camera capture and stream FPS")
    parser.add_argument("--quality", type=int, default=70, help="JPEG compression quality (1-100)")
    parser.add_argument("--tof-rate", type=float, default=30.0, help="ToF range polling rate in Hz")
    parser.add_argument("--watchdog", type=float, default=1.5, help="Watchdog timeout in seconds")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = infinite)")
    parser.add_argument("--i2c-bus", type=int, default=1, help="I2C bus index")
    args = parser.parse_args()

    client = RPiThinClient(
        mock=args.mock,
        camera_fps=args.fps,
        jpeg_quality=args.quality,
        tof_rate_hz=args.tof_rate,
        watchdog_timeout=args.watchdog,
        i2c_bus=args.i2c_bus,
    )
    client.run(duration=args.duration)


if __name__ == "__main__":
    main()
