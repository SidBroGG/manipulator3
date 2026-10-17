import argparse
import time
import cv2
import numpy as np

from manipulator3.hardware import (
    PIN_STEP,
    PIN_DIR,
    PIN_SERVO_SHOULDER_A,
    PIN_SERVO_SHOULDER_B,
    PIN_SERVO_ELBOW,
    PIN_SERVO_WRIST_PITCH,
    PIN_SERVO_WRIST_ROLL,
    PIN_SERVO_GRIPPER,
)
from manipulator3.thin_client import ThinClient
from manipulator3.zenoh_client import HostClient


def run_rpi_preview(duration: float = 10.0, mock: bool = True, headless: bool = False) -> None:
    print("[INFO] Starting Stage 8 RPi Zero 2 W Thin Client Preview...")
    print(f"[INFO] Hardware pinout:")
    print(f"       Stepper (NEMA 17 / A4988): STEP={PIN_STEP}, DIR={PIN_DIR}")
    print(f"       Servo 1 (Shoulder A): GPIO {PIN_SERVO_SHOULDER_A}")
    print(f"       Servo 2 (Shoulder B, reversed): GPIO {PIN_SERVO_SHOULDER_B}")
    print(f"       Servo 3 (Elbow): GPIO {PIN_SERVO_ELBOW}")
    print(f"       Servo 4 (Wrist Pitch): GPIO {PIN_SERVO_WRIST_PITCH}")
    print(f"       Servo 5 (Wrist Roll): GPIO {PIN_SERVO_WRIST_ROLL}")
    print(f"       Servo 6 (Gripper): GPIO {PIN_SERVO_GRIPPER}")

    # Launch thin client
    thin_client = ThinClient(
        camera_fps=20,
        tof_rate_hz=20.0,
        force_mock_hardware=mock,
        watchdog_timeout_s=2.0,
    )
    thin_client.start()

    # Launch host client to send commands and inspect feedback
    host = HostClient()
    start_time = time.time()
    last_log = start_time

    cv_window_name = "Manipulator3 - RPi Thin Client Stream"
    if not headless:
        cv2.namedWindow(cv_window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if duration > 0 and elapsed >= duration:
                break

            # Send test sinusoidal movement
            phase = 0.5 * (1.0 + np.sin(elapsed * 2.0))
            test_cmd = [
                0.2 * np.sin(elapsed),          # Base yaw (stepper)
                -0.3 * phase,                   # Shoulder (opposed dual servos)
                -0.5 * phase,                   # Elbow
                0.4 * phase,                    # Wrist pitch
                0.0,                            # Wrist roll
                -0.4 * (1.0 - phase),           # Gripper jaw
            ]
            host.send_joints_command(test_cmd)

            # Receive sensor data
            cam_data = host.get_camera_frame(decode=True)
            tof_data = host.get_tof_range()
            state_data = host.get_joints_state()
            status_data = host.get_status()

            if now - last_log >= 1.5:
                tof_str = f"{tof_data.distance_mm:.1f} mm" if tof_data else "None"
                st_str = status_data.state if status_data else "UNKNOWN"
                q_base = state_data.positions.get("revolute_1", 0.0) if state_data else 0.0
                print(f"[INFO] Thin Client | State: {st_str} | Base Angle: {q_base:.3f} rad | ToF: {tof_str}")
                last_log = now

            if not headless and cam_data is not None:
                bgr_frame, _ = cam_data
                cv2.imshow(cv_window_name, bgr_frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            time.sleep(0.02)

    finally:
        if not headless:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        host.close()
        thin_client.close()
        print("[INFO] RPi Preview ended.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8 RPi Zero 2 W Thin Client Preview")
    parser.add_argument("--duration", type=float, default=5.0, help="Run duration in seconds")
    parser.add_argument("--mock", action="store_true", default=True, help="Force mock hardware")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without OpenCV GUI")
    args = parser.parse_args()

    run_rpi_preview(duration=args.duration, mock=args.mock, headless=args.headless)


if __name__ == "__main__":
    main()
