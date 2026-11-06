import argparse
import time
from typing import Optional
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


def draw_hud(
    frame: np.ndarray,
    tof_mm: Optional[float],
    state_str: str,
    joint_angles: dict,
    fps: float,
) -> np.ndarray:
    """Draw telemetry HUD on camera frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2

    # Reticle
    reticle_color = (0, 255, 255) if (tof_mm is not None and tof_mm > 0) else (100, 100, 100)
    cv2.circle(canvas, (cx, cy), 16, reticle_color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 2, (0, 0, 255), -1)
    cv2.line(canvas, (cx - 24, cy), (cx - 8, cy), reticle_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx + 8, cy), (cx + 24, cy), reticle_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - 24), (cx, cy - 8), reticle_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy + 8), (cx, cy + 24), reticle_color, 1, cv2.LINE_AA)

    # Top overlay header
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Status badge
    badge_color = (0, 200, 0) if state_str == "READY" else (0, 0, 220)
    cv2.rectangle(canvas, (10, 10), (120, 40), badge_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, state_str, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # ToF distance text
    tof_text = f"ToF: {tof_mm:.1f} mm" if (tof_mm is not None and tof_mm > 0) else "ToF: Out of Range"
    cv2.putText(canvas, tof_text, (135, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # FPS
    cv2.putText(canvas, f"FPS: {fps:.1f}", (w - 110, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Joint positions telemetry at bottom
    q1 = joint_angles.get("revolute_1", 0.0)
    q2 = joint_angles.get("revolute_2", 0.0)
    q3 = joint_angles.get("revolute_3", 0.0)
    q4 = joint_angles.get("revolute_4", 0.0)
    q5 = joint_angles.get("revolute_5", 0.0)
    q7 = joint_angles.get("revolute_7", 0.0)

    line1 = f"Base: {q1:+.2f} rad | Shldr A/B (rev): {q2:+.2f} rad | Elbow: {q3:+.2f} rad"
    line2 = f"Wrist Pitch: {q4:+.2f} rad | Roll: {q5:+.2f} rad | Gripper: {q7:+.2f} rad"
    cv2.putText(canvas, line1, (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(canvas, line2, (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 180), 1)

    return canvas


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
    measured_fps = 20.0

    cv_window_name = "Manipulator3 - RPi Thin Client Stream (OV5647 & VL53L0X)"
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
                0.25 * np.sin(elapsed),         # Base yaw (stepper)
                -0.35 * phase,                  # Shoulder (opposed dual servos)
                -0.50 * phase,                  # Elbow
                0.45 * phase,                   # Wrist pitch
                0.0,                            # Wrist roll
                -0.40 * (1.0 - phase),          # Gripper jaw
            ]
            host.send_joints_command(test_cmd)

            # Receive sensor data
            cam_data = host.get_camera_frame(decode=True)
            tof_data = host.get_tof_range()
            state_data = host.get_joints_state()
            status_data = host.get_status()
            fps = host.get_camera_fps()
            if fps > 0:
                measured_fps = fps

            tof_dist_mm = tof_data.distance_mm if (tof_data and tof_data.valid) else None
            st_str = status_data.state if status_data else "READY"
            joint_positions = state_data.positions if state_data else {}

            if now - last_log >= 1.5:
                tof_str = f"{tof_dist_mm:.1f} mm" if tof_dist_mm is not None else "None"
                q_base = joint_positions.get("revolute_1", 0.0)
                print(f"[INFO] Thin Client | State: {st_str} | Base Angle: {q_base:.3f} rad | ToF: {tof_str}")
                last_log = now

            if not headless and cam_data is not None:
                bgr_frame, _ = cam_data
                annotated = draw_hud(bgr_frame, tof_dist_mm, st_str, joint_positions, measured_fps)
                cv2.imshow(cv_window_name, annotated)
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
    parser.add_argument("--duration", type=float, default=10.0, help="Run duration in seconds (0 = infinite)")
    parser.add_argument("--mock", action="store_true", default=True, help="Force mock hardware")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI")
    args = parser.parse_args()

    run_rpi_preview(duration=args.duration, mock=args.mock, headless=args.headless)


if __name__ == "__main__":
    main()
