import argparse
import sys
import time
from typing import Optional
import cv2
import numpy as np

from manipulator3.zenoh_client import HostClient
from manipulator3.rpi_client import RPiThinClient
from manipulator3.simulation import draw_sensor_overlay


def run_rpi_preview(
    duration: float = 0.0,
    headless: bool = False,
    mock: bool = True,
) -> None:
    """Run interactive preview testing RPi Thin Client node communicating with HostClient."""
    print(f"[INFO] Starting Stage 8 RPi Client Preview (mock={mock})...")

    # Start thin client
    rpi_node = RPiThinClient(mock=mock, camera_fps=25.0, jpeg_quality=75)
    rpi_node.start()

    # Start host client
    host = HostClient()
    start_time = time.time()
    last_log_time = start_time
    cv_window_name = "Manipulator3 - RPi Zero 2 W Client Preview"

    if not headless:
        cv2.namedWindow(cv_window_name, cv2.WINDOW_AUTOSIZE)

    print("[INFO] Waiting for RPi client data stream...")
    if not host.wait_for_data(timeout=5.0):
        print("[WARNING] No data received within 5s from RPi client.")

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if duration > 0 and elapsed >= duration:
                break

            # 1. Host commands joints with smooth test motion
            test_angles = {
                "revolute_1": float(0.4 * np.sin(elapsed * 1.0)),
                "revolute_2": float(-0.3 + 0.3 * np.cos(elapsed * 0.8)),
                "revolute_3": float(-0.2 + 0.2 * np.sin(elapsed * 0.9)),
                "revolute_4": float(0.1 * np.sin(elapsed * 1.2)),
                "revolute_5": float(0.0),
                "act_gripper": float(0.2 * np.sin(elapsed * 1.5)),
            }
            host.send_joints_command(test_angles)

            # 2. Receive camera frame and ToF data
            cam_data = host.get_camera_frame(decode=True)
            tof_reading = host.get_tof_range()
            measured_fps = host.get_camera_fps()
            joint_state = host.get_joints_state()

            # 3. Periodic console logging
            if now - last_log_time >= 1.5:
                tof_str = (
                    f"{tof_reading.distance_mm:.1f} mm"
                    if (tof_reading and tof_reading.valid)
                    else "Out of range"
                )
                j1_pos = (
                    f"{joint_state.positions.get('revolute_1', 0.0):.2f} rad"
                    if joint_state
                    else "N/A"
                )
                print(f"[INFO] Host Link | FPS: {measured_fps:.1f} | ToF: {tof_str} | J1: {j1_pos}")
                last_log_time = now

            # 4. GUI preview
            if not headless and cam_data is not None:
                bgr_frame, _ = cam_data
                tof_m = (tof_reading.distance_mm / 1000.0) if (tof_reading and tof_reading.valid) else -1.0
                annotated = draw_sensor_overlay(
                    bgr_frame,
                    tof_distance_m=tof_m,
                    fps=measured_fps,
                    camera_name="RPi OV5647 Stream",
                    status_text="RPi Client: ONLINE",
                )
                cv2.imshow(cv_window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:
                    break

            time.sleep(0.02)

    finally:
        if not headless:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        host.close()
        rpi_node.stop()
        print("[INFO] RPi Client Preview stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8 RPi Thin Client Preview")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = infinite)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display")
    parser.add_argument("--hardware", action="store_true", help="Try real hardware instead of mock")
    args = parser.parse_args()

    run_rpi_preview(
        duration=args.duration,
        headless=args.headless,
        mock=not args.hardware,
    )


if __name__ == "__main__":
    main()
