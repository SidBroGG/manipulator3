import argparse
import multiprocessing
import sys
import time
from typing import Optional
import cv2
import numpy as np

from manipulator3.protocol import TOPIC_CAMERA_RAW, TOPIC_TOF_RANGE, TOPIC_JOINTS_COMMAND
from manipulator3.simulation import draw_sensor_overlay
from manipulator3.zenoh_client import HostClient
from manipulator3.sim_bridge import SimZenohBridge


def _bridge_worker_proc() -> None:
    bridge = SimZenohBridge()
    try:
        bridge.run(fps=30.0)
    finally:
        bridge.close()


def run_zenoh_preview(
    duration: float = 0.0,
    headless: bool = False,
    auto_bridge: bool = False,
) -> None:
    """Run Zenoh host preview receiving camera and ToF data and driving manipulator joints."""
    print("[INFO] Starting Stage 3 Zenoh Preview...")

    bridge_proc: Optional[multiprocessing.Process] = None

    if auto_bridge:
        print("[INFO] Launching embedded simulation bridge process...")
        bridge_proc = multiprocessing.Process(target=_bridge_worker_proc, daemon=True)
        bridge_proc.start()

    client = HostClient()
    start_time = time.time()
    last_log_time = start_time
    cv_window_name = "Manipulator3 - Host Zenoh Feed (OV5647 & VL53L0X)"

    if not headless:
        cv2.namedWindow(cv_window_name, cv2.WINDOW_AUTOSIZE)

    print("[INFO] Waiting for Zenoh sensor stream...")
    if not client.wait_for_data(timeout=5.0):
        print("[WARNING] No data received within 5s. Is sim_bridge running?")

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if duration > 0 and elapsed >= duration:
                break

            # 1. Drive manipulator arm via Zenoh joint commands
            # Sine wave trajectory pitching shoulder and wrist to scan table
            phase = 0.5 * (1.0 + np.sin(elapsed * 1.5))
            cmd_angles = {
                "act_revolute_1": float(0.3 * np.sin(elapsed * 0.8)),   # base yaw
                "act_revolute_2": float(-0.2 - 0.5 * phase),             # shoulder
                "act_revolute_3": float(-0.1 - 0.4 * phase),             # elbow
                "act_revolute_4": float(0.2 + 1.1 * phase),              # wrist pitch
                "act_revolute_5": float(0.0),                            # wrist roll
                "act_gripper": float(0.25 * (1.0 - phase)),              # gripper jaw
            }
            client.send_joints_command(cmd_angles)

            # 2. Receive camera frame
            cam_data = client.get_camera_frame(decode=True)
            tof_reading = client.get_tof_range()
            measured_fps = client.get_camera_fps()

            tof_dist_m = (tof_reading.distance_mm / 1000.0) if (tof_reading and tof_reading.valid) else -1.0

            # 3. Periodic console logging
            if now - last_log_time >= 1.5:
                tof_str = f"{tof_reading.distance_mm:.1f} mm" if (tof_reading and tof_reading.valid) else "> 2000 mm (Clear)"
                print(f"[INFO] Zenoh Host | FPS: {measured_fps:.1f} | ToF: {tof_str}")
                last_log_time = now

            # 4. Display annotated frame in OpenCV
            if not headless and cam_data is not None:
                bgr_frame, _ = cam_data
                annotated = draw_sensor_overlay(
                    bgr_frame,
                    tof_distance_m=tof_dist_m,
                    fps=measured_fps,
                    camera_name="Zenoh: gripper_camera",
                    status_text="Zenoh Link: ACTIVE",
                )
                cv2.imshow(cv_window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:
                    break

            time.sleep(0.01)

    finally:
        if not headless:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        client.close()
        if bridge_proc and bridge_proc.is_alive():
            bridge_proc.terminate()
            bridge_proc.join(timeout=1.0)
        print("[INFO] Zenoh Preview stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 Zenoh Host Preview")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = infinite)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI")
    parser.add_argument("--auto-bridge", action="store_true", help="Automatically run background sim bridge")
    args = parser.parse_args()

    run_zenoh_preview(
        duration=args.duration,
        headless=args.headless,
        auto_bridge=args.auto_bridge,
    )


if __name__ == "__main__":
    main()
