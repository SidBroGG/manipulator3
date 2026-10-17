import argparse
import sys
import time
import cv2
import numpy as np
import mujoco

from manipulator3.simulation import SimulationEnvironment, draw_sensor_overlay


def run_preview(
    duration: float = 0.0,
    headless: bool = False,
    initial_camera: str = "gripper_camera",
    native_viewer: bool = False,
) -> None:
    """Run interactive 3D MuJoCo simulation with OpenCV camera preview."""
    print("[INFO] Starting Stage 2 simulation preview...")
    sim = SimulationEnvironment()

    cam_list = ["gripper_camera", "overview_camera", "top_camera"]
    current_cam_idx = cam_list.index(initial_camera) if initial_camera in cam_list else 0

    # Actuator references
    act_s = sim.actuator_map.get("act_revolute_2", 1)  # shoulder
    act_e = sim.actuator_map.get("act_revolute_3", 2)  # elbow
    act_w = sim.actuator_map.get("act_revolute_4", 3)  # wrist pitch
    act_g = sim.actuator_map.get("act_gripper", 5)     # gripper jaw

    auto_approach = True
    start_time = time.time()
    last_frame_time = start_time
    last_log_time = 0.0
    frame_count = 0
    fps = 0.0

    cv_window_name = "Manipulator3 - Simulation Preview (OV5647 & 3D Overview)"
    if not headless:
        cv2.namedWindow(cv_window_name, cv2.WINDOW_AUTOSIZE)

    # Disable native GLFW viewer on macOS to prevent Cocoa main-thread conflicts with OpenCV
    viewer_ctx = None
    if native_viewer and not headless:
        if sys.platform == "darwin":
            print("[INFO] macOS detected: Using unified OpenCV viewer (native GLFW viewer disabled for Cocoa compatibility).")
        else:
            try:
                import mujoco.viewer
                viewer_ctx = mujoco.viewer.launch_passive(sim.model, sim.data)
            except Exception as e:
                print(f"[WARNING] Native viewer could not be launched: {e}. Using OpenCV viewer.")

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            if duration > 0 and elapsed >= duration:
                break

            if viewer_ctx and not viewer_ctx.is_running():
                break

            # Approach trajectory towards red cube on table
            if auto_approach:
                # Oscillate between elevated pose and near-table approach pose
                phase = 0.5 * (1.0 + np.sin(elapsed * 1.2))  # [0, 1]
                # Pitch shoulder, elbow, wrist towards target cube (camera on top)
                sim.data.ctrl[act_s] = -0.2 - 0.65 * phase
                sim.data.ctrl[act_e] = -0.1 - 0.45 * phase
                sim.data.ctrl[act_w] = 0.2 + 1.25 * phase
                sim.data.ctrl[act_g] = 0.25 * (1.0 - phase)

            # Physics stepping
            sim.step(10)

            if viewer_ctx:
                viewer_ctx.sync()

            # Render camera frame
            cam_name = cam_list[current_cam_idx]
            bgr_frame = sim.render_bgr(camera_name=cam_name)
            tof_dist = sim.get_tof_reading()

            # Measure FPS
            frame_count += 1
            if now - last_frame_time >= 0.5:
                fps = frame_count / (now - last_frame_time)
                frame_count = 0
                last_frame_time = now

            # Periodic console logging (every 1.5s)
            if now - last_log_time > 1.5:
                tof_str = f"{tof_dist * 1000.0:.1f} mm" if tof_dist > 0 else "> 2000 mm (Clear)"
                print(f"[INFO] Cam: {cam_name} | ToF reading: {tof_str} | FPS: {fps:.1f}")
                last_log_time = now

            if not headless:
                mode_str = "Auto Approach: ON ('a')" if auto_approach else "Manual Mode ('a')"
                annotated = draw_sensor_overlay(
                    bgr_frame,
                    tof_distance_m=tof_dist,
                    fps=fps,
                    camera_name=cam_name,
                    status_text=mode_str,
                )

                if viewer_ctx is None:
                    # Multi-panel composite for macOS / standard mode
                    overview_frame = sim.render_bgr("overview_camera")
                    cv2.rectangle(overview_frame, (0, 0), (overview_frame.shape[1], 36), (20, 20, 20), -1)
                    cv2.putText(
                        overview_frame,
                        "MuJoCo 3D Tabletop Overview",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (220, 220, 220),
                        1,
                        cv2.LINE_AA,
                    )
                    display = np.hstack([annotated, overview_frame])
                else:
                    display = annotated

                cv2.imshow(cv_window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:  # 'q' or ESC
                    break
                elif key == ord("a"):
                    auto_approach = not auto_approach
                    print(f"[INFO] Auto approach toggled: {auto_approach}")
                elif key == ord("c"):
                    current_cam_idx = (current_cam_idx + 1) % len(cam_list)
                    print(f"[INFO] Switched camera to: {cam_list[current_cam_idx]}")
                elif key == ord("r"):
                    sim.reset()
                    print("[INFO] Simulation reset.")

    finally:
        if viewer_ctx:
            viewer_ctx.close()
        if not headless:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        sim.close()
        print("[INFO] Simulation preview finished successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 MuJoCo Simulation and Sensor Preview")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = infinite)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    parser.add_argument("--camera", type=str, default="gripper_camera", help="Initial camera name")
    parser.add_argument("--native-viewer", action="store_true", help="Launch native GLFW viewer (Linux/Windows only)")
    args = parser.parse_args()

    run_preview(
        duration=args.duration,
        headless=args.headless,
        initial_camera=args.camera,
        native_viewer=args.native_viewer,
    )


if __name__ == "__main__":
    main()
