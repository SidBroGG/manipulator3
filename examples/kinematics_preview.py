import argparse
import sys
import time
import numpy as np
import mujoco
import cv2

from manipulator3.simulation import SimulationEnvironment, draw_sensor_overlay
from manipulator3.kinematics import (
    KinematicsEngine,
    TableCollisionError,
    WorkspaceLimitError,
    POSE_HOME,
    POSE_SCAN,
    POSE_DROP_OFF,
    TABLE_Z_MIN,
)
from manipulator3.trajectory import TrajectoryPlanner


def run_kinematics_preview(
    headless: bool = False,
    loop: bool = False,
    native_viewer: bool = False,
) -> None:
    """Demonstrate Stage 4 IK, smooth trajectory transitions, and safety limits."""
    print("[INFO] Starting Stage 4 Kinematics & Trajectory Preview...")

    sim = SimulationEnvironment()
    engine = KinematicsEngine()
    planner = TrajectoryPlanner(kinematics_engine=engine, max_velocity=1.2, max_acceleration=2.5)

    # 1. Target IK demonstration on tabletop cube
    target_point = np.array([0.18, 0.0, 0.075], dtype=np.float64)
    target_z_axis = np.array([0.25, 0.0, -0.96], dtype=np.float64)
    print(f"[INFO] Solving IK for target point on table: {target_point.round(3)}...")
    q_target = engine.inverse_kinematics(target_point, target_z_axis=target_z_axis)
    pos_reached, _ = engine.forward_kinematics(q_target)
    err_mm = float(np.linalg.norm(target_point - pos_reached) * 1000.0)
    print(f"[INFO] Target acquired via IK at {pos_reached.round(4)}, residual error: {err_mm:.3f} mm")

    # 2. Software safety limit verification
    print("[INFO] Testing software safety intercept for target below table (Z = -0.05m)...")
    try:
        engine.inverse_kinematics([0.18, 0.0, -0.05], enforce_safety=True)
        print("[ERROR] Safety limiter failed to intercept invalid target!")
    except TableCollisionError as e:
        print(f"[INFO] Safety boundary intercept active: {e}")

    # 3. Trajectory waypoints sequence: HOME -> SCAN -> TARGET -> DROP_OFF -> HOME
    waypoints = [
        ("HOME", POSE_HOME),
        ("SCAN", POSE_SCAN),
        ("TARGET", q_target),
        ("DROP_OFF", POSE_DROP_OFF),
        ("HOME", POSE_HOME),
    ]

    cv_window_name = "Manipulator3 - Kinematics & Trajectory Preview"
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

    # Actuator mappings
    arm_acts = [sim.actuator_map[f"act_{j}"] for j in engine.arm_joints]
    gripper_act = sim.actuator_map["act_gripper"]

    try:
        while True:
            for idx in range(len(waypoints) - 1):
                from_name, q_from = waypoints[idx]
                to_name, q_to = waypoints[idx + 1]

                print(f"[INFO] Planning smooth trajectory: {from_name} -> {to_name}...")
                traj = planner.generate_trajectory(q_from, q_to)
                print(f"[INFO] Trajectory generated: {traj.duration:.2f}s, steps: {len(traj.times)}")

                # Close gripper at TARGET and open at DROP_OFF
                grip_cmd = 0.25 if to_name in ("TARGET", "DROP_OFF") and from_name != "DROP_OFF" else 0.0

                start_t = time.time()
                for step_idx in range(len(traj.times)):
                    if viewer_ctx and not viewer_ctx.is_running():
                        return

                    target_q = traj.positions[step_idx]
                    for i, act_id in enumerate(arm_acts):
                        sim.data.ctrl[act_id] = target_q[i]
                    sim.data.ctrl[gripper_act] = grip_cmd

                    sim.step(10)

                    if viewer_ctx:
                        viewer_ctx.sync()

                    if not headless:
                        gripper_frame = sim.render_bgr("gripper_camera")
                        tof_dist = sim.get_tof_reading()
                        status = f"{from_name} -> {to_name} ({step_idx}/{len(traj.times)})"
                        left_panel = draw_sensor_overlay(
                            gripper_frame,
                            tof_distance_m=tof_dist,
                            fps=50.0,
                            camera_name="gripper_camera",
                            status_text=status,
                        )

                        if viewer_ctx is None:
                            overview_frame = sim.render_bgr("overview_camera")
                            cv2.rectangle(overview_frame, (0, 0), (overview_frame.shape[1], 36), (20, 20, 20), -1)
                            cv2.putText(
                                overview_frame,
                                f"MuJoCo 3D Overview | {status}",
                                (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (220, 220, 220),
                                1,
                                cv2.LINE_AA,
                            )
                            display = np.hstack([left_panel, overview_frame])
                        else:
                            display = left_panel

                        cv2.imshow(cv_window_name, display)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            return

                    # Maintain real-time playback
                    elapsed = time.time() - start_t
                    expected = traj.times[step_idx]
                    if expected > elapsed:
                        time.sleep(expected - elapsed)

                time.sleep(0.3)

            if not loop:
                break

    finally:
        if viewer_ctx:
            viewer_ctx.close()
        if not headless:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        sim.close()
        print("[INFO] Kinematics & Trajectory preview finished successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 Kinematics & Trajectory Preview")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    parser.add_argument("--loop", action="store_true", help="Continuously loop through poses")
    parser.add_argument("--native-viewer", action="store_true", help="Launch native GLFW viewer (Linux/Windows only)")
    args = parser.parse_args()

    run_kinematics_preview(headless=args.headless, loop=args.loop, native_viewer=args.native_viewer)


if __name__ == "__main__":
    main()
