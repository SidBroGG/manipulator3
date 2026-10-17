import logging
import time
import cv2
import numpy as np

from manipulator3.detector import ColorBlobDetector, draw_detection_overlay
from manipulator3.simulation import SimulationEnvironment, draw_sensor_overlay
from manipulator3.state_machine import PickAndPlaceStateMachine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("statemachine_preview")


def main() -> None:
    logger.info("Starting Autonomous Pick-and-Place State Machine Preview")
    env = SimulationEnvironment()
    sm = PickAndPlaceStateMachine(target_name="red_cube")

    # Trigger pointing at center table sector
    sm.trigger_pointing(azimuth_rad=0.0)

    window_name = "Manipulator3 - Stage 7 Pick & Place Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    curr_grip = 0.40
    try:
        for step_idx in range(600):
            t0 = time.time()
            frame = env.render_bgr("gripper_camera")
            overview_frame = env.render_bgr("overview_camera")
            tof_mm = env.get_tof_distance_mm()
            jpos = env.get_joint_positions()
            actual_q = np.array([jpos[f"revolute_{i+1}"] for i in range(5)], dtype=np.float64)
            grip_pos = jpos.get("revolute_7", 0.4)

            output = sm.step(
                frame=frame,
                tof_mm=tof_mm,
                actual_q=actual_q,
                gripper_joint_pos=grip_pos,
            )

            # Control actuators with smooth gripper velocity ramp
            diff = output.commanded_gripper - curr_grip
            curr_grip += float(np.clip(diff, -0.04, 0.04))
            targets = list(output.commanded_q) + [curr_grip]
            env.set_actuator_targets(targets)
            for _ in range(15):
                env.step()

            # Render overlay
            fps = 1.0 / max(1e-4, time.time() - prev_time)
            prev_time = time.time()

            display = frame.copy()
            tof_m = tof_mm / 1000.0 if tof_mm > 0 else -1.0
            draw_sensor_overlay(display, tof_distance_m=tof_m, fps=fps)

            # State machine status banner
            cv2.putText(
                display,
                f"STATE: {output.state.value}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"{output.message}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            # 3D Overview banner
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

            combined = np.hstack([display, overview_frame])
            cv2.imshow(window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("User requested exit")
                break

            if output.is_finished:
                logger.info("Cycle finished successfully!")
                time.sleep(1.0)
                break
    finally:
        env.close()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
