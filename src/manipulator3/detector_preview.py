import logging
import time
import cv2
import numpy as np

from manipulator3.detector import (
    YOLOObjectDetector,
    ColorBlobDetector,
    draw_detection_overlay,
)
from manipulator3.kinematics import POSE_HOME, POSE_SCAN
from manipulator3.servoing import (
    VisualServoController,
    SectorScanner,
    VisualSearchTracker,
    TrackingState,
)
from manipulator3.simulation import SimulationEnvironment, draw_sensor_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("detector_preview")


def main() -> None:
    logger.info("Starting YOLO11n & Visual Servoing preview")
    env = SimulationEnvironment()

    # Move to initial scan pose
    current_q = POSE_SCAN.copy()
    env.set_actuator_targets(current_q)
    for _ in range(100):
        env.step()

    # Initialize detectors
    color_detector = ColorBlobDetector()
    yolo_detector = None
    try:
        yolo_detector = YOLOObjectDetector(conf_threshold=0.20)
        logger.info("YOLO11n detector ready")
    except Exception as exc:
        logger.warning("Could not load YOLO11n (%s), using ColorBlobDetector", exc)

    active_detector = color_detector
    active_mode = "COLOR"

    tracker = VisualSearchTracker(
        detector=active_detector,
        servo_controller=VisualServoController(k_yaw=0.35, k_pitch=0.30),
        scanner=SectorScanner(),
    )

    auto_servo = True
    azimuth_target = 0.0

    logger.info("Controls: [Y] Toggle YOLO/Color | [S] Search Sweep | [C] Toggle Servoing | [R] Reset | [Q] Quit")

    while True:
        t0 = time.perf_counter()

        # Step simulation physics
        env.step(5)

        # Render feeds
        gripper_frame = env.render_bgr("gripper_camera")
        overview_frame = env.render_bgr("overview_camera")
        tof_m = env.get_tof_reading()

        # Update tracker & servoing if active
        if auto_servo:
            new_q, state, best_target = tracker.update(gripper_frame, current_q)
            current_q = new_q
            env.set_actuator_targets(current_q)
        else:
            state = TrackingState.IDLE
            detections = active_detector.detect(gripper_frame)
            best_target = active_detector.get_best_target(detections) if hasattr(active_detector, "get_best_target") else (detections[0] if detections else None)

        detections = active_detector.detect(gripper_frame)
        det_overlay = draw_detection_overlay(gripper_frame, detections, best_target)
        hud = draw_sensor_overlay(
            det_overlay,
            tof_distance_m=tof_m,
            fps=1.0 / max(1e-4, time.perf_counter() - t0),
            status_text=f"Mode:{active_mode} State:{state.value} Target:{best_target.class_name if best_target else 'None'}",
        )

        # Overview preview resize
        h, w = hud.shape[:2]
        overview_small = cv2.resize(overview_frame, (w // 2, h // 2))
        combined = np.hstack([hud, cv2.resize(overview_frame, (w, h))])

        cv2.imshow("Manipulator3 - YOLO11n & Visual Servoing", combined)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("y"):
            if yolo_detector is not None:
                if active_mode == "COLOR":
                    active_detector = yolo_detector
                    active_mode = "YOLO11n"
                else:
                    active_detector = color_detector
                    active_mode = "COLOR"
                tracker.detector = active_detector
                logger.info("Switched detector to %s", active_mode)
        elif key == ord("s"):
            azimuth_target = float(np.random.uniform(-0.35, 0.35))
            logger.info("Starting sector search sweep at azimuth %.2f rad", azimuth_target)
            current_q = tracker.start_search(azimuth_rad=azimuth_target)
            env.set_actuator_targets(current_q)
            auto_servo = True
        elif key == ord("c"):
            auto_servo = not auto_servo
            logger.info("Auto-servoing: %s", auto_servo)
        elif key == ord("r"):
            current_q = POSE_SCAN.copy()
            env.set_actuator_targets(current_q)
            tracker.state = TrackingState.IDLE
            logger.info("Reset to POSE_SCAN")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
