import argparse
import sys
import time
import cv2
import numpy as np

from manipulator3.pointing import (
    HandTracker,
    PointingStatus,
    PointingTarget,
    HandLandmarks,
    intersect_ray_with_table,
    compute_pointing_vector,
    DEFAULT_TABLE_BOUNDS,
)


def create_synthetic_pointing_hand(
    tip_screen_pos: tuple[float, float],
    direction_2d: tuple[float, float],
) -> tuple[np.ndarray, HandLandmarks]:
    """Generate a synthetic RGB hand image and corresponding HandLandmarks for simulation/testing."""
    h, w = 480, 640
    canvas = np.full((h, w, 3), 40, dtype=np.uint8)

    # Table area guide in background
    cv2.rectangle(canvas, (100, 100), (540, 420), (60, 60, 60), 1)
    cv2.putText(
        canvas,
        "VIRTUAL TABLE AREA",
        (110, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 100, 100),
        1,
    )

    tx, ty = tip_screen_pos
    dx, dy = direction_2d
    d_norm = float(np.hypot(dx, dy)) + 1e-6
    dx, dy = dx / d_norm, dy / d_norm

    # Index finger line
    mcp_x = tx - dx * 0.15
    mcp_y = ty - dy * 0.15
    wrist_x = mcp_x - dx * 0.10
    wrist_y = mcp_y - dy * 0.10

    lms = np.zeros((21, 3), dtype=np.float64)
    # 2D screen coords in [0, 1]
    lms[0] = [wrist_x, wrist_y, 0.0]
    lms[5] = [mcp_x, mcp_y, 0.0]
    lms[6] = [mcp_x + dx * 0.05, mcp_y + dy * 0.05, 0.0]
    lms[7] = [mcp_x + dx * 0.10, mcp_y + dy * 0.10, 0.0]
    lms[8] = [tx, ty, 0.0]
    # Curled fingers
    perp_x, perp_y = -dy, dx
    lms[9] = [mcp_x + perp_x * 0.04, mcp_y + perp_y * 0.04, 0.0]
    lms[10] = [mcp_x + perp_x * 0.04 + dx * 0.02, mcp_y + perp_y * 0.04 + dy * 0.02, 0.0]
    lms[12] = [mcp_x + perp_x * 0.04, mcp_y + perp_y * 0.04, 0.02]
    lms[16] = [mcp_x + perp_x * 0.07, mcp_y + perp_y * 0.07, 0.02]
    lms[20] = [mcp_x + perp_x * 0.10, mcp_y + perp_y * 0.10, 0.02]

    # World landmarks in meters
    world_lms = lms.copy()
    world_lms[:, 0] = (world_lms[:, 0] - 0.5) * 0.4
    world_lms[:, 1] = (world_lms[:, 1] - 0.5) * 0.3
    world_lms[:, 2] = 0.35

    # Draw palm and fingers on canvas
    pts_int = np.array([[int(p[0] * w), int(p[1] * h)] for p in lms])
    cv2.circle(canvas, tuple(pts_int[0]), 18, (140, 180, 220), -1)
    cv2.line(canvas, tuple(pts_int[5]), tuple(pts_int[8]), (140, 180, 220), 10)
    for idx in [5, 6, 7, 8]:
        cv2.circle(canvas, tuple(pts_int[idx]), 6, (100, 150, 200), -1)

    hl = HandLandmarks(
        landmarks=lms,
        world_landmarks=world_lms,
        handedness="Right",
        score=0.98,
    )
    return canvas, hl


def run_hand_preview(
    camera_id: int = -1,
    headless: bool = False,
    num_frames: int = 150,
) -> None:
    """Run interactive hand tracking and pointing target estimation preview."""
    print("[INFO] Starting Stage 5 Hand Tracking & Pointing Ray Preview...")

    tracker = HandTracker()
    cap = None
    use_synthetic = True

    if camera_id >= 0:
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            use_synthetic = False
            print(f"[INFO] Using physical camera device ID: {camera_id}")
        else:
            print(f"[WARNING] Camera ID {camera_id} unavailable, falling back to synthetic preview mode.")
            cap = None

    window_name = "Manipulator3 - Hand Pointing & Target Estimation Preview"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    start_time = time.time()

    # Pre-defined test cases for synthetic preview sequence
    test_scenarios = [
        # (name, origin_3d, direction_3d, description)
        (
            "VALID_CENTER",
            np.array([0.25, 0.0, 0.35]),
            np.array([-0.25, 0.0, -0.85]),
            "Pointing at reachable table center",
        ),
        (
            "VALID_RIGHT_SECTOR",
            np.array([0.25, 0.15, 0.35]),
            np.array([-0.20, -0.20, -0.85]),
            "Pointing at right sector on table",
        ),
        (
            "OUT_OF_REACH",
            np.array([0.65, 0.30, 0.35]),
            np.array([-0.10, -0.10, -0.85]),
            "Pointing at far table edge out of reach",
        ),
        (
            "OUT_OF_TABLE",
            np.array([0.80, 0.60, 0.35]),
            np.array([0.20, 0.30, -0.85]),
            "Pointing off the table boundary",
        ),
        (
            "PARALLEL_RAY",
            np.array([0.20, 0.0, 0.35]),
            np.array([0.50, 0.0, 0.0]),
            "Pointing horizontally towards horizon",
        ),
    ]

    try:
        while True:
            if not use_synthetic and cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Failed to read frame from camera.")
                    break
                target, landmarks = tracker.estimate_pointing_target(frame)
                annotated = tracker.draw_overlay(frame, target, landmarks)
            else:
                # Cycle through scenarios in synthetic mode
                scenario_idx = (frame_count // 35) % len(test_scenarios)
                name, origin, dir_vec, desc = test_scenarios[scenario_idx]

                # Slight dynamic movement
                t = (frame_count % 35) / 35.0
                curr_origin = origin + np.array([0.02 * np.sin(t * 2 * np.pi), 0.0, 0.0])
                target = intersect_ray_with_table(curr_origin, dir_vec)

                # Render synthetic canvas
                screen_tip = (0.5 + curr_origin[0] * 0.3, 0.5 + curr_origin[1] * 0.3)
                canvas, landmarks = create_synthetic_pointing_hand(screen_tip, (dir_vec[0], dir_vec[1]))
                target.ray_origin = curr_origin
                target.ray_direction = dir_vec / np.linalg.norm(dir_vec)

                annotated = tracker.draw_overlay(canvas, target, landmarks)
                cv2.putText(
                    annotated,
                    f"Scenario: {name} ({desc})",
                    (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

            frame_count += 1

            if not headless:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(30) & 0xFF
                if key == 27 or key == ord("q"):
                    print("[INFO] User requested exit.")
                    break

            if frame_count >= num_frames:
                break

    finally:
        if cap is not None:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)

    elapsed = time.time() - start_time
    fps = frame_count / (elapsed + 1e-6)
    print(f"[INFO] Preview finished. Processed {frame_count} frames in {elapsed:.2f}s ({fps:.1f} FPS).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manipulator3 Hand Tracking & Pointing Estimation Preview")
    parser.add_argument("--camera", type=int, default=-1, help="Camera device index (default: -1 for synthetic mode)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI windows")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames to process in preview")
    args = parser.parse_args()

    run_hand_preview(camera_id=args.camera, headless=args.headless, num_frames=args.frames)


if __name__ == "__main__":
    main()
