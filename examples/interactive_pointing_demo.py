import argparse
from manipulator3.interactive import InteractivePointingDemo


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Hand Pointing Pick-and-Place")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0, set -1 for mouse/keyboard mode)")
    args = parser.parse_args()

    demo = InteractivePointingDemo(camera_id=args.camera)
    demo.run()


if __name__ == "__main__":
    main()
