import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from manipulator3.hardware import (
    GPIOBackend,
    MockGPIOBackend,
    ServoController,
    StepperDriverA4988,
    get_default_gpio_backend,
    PIN_STEP,
    PIN_DIR,
    PIN_SERVO_SHOULDER_A,
    PIN_SERVO_SHOULDER_B,
    PIN_SERVO_ELBOW,
    PIN_SERVO_WRIST_PITCH,
    PIN_SERVO_WRIST_ROLL,
    PIN_SERVO_GRIPPER,
)

logger = logging.getLogger("servo_tester")
STATE_FILE_PATH = os.environ.get("MANIPULATOR_STATE_FILE", "/tmp/manipulator3_servo_state.json")


class ServoTester:
    """Utility to test, calibrate, and manually command servos on Raspberry Pi Zero W."""

    MOTOR_DEFINITIONS: Dict[str, Dict[str, Any]] = {
        "shoulder_a": {
            "id": 1,
            "key": "shoulder_a",
            "name": "Shoulder A (MG996R)",
            "pin": PIN_SERVO_SHOULDER_A,
            "min_deg": -90.0,
            "max_deg": 90.0,
            "min_rad": -1.5708,
            "max_rad": 1.5708,
            "type": "servo",
        },
        "shoulder_b": {
            "id": 2,
            "key": "shoulder_b",
            "name": "Shoulder B (MG996R, reversed)",
            "pin": PIN_SERVO_SHOULDER_B,
            "min_deg": -90.0,
            "max_deg": 90.0,
            "min_rad": -1.5708,
            "max_rad": 1.5708,
            "type": "servo",
        },
        "shoulder": {
            "id": None,
            "key": "shoulder",
            "name": "Shoulder Dual (A + B coupled)",
            "pin": f"{PIN_SERVO_SHOULDER_A},{PIN_SERVO_SHOULDER_B}",
            "min_deg": -90.0,
            "max_deg": 90.0,
            "min_rad": -1.5708,
            "max_rad": 1.5708,
            "type": "coupled_servo",
        },
        "elbow": {
            "id": 3,
            "key": "elbow",
            "name": "Elbow (MG996R)",
            "pin": PIN_SERVO_ELBOW,
            "min_deg": -90.0,
            "max_deg": 90.0,
            "min_rad": -1.5708,
            "max_rad": 1.5708,
            "type": "servo",
        },
        "wrist_pitch": {
            "id": 4,
            "key": "wrist_pitch",
            "name": "Wrist Pitch (MG996R)",
            "pin": PIN_SERVO_WRIST_PITCH,
            "min_deg": -90.0,
            "max_deg": 90.0,
            "min_rad": -1.5708,
            "max_rad": 1.5708,
            "type": "servo",
        },
        "wrist_roll": {
            "id": 5,
            "key": "wrist_roll",
            "name": "Wrist Roll (MG90S)",
            "pin": PIN_SERVO_WRIST_ROLL,
            "min_deg": -90.0,
            "max_deg": 90.0,
            "min_rad": -1.5708,
            "max_rad": 1.5708,
            "type": "servo",
        },
        "gripper": {
            "id": 6,
            "key": "gripper",
            "name": "Gripper Jaw (MG90S)",
            "pin": PIN_SERVO_GRIPPER,
            "min_deg": -34.4,
            "max_deg": 34.4,
            "min_rad": -0.6,
            "max_rad": 0.6,
            "type": "servo",
        },
        "base": {
            "id": 0,
            "key": "base",
            "name": "Azimuth Base (NEMA 17 / A4988)",
            "pin": f"{PIN_STEP},{PIN_DIR}",
            "min_deg": -180.0,
            "max_deg": 180.0,
            "min_rad": -math.pi,
            "max_rad": math.pi,
            "type": "stepper",
        },
    }

    ALIAS_MAP: Dict[str, str] = {
        "1": "shoulder_a",
        "shoulder_a": "shoulder_a",
        "servo_1": "shoulder_a",
        "sh_a": "shoulder_a",
        "2": "shoulder_b",
        "shoulder_b": "shoulder_b",
        "servo_2": "shoulder_b",
        "sh_b": "shoulder_b",
        "s": "shoulder",
        "sh": "shoulder",
        "shoulder": "shoulder",
        "revolute_2": "shoulder",
        "3": "elbow",
        "elbow": "elbow",
        "servo_3": "elbow",
        "el": "elbow",
        "revolute_3": "elbow",
        "4": "wrist_pitch",
        "wrist_pitch": "wrist_pitch",
        "pitch": "wrist_pitch",
        "wp": "wrist_pitch",
        "servo_4": "wrist_pitch",
        "revolute_4": "wrist_pitch",
        "5": "wrist_roll",
        "wrist_roll": "wrist_roll",
        "roll": "wrist_roll",
        "wr": "wrist_roll",
        "servo_5": "wrist_roll",
        "revolute_5": "wrist_roll",
        "6": "gripper",
        "gripper": "gripper",
        "jaw": "gripper",
        "g": "gripper",
        "servo_6": "gripper",
        "revolute_7": "gripper",
        "0": "base",
        "base": "base",
        "stepper": "base",
        "b": "base",
        "azimuth": "base",
        "revolute_1": "base",
    }

    def __init__(
        self,
        gpio: Optional[GPIOBackend] = None,
        force_mock: bool = False,
        freq_hz: float = 50.0,
    ) -> None:
        self.gpio = gpio or get_default_gpio_backend(force_mock=force_mock)
        self.servos = ServoController(self.gpio, freq_hz=freq_hz)
        self.stepper = StepperDriverA4988(self.gpio)
        self.freq_hz = freq_hz
        self.current_angles_rad: Dict[str, float] = {
            "shoulder_a": 0.0,
            "shoulder_b": 0.0,
            "shoulder": 0.0,
            "elbow": 0.0,
            "wrist_pitch": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
            "base": 0.0,
        }
        self._load_persisted_state()

    def _load_persisted_state(self) -> None:
        """Load last commanded angles from local disk state cache."""
        try:
            if os.path.exists(STATE_FILE_PATH):
                with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if k in self.current_angles_rad and isinstance(v, (int, float)):
                                self.current_angles_rad[k] = float(v)
        except Exception:
            pass

    def _save_persisted_state(self) -> None:
        """Persist current angles to disk for smooth multi-command continuity."""
        try:
            with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.current_angles_rad, f)
        except Exception:
            pass

    def resolve_motor(self, identifier: Union[int, str]) -> str:
        """Resolve motor name or index alias to canonical key."""
        clean = str(identifier).lower().strip()
        if clean in self.ALIAS_MAP:
            return self.ALIAS_MAP[clean]
        raise ValueError(
            f"Unknown motor '{identifier}'. Valid options: 0-6, shoulder_a, shoulder_b, shoulder, elbow, wrist_pitch, wrist_roll, gripper, base"
        )

    def set_motor_angle(
        self,
        motor: Union[int, str],
        angle: float,
        in_degrees: bool = True,
        smooth: bool = False,
        speed_deg_per_s: float = 60.0,
        from_angle: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Set target angle for a single motor or coupled shoulder with optional smooth interpolation."""
        key = self.resolve_motor(motor)
        mdef = self.MOTOR_DEFINITIONS[key]

        target_rad = math.radians(angle) if in_degrees else float(angle)
        target_deg = float(angle) if in_degrees else math.degrees(target_rad)

        clamped = False
        if target_rad < mdef["min_rad"] or target_rad > mdef["max_rad"]:
            target_rad = max(mdef["min_rad"], min(mdef["max_rad"], target_rad))
            target_deg = math.degrees(target_rad)
            clamped = True

        if from_angle is not None:
            current_rad = math.radians(from_angle) if in_degrees else float(from_angle)
            current_deg = math.degrees(current_rad)
            self.current_angles_rad[key] = current_rad
        else:
            current_rad = self.current_angles_rad.get(key, 0.0)
            current_deg = math.degrees(current_rad)

        diff_deg = abs(target_deg - current_deg)

        if smooth and diff_deg > 0.2:
            step_rate_hz = 50.0  # 50 Hz PWM frame rate
            duration = max(0.04, diff_deg / max(1.0, speed_deg_per_s))
            num_steps = max(2, int(round(duration * step_rate_hz)))
            step_dt = duration / num_steps
            for i in range(1, num_steps + 1):
                alpha = i / num_steps
                interp_rad = current_rad + (target_rad - current_rad) * alpha
                self._apply_angle(key, interp_rad)
                time.sleep(step_dt)
        else:
            self._apply_angle(key, target_rad)

        self._save_persisted_state()
        return self.get_motor_status(key, clamped=clamped)

    def _apply_angle(self, key: str, angle_rad: float) -> None:
        """Internal low-level dispatcher to hardware backends."""
        if key == "base":
            self.stepper.set_target_angle(angle_rad)
            self.stepper.step_towards_target()
            self.current_angles_rad["base"] = angle_rad
        elif key == "shoulder":
            self.servos.set_shoulder_angle(angle_rad)
            self.current_angles_rad["shoulder"] = angle_rad
            self.current_angles_rad["shoulder_a"] = angle_rad
            self.current_angles_rad["shoulder_b"] = angle_rad
        elif key in ("shoulder_a", "shoulder_b"):
            self.servos.set_single_servo(key, angle_rad)
            self.current_angles_rad[key] = angle_rad
        else:
            self.servos.set_joint_angle(key, angle_rad)
            self.current_angles_rad[key] = angle_rad

    def get_motor_status(self, key: str, clamped: bool = False) -> Dict[str, Any]:
        """Retrieve telemetry dict for a specific motor."""
        key = self.resolve_motor(key)
        mdef = self.MOTOR_DEFINITIONS[key]
        angle_rad = self.current_angles_rad.get(key, 0.0)
        angle_deg = math.degrees(angle_rad)

        pulse_us = 0.0
        duty_pct = 0.0

        if mdef["type"] == "servo":
            cfg = self.servos.configs[key]
            pulse_us = cfg.angle_to_pulse_us(angle_rad)
            duty_pct = cfg.pulse_us_to_duty_cycle(pulse_us, self.freq_hz)
        elif mdef["type"] == "coupled_servo":
            cfg_a = self.servos.configs["shoulder_a"]
            cfg_b = self.servos.configs["shoulder_b"]
            pulse_a = cfg_a.angle_to_pulse_us(angle_rad)
            pulse_b = cfg_b.angle_to_pulse_us(angle_rad)
            pulse_us = pulse_a
            duty_pct = cfg_a.pulse_us_to_duty_cycle(pulse_a, self.freq_hz)

        return {
            "key": key,
            "id": mdef["id"],
            "name": mdef["name"],
            "pin": mdef["pin"],
            "angle_deg": round(angle_deg, 2),
            "angle_rad": round(angle_rad, 4),
            "pulse_us": round(pulse_us, 1),
            "duty_pct": round(duty_pct, 2),
            "min_deg": mdef["min_deg"],
            "max_deg": mdef["max_deg"],
            "clamped": clamped,
            "type": mdef["type"],
        }

    def center_all(self, smooth: bool = False, speed_deg_per_s: float = 60.0) -> None:
        """Reset all motors to center position (0 deg / 1500 us)."""
        sequence = ["gripper", "wrist_roll", "wrist_pitch", "elbow", "shoulder", "base"]
        for key in sequence:
            self.set_motor_angle(key, 0.0, in_degrees=True, smooth=smooth, speed_deg_per_s=speed_deg_per_s)
        self._save_persisted_state()

    def relax(self, motor: Optional[Union[int, str]] = None) -> None:
        """Cut off PWM duty cycle to relax servo holding torque."""
        if motor is None:
            self.servos.emergency_stop()
        else:
            key = self.resolve_motor(motor)
            self.servos.relax_servo(key)
        self._save_persisted_state()

    def sweep(
        self,
        motor: Union[int, str],
        min_deg: Optional[float] = None,
        max_deg: Optional[float] = None,
        cycles: int = 1,
        speed_deg_per_s: float = 45.0,
    ) -> None:
        """Sweep motor between minimum and maximum bounds for verification."""
        key = self.resolve_motor(motor)
        mdef = self.MOTOR_DEFINITIONS[key]
        lo = mdef["min_deg"] if min_deg is None else max(mdef["min_deg"], min_deg)
        hi = mdef["max_deg"] if max_deg is None else min(mdef["max_deg"], max_deg)

        for _ in range(cycles):
            self.set_motor_angle(key, lo, in_degrees=True, smooth=True, speed_deg_per_s=speed_deg_per_s)
            self.set_motor_angle(key, hi, in_degrees=True, smooth=True, speed_deg_per_s=speed_deg_per_s)
            self.set_motor_angle(key, 0.0, in_degrees=True, smooth=True, speed_deg_per_s=speed_deg_per_s)

    def print_status_table(self) -> None:
        """Print clear ASCII table with all motor statuses."""
        print("+-----+------------------------------+-------+----------+-------------+--------------+")
        print("| ID  | Motor Name                   | GPIO  | Angle(°) | Pulse (µs)  | Limits (°)   |")
        print("+-----+------------------------------+-------+----------+-------------+--------------+")
        keys = ["base", "shoulder_a", "shoulder_b", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
        for k in keys:
            st = self.get_motor_status(k)
            id_str = str(st["id"]) if st["id"] is not None else "s"
            pulse_str = f"{st['pulse_us']:6.1f} µs" if st["type"] != "stepper" else "   N/A    "
            limits_str = f"[{st['min_deg']:+4.0f}, {st['max_deg']:+4.0f}]"
            print(
                f"| {id_str:<3} | {st['name']:<28} | {str(st['pin']):<5} | {st['angle_deg']:+7.1f}° | {pulse_str} | {limits_str:<12} |"
            )
        print("+-----+------------------------------+-------+----------+-------------+--------------+")

    def run_interactive(self, default_smooth: bool = False, default_speed: float = 60.0) -> None:
        """Interactive console interface for manual calibration and testing."""
        smooth_mode = default_smooth
        smooth_speed = default_speed
        print("\n==================================================================")
        print("   Raspberry Pi Zero W Manipulator - Servo Testing Console       ")
        print("==================================================================")
        self.print_status_table()
        mode_str = "ENABLED" if smooth_mode else "DISABLED"
        print(f"[INFO] Smooth mode: {mode_str} (speed: {smooth_speed:.1f} deg/s)")
        print("\nCommands:")
        print("  <motor> <angle>         : Set motor angle in degrees (e.g. '3 45', 'elbow -20')")
        print("  <motor> <angle> [s|f]   : Move smoothly [s] or fast [f]")
        print("  smooth [on|off|<speed>] : Toggle smooth mode or set speed (e.g. 'smooth 45')")
        print("  center [smooth|fast]    : Set all motors to center 0°")
        print("  relax [motor]           : Cut PWM power to relax motor (or all if omitted)")
        print("  sweep <motor>           : Smoothly sweep motor between min and max limits")
        print("  status                  : Print current status table")
        print("  quit / exit / q         : Safely exit and cleanup GPIO\n")

        while True:
            try:
                raw = input("servo-test> ").strip()
                if not raw:
                    continue

                parts = raw.split()
                cmd = parts[0].lower()

                if cmd in ("quit", "exit", "q"):
                    print("[INFO] Exiting servo tester.")
                    break
                elif cmd == "status":
                    self.print_status_table()
                elif cmd == "smooth":
                    if len(parts) == 1:
                        smooth_mode = not smooth_mode
                    elif parts[1].lower() in ("on", "1", "true"):
                        smooth_mode = True
                    elif parts[1].lower() in ("off", "0", "false"):
                        smooth_mode = False
                    else:
                        try:
                            smooth_speed = float(parts[1])
                            smooth_mode = True
                        except ValueError:
                            pass
                    st_str = "ENABLED" if smooth_mode else "DISABLED"
                    print(f"[INFO] Smooth mode: {st_str} ({smooth_speed:.1f} deg/s)")
                elif cmd == "center":
                    use_smooth = smooth_mode
                    if len(parts) > 1:
                        if parts[1].lower() in ("smooth", "s", "-s"):
                            use_smooth = True
                        elif parts[1].lower() in ("fast", "f", "-f"):
                            use_smooth = False
                    self.center_all(smooth=use_smooth, speed_deg_per_s=smooth_speed)
                    mode_txt = "smoothly " if use_smooth else ""
                    print(f"[INFO] All motors {mode_txt}centered to 0°.")
                elif cmd == "relax":
                    motor = parts[1] if len(parts) > 1 else None
                    self.relax(motor)
                    target_str = motor if motor else "all servos"
                    print(f"[INFO] Relaxed holding torque on {target_str}.")
                elif cmd == "sweep":
                    if len(parts) < 2:
                        print("[ERROR] Please specify motor to sweep (e.g. 'sweep 3' or 'sweep elbow').")
                        continue
                    print(f"[INFO] Sweeping motor {parts[1]}...")
                    self.sweep(parts[1], speed_deg_per_s=smooth_speed)
                    print("[INFO] Sweep complete.")
                else:
                    use_smooth = smooth_mode
                    if len(parts) == 1:
                        motor_input = parts[0]
                        resolved = self.resolve_motor(motor_input)
                        mdef = self.MOTOR_DEFINITIONS[resolved]
                        angle_str = input(f"Enter target angle for {resolved} ({mdef['min_deg']}° to {mdef['max_deg']}°): ").strip()
                        angle_val = float(angle_str)
                    else:
                        motor_input = parts[0]
                        angle_val = float(parts[1])
                        if len(parts) >= 3:
                            flag = parts[2].lower()
                            if flag in ("smooth", "s", "-s"):
                                use_smooth = True
                            elif flag in ("fast", "f", "-f"):
                                use_smooth = False

                    res = self.set_motor_angle(motor_input, angle_val, in_degrees=True, smooth=use_smooth, speed_deg_per_s=smooth_speed)
                    pulse_info = f" | Pulse: {res['pulse_us']} µs ({res['duty_pct']}%)" if res["type"] != "stepper" else ""
                    clamped_info = " [CLAMPED]" if res["clamped"] else ""
                    smooth_tag = " (smooth)" if use_smooth else ""
                    print(
                        f"[INFO] Motor '{res['key']}' (GPIO {res['pin']}) -> {res['angle_deg']}° ({res['angle_rad']} rad){pulse_info}{clamped_info}{smooth_tag}"
                    )
            except (ValueError, KeyError) as e:
                print(f"[WARNING] Invalid input: {e}")
            except (KeyboardInterrupt, EOFError):
                print("\n[INFO] Interrupted by user. Relaxing servos.")
                self.relax()
                break


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi Zero W Servo Testing Utility")
    parser.add_argument("-m", "--motor", type=str, help="Motor ID (0-6) or name (e.g. elbow, shoulder, gripper)")
    parser.add_argument("-a", "--angle", type=float, help="Target angle (degrees by default)")
    parser.add_argument(
        "-f",
        "--from-angle",
        type=float,
        default=None,
        help="Explicit starting angle in degrees for smooth interpolation (default: last tracked position)",
    )
    parser.add_argument("--rad", action="store_true", help="Interpret angle in radians instead of degrees")
    parser.add_argument("--smooth", action="store_true", help="Interpolate movement gradually")
    parser.add_argument("--speed", type=float, default=60.0, help="Smooth movement speed (deg/s)")
    parser.add_argument(
        "--hold",
        type=float,
        default=None,
        help="Seconds to hold PWM signal active (default: 0.2s for smooth, 0.5s for instant; 0=exit immediately; -1=indefinitely)",
    )
    parser.add_argument("--center", action="store_true", help="Center all motors to 0°")
    parser.add_argument("--relax", action="store_true", help="Relax holding torque on all servos")
    parser.add_argument("--sweep", type=str, help="Sweep specified motor through its range")
    parser.add_argument("--status", action="store_true", help="Print current status table and exit")
    parser.add_argument("--mock", action="store_true", help="Force MockGPIOBackend (for testing on non-RPi)")
    parser.add_argument(
        "--pin-shoulder-b",
        type=int,
        default=None,
        help="BCM GPIO pin for Shoulder B (default: PIN_SERVO_SHOULDER_B env or 19 / Pin 35)",
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive testing console")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show informational debug logs")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    tester = ServoTester(force_mock=args.mock)

    if args.pin_shoulder_b is not None:
        tester.servos.configs["shoulder_b"].pin = args.pin_shoulder_b
        tester.MOTOR_DEFINITIONS["shoulder_b"]["pin"] = args.pin_shoulder_b
        tester.MOTOR_DEFINITIONS["shoulder"]["pin"] = f"{PIN_SERVO_SHOULDER_A},{args.pin_shoulder_b}"

    if args.status:
        tester.print_status_table()
        return 0

    if args.relax:
        tester.relax()
        print("[INFO] Relaxed holding torque on all servos.")
        return 0

    if args.center:
        tester.center_all(smooth=args.smooth, speed_deg_per_s=args.speed)
        print("[INFO] All motors centered to 0°.")
        hold_time = args.hold if args.hold is not None else (0.2 if args.smooth else 0.5)
        if hold_time > 0:
            if hold_time > 0.3:
                print(f"[INFO] Holding center position for {hold_time:.1f}s...")
            try:
                time.sleep(hold_time)
            except KeyboardInterrupt:
                pass
        return 0

    if args.sweep:
        print(f"[INFO] Sweeping motor '{args.sweep}'...")
        tester.sweep(args.sweep, speed_deg_per_s=args.speed)
        print("[INFO] Sweep finished.")
        return 0

    if args.motor is not None and args.angle is not None:
        in_degrees = not args.rad
        res = tester.set_motor_angle(
            args.motor,
            args.angle,
            in_degrees=in_degrees,
            smooth=args.smooth,
            speed_deg_per_s=args.speed,
            from_angle=args.from_angle,
        )
        pulse_info = f" | Pulse: {res['pulse_us']} µs ({res['duty_pct']}%)" if res["type"] != "stepper" else ""
        clamped_info = " [CLAMPED TO LIMITS]" if res["clamped"] else ""
        smooth_tag = " (smooth)" if args.smooth else ""
        print(
            f"[INFO] Motor '{res['key']}' (GPIO {res['pin']}) -> {res['angle_deg']}° ({res['angle_rad']} rad){pulse_info}{clamped_info}{smooth_tag}"
        )

        hold_time = args.hold if args.hold is not None else (0.2 if args.smooth else 0.5)
        if hold_time != 0 and res["type"] != "stepper":
            if hold_time < 0:
                print("[INFO] Holding position indefinitely. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    print("\n[INFO] Stopped holding.")
            elif hold_time > 0.3:
                print(f"[INFO] Holding PWM signal for {hold_time:.1f}s to allow servo to reach position...")
                try:
                    time.sleep(hold_time)
                except KeyboardInterrupt:
                    pass
            elif hold_time > 0:
                try:
                    time.sleep(hold_time)
                except KeyboardInterrupt:
                    pass
        return 0

    # Default to interactive console
    tester.run_interactive(default_smooth=args.smooth, default_speed=args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
