#!/usr/bin/env python3
"""GPIO Diagnostics and Hardware Health Check Tool for Raspberry Pi.

This tool verifies whether Raspberry Pi GPIO pins are functional or burned out.
It supports:
  1. Static HIGH (3.3V) / LOW (0.0V) level hold (multimeter / oscilloscope test).
  2. Slow square wave blinking (1 Hz) for visual scope triggering.
  3. Continuous 50 Hz servo PWM (1500 µs pulse) for servo signal verification.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from manipulator3.hardware import ensure_pigpiod_running, set_rpi_cpu_performance_mode
except Exception:
    def ensure_pigpiod_running() -> bool:
        return False
    def set_rpi_cpu_performance_mode() -> bool:
        return False

MANIPULATOR_PINS: Dict[str, Dict[str, Any]] = {
    "shoulder_a": {"name": "Shoulder A (MG996R)", "bcm": 26, "phys": 37, "gnd": 39},
    "shoulder_b": {"name": "Shoulder B (MG996R)", "bcm": 19, "phys": 35, "gnd": 39},
    "elbow": {"name": "Elbow (MG996R)", "bcm": 13, "phys": 33, "gnd": 34},
    "wrist_pitch": {"name": "Wrist Pitch (MG996R)", "bcm": 6, "phys": 31, "gnd": 30},
    "wrist_roll": {"name": "Wrist Roll (MG90S)", "bcm": 5, "phys": 29, "gnd": 30},
    "gripper": {"name": "Gripper Jaw (MG90S)", "bcm": 22, "phys": 15, "gnd": 14},
    "step": {"name": "Azimuth Step (A4988)", "bcm": 17, "phys": 11, "gnd": 9},
    "dir": {"name": "Azimuth Dir (A4988)", "bcm": 27, "phys": 13, "gnd": 14},
}

# Lookup BCM to physical pin mapping for standard 40-pin header
BCM_TO_PHYS: Dict[int, Tuple[int, int]] = {
    # bcm: (physical_pin, nearest_gnd_pin)
    2: (3, 6),
    3: (5, 6),
    4: (7, 6),
    17: (11, 9),
    27: (13, 14),
    22: (15, 14),
    10: (19, 20),
    9: (21, 20),
    11: (23, 20),
    5: (29, 30),
    6: (31, 30),
    13: (33, 34),
    19: (35, 34),
    26: (37, 39),
    14: (8, 6),
    15: (10, 9),
    18: (12, 14),
    23: (16, 14),
    24: (18, 20),
    25: (22, 20),
    8: (24, 25),
    7: (26, 25),
    1: (28, 30),
    12: (32, 34),
    16: (36, 34),
    20: (38, 39),
    21: (40, 39),
}


class DirectGPIOController:
    """Hardware abstraction supporting pigpio, lgpio, RPi.GPIO, or Mock."""

    def __init__(self, force_mock: bool = False) -> None:
        self.backend_name = "mock"
        self._pigpio_pi = None
        self._lgpio_handle = None
        self._lgpio_mod = None
        self._rpi_gpio = None
        self._active_pins: set[int] = set()
        self._current_pulses: Dict[int, int] = {}

        if force_mock:
            self.backend_name = "Mock (Virtual)"
            return

        set_rpi_cpu_performance_mode()
        ensure_pigpiod_running()

        # 1. Try pigpio (best for DMA PWM)
        try:
            import pigpio
            pi = pigpio.pi()
            if not pi.connected:
                ensure_pigpiod_running()
                pi = pigpio.pi()
            if pi.connected:
                self._pigpio_pi = pi
                self.backend_name = "pigpio (DMA hardware PWM - ZERO JITTER)"
                try:
                    self._pigpio_pi.set_pad_strength(0, 16)
                except Exception:
                    pass
                return
        except Exception:
            pass

        # 2. Try lgpio (standard on Raspberry Pi OS Bookworm)
        try:
            import lgpio
            self._lgpio_mod = lgpio
            self._lgpio_handle = lgpio.gpiochip_open(0)
            self.backend_name = "lgpio (kernel gpiochip)"
            return
        except Exception:
            pass

        # 3. Try RPi.GPIO (legacy)
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self._rpi_gpio = GPIO
            self.backend_name = "RPi.GPIO"
            return
        except Exception:
            pass

        self.backend_name = "Mock (Virtual - no hardware libs detected)"

    def setup_output(self, bcm_pin: int) -> None:
        if self._pigpio_pi:
            self._pigpio_pi.set_mode(bcm_pin, self._pigpio_pi.OUTPUT)
            try:
                self._pigpio_pi.set_pull_up_down(bcm_pin, self._pigpio_pi.PUD_OFF)
            except Exception:
                pass
        elif self._lgpio_handle is not None and self._lgpio_mod is not None:
            self._lgpio_mod.gpio_claim_output(self._lgpio_handle, bcm_pin)
        elif self._rpi_gpio:
            self._rpi_gpio.setup(bcm_pin, self._rpi_gpio.OUT)
        self._active_pins.add(bcm_pin)

    def write(self, bcm_pin: int, value: int) -> None:
        if bcm_pin not in self._active_pins:
            self.setup_output(bcm_pin)
        if self._pigpio_pi:
            self._pigpio_pi.write(bcm_pin, int(value))
        elif self._lgpio_handle is not None and self._lgpio_mod is not None:
            self._lgpio_mod.gpio_write(self._lgpio_handle, bcm_pin, int(value))
        elif self._rpi_gpio:
            self._rpi_gpio.output(bcm_pin, int(value))

    def set_servo_pulse(self, bcm_pin: int, pulse_us: float) -> None:
        if bcm_pin not in self._active_pins:
            self.setup_output(bcm_pin)
        pulse_int = int(round(pulse_us)) if pulse_us > 0.0 else 0
        if self._current_pulses.get(bcm_pin) == pulse_int:
            return
        self._current_pulses[bcm_pin] = pulse_int

        if self._pigpio_pi:
            self._pigpio_pi.set_servo_pulsewidth(bcm_pin, pulse_int)
        elif self._lgpio_handle is not None and self._lgpio_mod is not None:
            if hasattr(self._lgpio_mod, "tx_servo"):
                self._lgpio_mod.tx_servo(self._lgpio_handle, bcm_pin, pulse_int, 50)
            else:
                duty = (pulse_us / 20000.0) * 100.0
                self._lgpio_mod.tx_pwm(self._lgpio_handle, bcm_pin, 50, float(duty))
        elif self._rpi_gpio:
            duty = (pulse_us / 20000.0) * 100.0
            pwm = self._rpi_gpio.PWM(bcm_pin, 50)
            pwm.start(duty)

    def stop_pwm(self, bcm_pin: int) -> None:
        self._current_pulses[bcm_pin] = 0
        if self._pigpio_pi:
            self._pigpio_pi.set_servo_pulsewidth(bcm_pin, 0)
        elif self._lgpio_handle is not None and self._lgpio_mod is not None:
            if hasattr(self._lgpio_mod, "tx_servo"):
                self._lgpio_mod.tx_servo(self._lgpio_handle, bcm_pin, 0, 50)
            else:
                self._lgpio_mod.tx_pwm(self._lgpio_handle, bcm_pin, 0, 0)

    def cleanup(self) -> None:
        for pin in self._active_pins:
            try:
                self.stop_pwm(pin)
                self.write(pin, 0)
            except Exception:
                pass
        if self._pigpio_pi:
            try:
                self._pigpio_pi.stop()
            except Exception:
                pass
        if self._lgpio_handle is not None and self._lgpio_mod is not None:
            try:
                self._lgpio_mod.gpiochip_close(self._lgpio_handle)
            except Exception:
                pass
        if self._rpi_gpio:
            try:
                self._rpi_gpio.cleanup()
            except Exception:
                pass


def get_pin_info(target: str) -> Tuple[int, int, int, str]:
    """Resolve user input (name or number) to (bcm, physical, gnd, label)."""
    target_clean = target.lower().strip()
    if target_clean in MANIPULATOR_PINS:
        info = MANIPULATOR_PINS[target_clean]
        return info["bcm"], info["phys"], info["gnd"], info["name"]

    try:
        bcm = int(target_clean)
        phys, gnd = BCM_TO_PHYS.get(bcm, (0, 0))
        return bcm, phys, gnd, f"Custom GPIO {bcm}"
    except ValueError:
        raise ValueError(f"Unknown pin or motor: '{target}'. Available: {list(MANIPULATOR_PINS.keys())}")


def run_static_test(ctl: DirectGPIOController, bcm: int, phys: int, gnd: int, label: str, auto_duration: Optional[float] = None) -> None:
    print("\n" + "=" * 65)
    print(f"[*] STATIC LEVEL TEST: {label}")
    print(f"    - BCM GPIO:       {bcm}")
    print(f"    - Physical Pin:   {phys}  (Connect signal probe here)")
    print(f"    - Ground (GND):   {gnd}  (Connect black ground clip here)")
    print("=" * 65)

    print(f"\n[1/2] Setting GPIO {bcm} (Pin {phys}) to HIGH (+3.3V)...")
    ctl.write(bcm, 1)
    print("  -> Pin is now HIGH (+3.3V).")
    print(f"  -> Check multimeter or oscilloscope: should read ~3.30 V.")

    if auto_duration is not None:
        time.sleep(auto_duration)
    else:
        input("  -> Press [Enter] to switch to LOW (0.0V)... ")

    print(f"\n[2/2] Setting GPIO {bcm} (Pin {phys}) to LOW (0.0V)...")
    ctl.write(bcm, 0)
    print("  -> Pin is now LOW (0.0V).")
    print(f"  -> Check multimeter or oscilloscope: should read ~0.00 V.")

    if auto_duration is not None:
        time.sleep(auto_duration)
    else:
        input("  -> Press [Enter] to finish test... ")

    print(f"[OK] Static test for {label} completed.\n")


def run_blink_test(ctl: DirectGPIOController, bcm: int, phys: int, gnd: int, label: str, duration_sec: Optional[float] = None) -> None:
    print("\n" + "=" * 65)
    print(f"[*] SLOW BLINK TEST (1 Hz Square Wave): {label}")
    print(f"    - BCM GPIO:       {bcm}")
    print(f"    - Physical Pin:   {phys}  (Signal)")
    print(f"    - Ground (GND):   {gnd}  (GND clip)")
    print("    - Frequency:      1.0 Hz (500 ms HIGH, 500 ms LOW)")
    print("    - Scope scale:    1V/div or 2V/div, 200ms/div or 500ms/div")
    print("=" * 65)
    print("Running square wave... Press Ctrl+C to stop.")

    start = time.time()
    state = 0
    interval = 0.5
    next_tick = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            if now < next_tick:
                rem = next_tick - now
                if rem > 0.002:
                    time.sleep(rem - 0.001)
                while time.perf_counter() < next_tick:
                    pass
            next_tick += interval
            state = 1 - state
            ctl.write(bcm, state)
            sys.stdout.write(f"\r  -> State: {'[ HIGH +3.3V ]' if state else '[  LOW  0.0V ]'}")
            sys.stdout.flush()
            if duration_sec and (time.time() - start) >= duration_sec:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctl.write(bcm, 0)
        print("\n[OK] Blink test stopped. Pin reset to 0.0V.")


def run_freq_test(ctl: DirectGPIOController, bcm: int, phys: int, gnd: int, label: str, freq_hz: float = 800.0, duration_sec: Optional[float] = None) -> None:
    print("\n" + "=" * 65)
    print(f"[*] CONSTANT FREQUENCY STEP WAVE TEST ({freq_hz:.1f} Hz): {label}")
    print(f"    - BCM GPIO:       {bcm}")
    print(f"    - Physical Pin:   {phys}  (Signal)")
    print(f"    - Ground (GND):   {gnd}  (GND clip)")
    print(f"    - Target Freq:    {freq_hz:.1f} Hz (Period: {1000.0/freq_hz:.3f} ms)")
    print("    - Scope settings: 1V/div, Timebase: 0.5ms/div or 1ms/div")
    print("=" * 65)
    print("Generating constant frequency pulses... Press Ctrl+C to stop.")

    interval = 1.0 / freq_hz
    next_tick = time.perf_counter()
    start = time.time()
    try:
        while True:
            now = time.perf_counter()
            if now < next_tick:
                rem = next_tick - now
                if rem > 0.001:
                    time.sleep(rem - 0.0005)
                while time.perf_counter() < next_tick:
                    pass
            next_tick += interval

            ctl.write(bcm, 1)
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < 0.000005:
                pass
            ctl.write(bcm, 0)

            if duration_sec and (time.time() - start) >= duration_sec:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctl.write(bcm, 0)
        print("\n[OK] Frequency test stopped. Pin reset to 0.0V.")


def run_pwm_test(ctl: DirectGPIOController, bcm: int, phys: int, gnd: int, label: str, duration_sec: Optional[float] = None) -> None:
    print("\n" + "=" * 65)
    print(f"[*] SERVO PWM TEST (50 Hz, 1500 µs pulse): {label}")
    print(f"    - BCM GPIO:       {bcm}")
    print(f"    - Physical Pin:   {phys}  (Signal)")
    print(f"    - Ground (GND):   {gnd}  (GND clip)")
    print("    - Frequency:      50 Hz (Period: 20 ms)")
    print("    - Pulse width:    1500 µs (Neutral servo position)")
    print("    - Scope settings: 1V/div or 2V/div, Timebase: 5ms/div")
    print("=" * 65)
    print("Generating continuous servo PWM... Press Ctrl+C to stop.")

    ctl.set_servo_pulse(bcm, 1500.0)
    start = time.time()
    try:
        while True:
            time.sleep(0.2)
            if duration_sec and (time.time() - start) >= duration_sec:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctl.stop_pwm(bcm)
        print("\n[OK] PWM stopped. Pin disabled.")


def run_all_static_sequence(ctl: DirectGPIOController) -> None:
    print("\n" + "=" * 65)
    print("[*] BATCH DIAGNOSTICS: Testing all manipulator pins in sequence")
    print("=" * 65)
    for key, info in MANIPULATOR_PINS.items():
        print(f"\n--> Next up: {info['name']} on BCM {info['bcm']} (Physical Pin {info['phys']}, GND {info['gnd']})")
        resp = input("    Press [Enter] to test (or 's' to skip, 'q' to quit): ").strip().lower()
        if resp == 'q':
            break
        if resp == 's':
            continue
        run_static_test(ctl, info["bcm"], info["phys"], info["gnd"], info["name"])


def interactive_menu(ctl: DirectGPIOController) -> None:
    while True:
        print("\n" + "=" * 55)
        print("          RASPBERRY PI GPIO HARDWARE CHECK")
        print(f"  Backend: {ctl.backend_name}")
        print("=" * 55)
        print("Manipulator Motors / Pins:")
        for idx, (key, info) in enumerate(MANIPULATOR_PINS.items(), 1):
            print(f"  {idx}. {info['name']:<28} BCM {info['bcm']:<2} (Pin {info['phys']:<2}, GND {info['gnd']})")
        print("  9. Custom BCM GPIO Pin")
        print("  A. Test ALL pins in sequence")
        print("  Q. Quit")
        print("-" * 55)

        choice = input("Select an option: ").strip().lower()
        if choice in ("q", "exit"):
            break

        if choice == "a":
            run_all_static_sequence(ctl)
            continue

        target_bcm = None
        target_phys = 0
        target_gnd = 0
        target_label = ""

        if choice == "9":
            try:
                bcm_num = int(input("Enter BCM GPIO number (0-27): ").strip())
                target_bcm, target_phys, target_gnd, target_label = get_pin_info(str(bcm_num))
            except Exception as e:
                print(f"[ERROR] {e}")
                continue
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(MANIPULATOR_PINS):
                    key = list(MANIPULATOR_PINS.keys())[idx - 1]
                    target_bcm, target_phys, target_gnd, target_label = get_pin_info(key)
                else:
                    print("[ERROR] Invalid selection.")
                    continue
            except ValueError:
                # Try by key name directly
                try:
                    target_bcm, target_phys, target_gnd, target_label = get_pin_info(choice)
                except Exception as e:
                    print(f"[ERROR] {e}")
                    continue

        # Choose test mode
        print(f"\nSelected: {target_label} (BCM {target_bcm}, Pin {target_phys}, GND {target_gnd})")
        print("Test modes:")
        print("  1. Static 3.3V / 0V test (Multimeter / DC scope)")
        print("  2. Slow 1 Hz blink (Square wave)")
        print("  3. Continuous 50 Hz Servo PWM (1500 µs)")
        print("  4. Constant Frequency Step Pulse Wave (800 Hz)")
        print("  B. Back to main menu")

        mode_choice = input("Choose mode [1-4]: ").strip().lower()
        if mode_choice == "1":
            run_static_test(ctl, target_bcm, target_phys, target_gnd, target_label)
        elif mode_choice == "2":
            run_blink_test(ctl, target_bcm, target_phys, target_gnd, target_label)
        elif mode_choice == "3":
            run_pwm_test(ctl, target_bcm, target_phys, target_gnd, target_label)
        elif mode_choice == "4":
            run_freq_test(ctl, target_bcm, target_phys, target_gnd, target_label, freq_hz=800.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Raspberry Pi GPIO Pin Diagnostics Tool")
    parser.add_argument("--pin", type=str, default=None, help="Motor name (e.g. 'elbow') or BCM pin number (e.g. 13)")
    parser.add_argument("--mode", choices=["static", "blink", "pwm", "freq", "all"], default=None, help="Test mode")
    parser.add_argument("--freq", type=float, default=800.0, help="Frequency in Hz for freq mode (default: 800 Hz)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds for blink/pwm/freq mode")
    parser.add_argument("--mock", action="store_true", help="Force Mock backend for local testing")
    args = parser.parse_args()

    ctl = DirectGPIOController(force_mock=args.mock)
    print(f"[*] Initialized GPIO Controller using backend: {ctl.backend_name}")

    try:
        if args.mode == "all":
            run_all_static_sequence(ctl)
            return 0

        if args.pin is not None:
            bcm, phys, gnd, label = get_pin_info(args.pin)
            mode = args.mode or "static"
            if mode == "static":
                run_static_test(ctl, bcm, phys, gnd, label, auto_duration=args.duration)
            elif mode == "blink":
                run_blink_test(ctl, bcm, phys, gnd, label, duration_sec=args.duration)
            elif mode == "pwm":
                run_pwm_test(ctl, bcm, phys, gnd, label, duration_sec=args.duration)
            elif mode == "freq":
                run_freq_test(ctl, bcm, phys, gnd, label, freq_hz=args.freq, duration_sec=args.duration)
            return 0

        interactive_menu(ctl)
    finally:
        ctl.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())

