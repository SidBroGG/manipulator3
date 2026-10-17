import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Hardware GPIO Pinout as defined in GEMINI.md
PIN_STEP = 17
PIN_DIR = 27
PIN_SERVO_SHOULDER_A = 26
PIN_SERVO_SHOULDER_B = 29
PIN_SERVO_ELBOW = 13
PIN_SERVO_WRIST_PITCH = 6
PIN_SERVO_WRIST_ROLL = 5
PIN_SERVO_GRIPPER = 22

I2C_SDA = 2
I2C_SCL = 3
I2C_BUS = 1
VL53L0X_I2C_ADDR = 0x29

DEFAULT_PWM_FREQ_HZ = 50.0  # Standard analog/digital servo frequency


class GPIOBackend:
    """Interface for GPIO and PWM control."""

    def setup_output(self, pin: int) -> None:
        raise NotImplementedError

    def write_digital(self, pin: int, value: int) -> None:
        raise NotImplementedError

    def start_pwm(self, pin: int, freq_hz: float, duty_percent: float) -> None:
        raise NotImplementedError

    def set_pwm_duty(self, pin: int, duty_percent: float) -> None:
        raise NotImplementedError

    def stop_pwm(self, pin: int) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError


class MockGPIOBackend(GPIOBackend):
    """In-memory mock GPIO backend for testing and simulation on non-RPi hosts."""

    def __init__(self) -> None:
        self.pin_modes: Dict[int, str] = {}
        self.digital_states: Dict[int, int] = {}
        self.pwm_states: Dict[int, Dict[str, float]] = {}

    def setup_output(self, pin: int) -> None:
        self.pin_modes[pin] = "OUT"
        self.digital_states[pin] = 0

    def write_digital(self, pin: int, value: int) -> None:
        self.digital_states[pin] = int(value)

    def start_pwm(self, pin: int, freq_hz: float, duty_percent: float) -> None:
        self.pwm_states[pin] = {"freq": freq_hz, "duty": duty_percent, "running": True}

    def set_pwm_duty(self, pin: int, duty_percent: float) -> None:
        if pin in self.pwm_states:
            self.pwm_states[pin]["duty"] = duty_percent
            self.pwm_states[pin]["running"] = duty_percent > 0.0

    def stop_pwm(self, pin: int) -> None:
        if pin in self.pwm_states:
            self.pwm_states[pin]["running"] = False
            self.pwm_states[pin]["duty"] = 0.0

    def cleanup(self) -> None:
        self.digital_states.clear()
        self.pwm_states.clear()


class LGPIOBackend(GPIOBackend):
    """Hardware GPIO backend using lgpio for Raspberry Pi OS (Bookworm)."""

    def __init__(self, chip: int = 0) -> None:
        import lgpio
        self.lgpio = lgpio
        self.handle = lgpio.gpiochip_open(chip)
        self.active_pins: List[int] = []

    def setup_output(self, pin: int) -> None:
        self.lgpio.gpio_claim_output(self.handle, pin)
        if pin not in self.active_pins:
            self.active_pins.append(pin)

    def write_digital(self, pin: int, value: int) -> None:
        self.lgpio.gpio_write(self.handle, pin, int(value))

    def start_pwm(self, pin: int, freq_hz: float, duty_percent: float) -> None:
        self.lgpio.tx_pwm(self.handle, pin, int(freq_hz), float(duty_percent))

    def set_pwm_duty(self, pin: int, duty_percent: float) -> None:
        self.lgpio.tx_pwm(self.handle, pin, int(DEFAULT_PWM_FREQ_HZ), float(duty_percent))

    def stop_pwm(self, pin: int) -> None:
        self.lgpio.tx_pwm(self.handle, pin, 0, 0)

    def cleanup(self) -> None:
        for pin in self.active_pins:
            try:
                self.stop_pwm(pin)
                self.lgpio.gpio_free(self.handle, pin)
            except Exception:
                pass
        try:
            self.lgpio.gpiochip_close(self.handle)
        except Exception:
            pass


class RPiGPIOBackend(GPIOBackend):
    """Hardware GPIO backend using RPi.GPIO (Bullseye / legacy)."""

    def __init__(self) -> None:
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        self.GPIO.setmode(self.GPIO.BCM)
        self.GPIO.setwarnings(False)
        self.pwm_instances: Dict[int, Any] = {}

    def setup_output(self, pin: int) -> None:
        self.GPIO.setup(pin, self.GPIO.OUT)

    def write_digital(self, pin: int, value: int) -> None:
        self.GPIO.output(pin, self.GPIO.HIGH if value else self.GPIO.LOW)

    def start_pwm(self, pin: int, freq_hz: float, duty_percent: float) -> None:
        if pin not in self.pwm_instances:
            pwm = self.GPIO.PWM(pin, freq_hz)
            pwm.start(duty_percent)
            self.pwm_instances[pin] = pwm
        else:
            self.pwm_instances[pin].ChangeDutyCycle(duty_percent)

    def set_pwm_duty(self, pin: int, duty_percent: float) -> None:
        if pin in self.pwm_instances:
            self.pwm_instances[pin].ChangeDutyCycle(duty_percent)

    def stop_pwm(self, pin: int) -> None:
        if pin in self.pwm_instances:
            self.pwm_instances[pin].stop()
            del self.pwm_instances[pin]

    def cleanup(self) -> None:
        for pwm in list(self.pwm_instances.values()):
            try:
                pwm.stop()
            except Exception:
                pass
        self.pwm_instances.clear()
        try:
            self.GPIO.cleanup()
        except Exception:
            pass


def get_default_gpio_backend(force_mock: bool = False) -> GPIOBackend:
    """Return hardware GPIO backend if available, otherwise MockGPIOBackend."""
    if force_mock:
        return MockGPIOBackend()

    try:
        return LGPIOBackend()
    except Exception:
        pass

    try:
        return RPiGPIOBackend()
    except Exception:
        pass

    logger.info("[INFO] RPi GPIO libraries not detected. Using MockGPIOBackend.")
    return MockGPIOBackend()


class StepperDriverA4988:
    """A4988 Stepper driver for NEMA 17 azimuth base (revolute_1)."""

    def __init__(
        self,
        gpio: GPIOBackend,
        step_pin: int = PIN_STEP,
        dir_pin: int = PIN_DIR,
        steps_per_rev: int = 200,
        microsteps: int = 16,
        min_angle_rad: float = -math.pi,
        max_angle_rad: float = math.pi,
    ) -> None:
        self.gpio = gpio
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.total_steps_per_rev = steps_per_rev * microsteps
        self.steps_per_rad = self.total_steps_per_rev / (2.0 * math.pi)
        self.min_angle_rad = min_angle_rad
        self.max_angle_rad = max_angle_rad

        self.current_steps: int = 0
        self.target_steps: int = 0
        self.current_angle_rad: float = 0.0

        self.gpio.setup_output(self.step_pin)
        self.gpio.setup_output(self.dir_pin)
        self.gpio.write_digital(self.dir_pin, 0)
        self.gpio.write_digital(self.step_pin, 0)

    def rad_to_steps(self, angle_rad: float) -> int:
        clamped = max(self.min_angle_rad, min(self.max_angle_rad, angle_rad))
        return int(round(clamped * self.steps_per_rad))

    def steps_to_rad(self, steps: int) -> float:
        return float(steps) / self.steps_per_rad

    def set_target_angle(self, angle_rad: float) -> int:
        """Set target angle and update target_steps."""
        self.target_steps = self.rad_to_steps(angle_rad)
        return self.target_steps

    def step_once(self) -> bool:
        """Perform a single step towards target. Returns True if a step occurred."""
        diff = self.target_steps - self.current_steps
        if diff == 0:
            return False

        direction = 1 if diff > 0 else 0
        self.gpio.write_digital(self.dir_pin, direction)
        self.gpio.write_digital(self.step_pin, 1)
        self.gpio.write_digital(self.step_pin, 0)

        self.current_steps += 1 if diff > 0 else -1
        self.current_angle_rad = self.steps_to_rad(self.current_steps)
        return True

    def step_towards_target(self, max_steps: Optional[int] = None) -> int:
        """Step towards target up to max_steps. Returns number of steps performed."""
        count = 0
        while self.current_steps != self.target_steps:
            if max_steps is not None and count >= max_steps:
                break
            self.step_once()
            count += 1
        return count

    def emergency_stop(self) -> None:
        """Abort motion and freeze at current position."""
        self.target_steps = self.current_steps
        self.gpio.write_digital(self.step_pin, 0)


@dataclass
class ServoConfig:
    pin: int
    min_pulse_us: float = 500.0
    max_pulse_us: float = 2500.0
    center_pulse_us: float = 1500.0
    min_angle_rad: float = -math.pi / 2.0
    max_angle_rad: float = math.pi / 2.0
    reversed: bool = False
    zero_offset_rad: float = 0.0

    def angle_to_pulse_us(self, angle_rad: float) -> float:
        """Convert joint angle in radians to PWM pulse width in microseconds."""
        clamped_angle = max(self.min_angle_rad, min(self.max_angle_rad, angle_rad))
        eff_angle = clamped_angle + self.zero_offset_rad
        if self.reversed:
            eff_angle = -eff_angle

        # Linear mapping between angle limits and pulse width limits
        span_rad = self.max_angle_rad - self.min_angle_rad
        norm = (eff_angle - self.min_angle_rad) / span_rad if span_rad > 0 else 0.5
        pulse = self.min_pulse_us + norm * (self.max_pulse_us - self.min_pulse_us)
        return float(max(self.min_pulse_us, min(self.max_pulse_us, pulse)))

    def pulse_us_to_duty_cycle(self, pulse_us: float, freq_hz: float = DEFAULT_PWM_FREQ_HZ) -> float:
        """Convert microsecond pulse to PWM duty cycle percentage (0-100%)."""
        period_us = 1_000_000.0 / freq_hz
        return (pulse_us / period_us) * 100.0


class ServoController:
    """Manages the 6 PWM servos for arm joints (Shoulder A/B, Elbow, Wrist, Gripper)."""

    def __init__(self, gpio: GPIOBackend, freq_hz: float = DEFAULT_PWM_FREQ_HZ) -> None:
        self.gpio = gpio
        self.freq_hz = freq_hz

        # Hardware mapping: Servo 2 (Shoulder B) is mechanically reversed to Servo 1 (Shoulder A)
        self.configs: Dict[str, ServoConfig] = {
            "shoulder_a": ServoConfig(
                pin=PIN_SERVO_SHOULDER_A,
                reversed=False,
                min_angle_rad=-1.5708,
                max_angle_rad=1.5708,
            ),
            "shoulder_b": ServoConfig(
                pin=PIN_SERVO_SHOULDER_B,
                reversed=True,  # Mechanical reverse (opposed rotors)
                min_angle_rad=-1.5708,
                max_angle_rad=1.5708,
            ),
            "elbow": ServoConfig(
                pin=PIN_SERVO_ELBOW,
                reversed=False,
                min_angle_rad=-1.5708,
                max_angle_rad=1.5708,
            ),
            "wrist_pitch": ServoConfig(
                pin=PIN_SERVO_WRIST_PITCH,
                reversed=False,
                min_angle_rad=-1.5708,
                max_angle_rad=1.5708,
            ),
            "wrist_roll": ServoConfig(
                pin=PIN_SERVO_WRIST_ROLL,
                reversed=False,
                min_angle_rad=-1.5708,
                max_angle_rad=1.5708,
            ),
            "gripper": ServoConfig(
                pin=PIN_SERVO_GRIPPER,
                reversed=False,
                min_angle_rad=-0.6,
                max_angle_rad=0.6,
            ),
        }

        self.current_angles: Dict[str, float] = {
            "revolute_2": 0.0,
            "revolute_3": 0.0,
            "revolute_4": 0.0,
            "revolute_5": 0.0,
            "revolute_7": 0.0,
        }

        # Setup pins and start at center pulse (1500 us)
        for cfg in self.configs.values():
            self.gpio.setup_output(cfg.pin)
            duty = cfg.pulse_us_to_duty_cycle(cfg.center_pulse_us, self.freq_hz)
            self.gpio.start_pwm(cfg.pin, self.freq_hz, duty)

    def set_shoulder_angle(self, angle_rad: float) -> Tuple[float, float]:
        """Apply angle to Shoulder A and mechanically reversed Shoulder B. Returns (pulse_a, pulse_b)."""
        cfg_a = self.configs["shoulder_a"]
        cfg_b = self.configs["shoulder_b"]

        pulse_a = cfg_a.angle_to_pulse_us(angle_rad)
        pulse_b = cfg_b.angle_to_pulse_us(angle_rad)

        duty_a = cfg_a.pulse_us_to_duty_cycle(pulse_a, self.freq_hz)
        duty_b = cfg_b.pulse_us_to_duty_cycle(pulse_b, self.freq_hz)

        self.gpio.set_pwm_duty(cfg_a.pin, duty_a)
        self.gpio.set_pwm_duty(cfg_b.pin, duty_b)
        self.current_angles["revolute_2"] = angle_rad
        return pulse_a, pulse_b

    def set_joint_angle(self, joint_name: str, angle_rad: float) -> float:
        """Set joint angle in radians. Returns pulse width in microseconds."""
        if joint_name in ("revolute_2", "shoulder"):
            p_a, _ = self.set_shoulder_angle(angle_rad)
            return p_a

        joint_key_map = {
            "revolute_3": ("elbow", "revolute_3"),
            "elbow": ("elbow", "revolute_3"),
            "revolute_4": ("wrist_pitch", "revolute_4"),
            "wrist_pitch": ("wrist_pitch", "revolute_4"),
            "revolute_5": ("wrist_roll", "revolute_5"),
            "wrist_roll": ("wrist_roll", "revolute_5"),
            "revolute_7": ("gripper", "revolute_7"),
            "gripper": ("gripper", "revolute_7"),
            "act_gripper": ("gripper", "revolute_7"),
        }

        if joint_name not in joint_key_map:
            return 0.0

        cfg_key, state_key = joint_key_map[joint_name]
        cfg = self.configs[cfg_key]
        pulse = cfg.angle_to_pulse_us(angle_rad)
        duty = cfg.pulse_us_to_duty_cycle(pulse, self.freq_hz)
        self.gpio.set_pwm_duty(cfg.pin, duty)
        self.current_angles[state_key] = angle_rad
        return pulse

    def set_joint_angles(self, angles: Union[Dict[str, float], List[float]]) -> None:
        """Apply joint angles from a list [rev1..rev5, gripper] or dictionary."""
        if isinstance(angles, (list, tuple)):
            # list index 0 is base (handled by stepper), 1..4 arm servos, 5 gripper
            if len(angles) > 1:
                self.set_shoulder_angle(float(angles[1]))
            if len(angles) > 2:
                self.set_joint_angle("revolute_3", float(angles[2]))
            if len(angles) > 3:
                self.set_joint_angle("revolute_4", float(angles[3]))
            if len(angles) > 4:
                self.set_joint_angle("revolute_5", float(angles[4]))
            if len(angles) > 5:
                self.set_joint_angle("revolute_7", float(angles[5]))
        elif isinstance(angles, dict):
            for k, v in angles.items():
                if k not in ("revolute_1", "base"):
                    self.set_joint_angle(k, float(v))

    def emergency_stop(self) -> None:
        """Cut off PWM duty cycle to safely release servo torque."""
        for cfg in self.configs.values():
            self.gpio.set_pwm_duty(cfg.pin, 0.0)


class ToFReaderVL53L0X:
    """I2C Reader for VL53L0X laser distance sensor on GPIO 2 (SDA) / GPIO 3 (SCL)."""

    def __init__(self, bus_id: int = I2C_BUS, address: int = VL53L0X_I2C_ADDR, mock: bool = False) -> None:
        self.bus_id = bus_id
        self.address = address
        self.mock = mock
        self._sensor = None
        self._mock_distance_mm = 150.0

        if not self.mock:
            try:
                import vl53l0x
                self._sensor = vl53l0x.VL53L0X(i2c_bus=bus_id, i2c_address=address)
                self._sensor.open()
                self._sensor.start_ranging(vl53l0x.Vl53l0xAccuracyMode.BETTER)
            except Exception:
                logger.info("[INFO] VL53L0X hardware library not available. Falling back to mock mode.")
                self.mock = True

    def set_mock_distance(self, distance_mm: float) -> None:
        self._mock_distance_mm = float(distance_mm)

    def read_distance_mm(self) -> Tuple[float, bool]:
        """Return (distance_mm, is_valid)."""
        if self.mock or self._sensor is None:
            val = self._mock_distance_mm
            return val, val > 0.0

        try:
            dist = float(self._sensor.get_distance())
            valid = 20.0 <= dist <= 2000.0
            return dist if valid else -1.0, valid
        except Exception:
            return -1.0, False

    def close(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop_ranging()
                self._sensor.close()
            except Exception:
                pass


class CameraStreamerOV5647:
    """OV5647 CSI Camera streamer using Picamera2 (or OpenCV fallback)."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 25,
        mock: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.mock = mock
        self.picam2 = None
        self.cap = None

        if not self.mock:
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"FrameRate": self.fps},
                )
                self.picam2.configure(config)
                self.picam2.start()
            except Exception:
                try:
                    import cv2
                    self.cap = cv2.VideoCapture(0)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    if not self.cap.isOpened():
                        self.mock = True
                except Exception:
                    self.mock = True

    def capture_frame(self) -> "np.ndarray":
        """Capture BGR frame as numpy ndarray."""
        import cv2
        import numpy as np

        if self.mock:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "OV5647 CSI STREAM (MOCK)",
                (20, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            return frame

        if self.picam2 is not None:
            rgb = self.picam2.capture_array()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame

        # Fallback to black frame
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
