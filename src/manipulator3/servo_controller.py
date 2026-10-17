import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np


@dataclass
class ServoConfig:
    channel: int
    name: str
    min_angle_rad: float
    max_angle_rad: float
    min_pulse_us: float = 500.0
    max_pulse_us: float = 2500.0
    zero_pulse_us: float = 1500.0
    inverted: bool = False
    calibration_offset_us: float = 0.0

    def angle_to_pulse_us(self, angle_rad: float) -> float:
        """Convert angle in radians to PWM pulse width in microseconds."""
        clamped_angle = float(np.clip(angle_rad, self.min_angle_rad, self.max_angle_rad))
        angle_span = self.max_angle_rad - self.min_angle_rad
        pulse_span = self.max_pulse_us - self.min_pulse_us

        if angle_span <= 0:
            return self.zero_pulse_us + self.calibration_offset_us

        scale = pulse_span / angle_span
        sign = -1.0 if self.inverted else 1.0
        target_pulse = (self.zero_pulse_us + self.calibration_offset_us) + sign * clamped_angle * scale
        return float(np.clip(target_pulse, self.min_pulse_us, self.max_pulse_us))

    def pulse_us_to_angle(self, pulse_us: float) -> float:
        """Convert PWM pulse width in microseconds back to angle in radians."""
        clamped_pulse = float(np.clip(pulse_us, self.min_pulse_us, self.max_pulse_us))
        angle_span = self.max_angle_rad - self.min_angle_rad
        pulse_span = self.max_pulse_us - self.min_pulse_us

        if pulse_span <= 0:
            return 0.0

        scale = angle_span / pulse_span
        sign = -1.0 if self.inverted else 1.0
        delta_pulse = clamped_pulse - (self.zero_pulse_us + self.calibration_offset_us)
        angle = sign * delta_pulse * scale
        return float(np.clip(angle, self.min_angle_rad, self.max_angle_rad))


DEFAULT_SERVO_CONFIGS: Dict[str, ServoConfig] = {
    "revolute_1": ServoConfig(
        channel=0,
        name="revolute_1",
        min_angle_rad=-3.14159,
        max_angle_rad=3.14159,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        zero_pulse_us=1500.0,
    ),
    "revolute_2": ServoConfig(
        channel=1,
        name="revolute_2",
        min_angle_rad=-1.5708,
        max_angle_rad=1.5708,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        zero_pulse_us=1500.0,
    ),
    "revolute_3": ServoConfig(
        channel=2,
        name="revolute_3",
        min_angle_rad=-1.5708,
        max_angle_rad=1.5708,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        zero_pulse_us=1500.0,
    ),
    "revolute_4": ServoConfig(
        channel=3,
        name="revolute_4",
        min_angle_rad=-1.5708,
        max_angle_rad=1.5708,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        zero_pulse_us=1500.0,
    ),
    "revolute_5": ServoConfig(
        channel=4,
        name="revolute_5",
        min_angle_rad=-1.5708,
        max_angle_rad=1.5708,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        zero_pulse_us=1500.0,
    ),
    "act_gripper": ServoConfig(
        channel=5,
        name="act_gripper",
        min_angle_rad=-0.6,
        max_angle_rad=0.6,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        zero_pulse_us=1500.0,
    ),
}

# Aliases for actuator naming compatibility
ACTUATOR_ALIAS_MAP: Dict[str, str] = {
    "act_revolute_1": "revolute_1",
    "act_revolute_2": "revolute_2",
    "act_revolute_3": "revolute_3",
    "act_revolute_4": "revolute_4",
    "act_revolute_5": "revolute_5",
    "revolute_7": "act_gripper",
    "gripper": "act_gripper",
    "gripper_joint": "act_gripper",
}


def pulse_us_to_pca_ticks(pulse_us: float, frequency_hz: float = 50.0) -> int:
    """Convert microsecond pulse to 12-bit PCA9685 tick (0..4095)."""
    period_us = 1e6 / frequency_hz
    ticks = int(round((pulse_us / period_us) * 4096.0))
    return int(np.clip(ticks, 0, 4095))


class MockSMBus:
    """Mock I2C bus for host testing without hardware."""

    def __init__(self, bus: int = 1) -> None:
        self.bus = bus
        self.registers: Dict[int, Dict[int, int]] = {}

    def write_byte_data(self, addr: int, cmd: int, val: int) -> None:
        if addr not in self.registers:
            self.registers[addr] = {}
        self.registers[addr][cmd] = val & 0xFF

    def read_byte_data(self, addr: int, cmd: int) -> int:
        return self.registers.get(addr, {}).get(cmd, 0)

    def write_i2c_block_data(self, addr: int, cmd: int, vals: List[int]) -> None:
        if addr not in self.registers:
            self.registers[addr] = {}
        for i, v in enumerate(vals):
            self.registers[addr][cmd + i] = v & 0xFF

    def close(self) -> None:
        pass


class PCA9685Driver:
    """Direct 16-channel PCA9685 I2C driver."""

    MODE1 = 0x00
    MODE2 = 0x01
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    ALL_LED_OFF_H = 0xFD

    def __init__(
        self,
        i2c_bus: int = 1,
        address: int = 0x40,
        frequency_hz: float = 50.0,
        mock: bool = False,
    ) -> None:
        self.address = address
        self.frequency_hz = frequency_hz
        self.mock = mock

        if mock:
            self.bus: Any = MockSMBus(i2c_bus)
        else:
            try:
                import smbus2
                self.bus = smbus2.SMBus(i2c_bus)
            except Exception:
                try:
                    import smbus
                    self.bus = smbus.SMBus(i2c_bus)
                except Exception:
                    self.bus = MockSMBus(i2c_bus)
                    self.mock = True

        self._init_hardware()

    def _init_hardware(self) -> None:
        """Initialize PCA9685 registers for 50Hz servo PWM."""
        try:
            self.bus.write_byte_data(self.address, self.MODE1, 0x00)
            self.set_pwm_frequency(self.frequency_hz)
            self.bus.write_byte_data(self.address, self.MODE2, 0x04)
        except Exception:
            pass

    def set_pwm_frequency(self, freq_hz: float) -> None:
        """Calculate and set prescale value for target PWM frequency."""
        self.frequency_hz = freq_hz
        prescale = int(round(25000000.0 / (4096.0 * freq_hz))) - 1
        prescale = int(np.clip(prescale, 3, 255))

        old_mode = self.bus.read_byte_data(self.address, self.MODE1)
        sleep_mode = (old_mode & 0x7F) | 0x10
        self.bus.write_byte_data(self.address, self.MODE1, sleep_mode)
        self.bus.write_byte_data(self.address, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.address, self.MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.MODE1, old_mode | 0xA1)

    def set_channel_ticks(self, channel: int, on_tick: int, off_tick: int) -> None:
        """Write 12-bit ON and OFF ticks for specified channel."""
        if not (0 <= channel <= 15):
            raise ValueError(f"Channel {channel} out of range (0..15).")
        reg_base = self.LED0_ON_L + 4 * channel
        data = [
            on_tick & 0xFF,
            (on_tick >> 8) & 0xFF,
            off_tick & 0xFF,
            (off_tick >> 8) & 0xFF,
        ]
        self.bus.write_i2c_block_data(self.address, reg_base, data)

    def set_channel_pulse_us(self, channel: int, pulse_us: float) -> None:
        """Set PWM pulse width in microseconds on specified channel."""
        ticks = pulse_us_to_pca_ticks(pulse_us, self.frequency_hz)
        self.set_channel_ticks(channel, 0, ticks)

    def disable_all(self) -> None:
        """Emergency cutoff: shut off all PWM outputs immediately."""
        try:
            self.bus.write_byte_data(self.address, self.ALL_LED_OFF_H, 0x10)
        except Exception:
            pass

    def close(self) -> None:
        self.disable_all()
        if hasattr(self.bus, "close"):
            self.bus.close()


class ServoController:
    """Manages multi-axis servo angle mapping, I2C driver, and safety watchdog."""

    def __init__(
        self,
        configs: Optional[Dict[str, ServoConfig]] = None,
        i2c_bus: int = 1,
        pca_address: int = 0x40,
        mock: bool = False,
        watchdog_timeout_sec: float = 1.5,
    ) -> None:
        self.configs = configs or DEFAULT_SERVO_CONFIGS
        self.driver = PCA9685Driver(i2c_bus=i2c_bus, address=pca_address, mock=mock)
        self.watchdog_timeout_sec = watchdog_timeout_sec

        self.current_angles: Dict[str, float] = {k: 0.0 for k in self.configs}
        self.last_command_time: float = time.time()
        self.is_cutoff: bool = False

    def canonical_name(self, name: str) -> Optional[str]:
        """Resolve joint name or alias to canonical config key."""
        if name in self.configs:
            return name
        alias = ACTUATOR_ALIAS_MAP.get(name)
        if alias in self.configs:
            return alias
        return None

    def apply_angles(self, angles: Union[Dict[str, float], List[float], np.ndarray]) -> Dict[str, float]:
        """Convert input angles to pulses and write to servo hardware."""
        self.last_command_time = time.time()
        self.is_cutoff = False

        if isinstance(angles, (list, tuple, np.ndarray)):
            ordered_keys = list(self.configs.keys())
            for idx, val in enumerate(angles):
                if idx < len(ordered_keys):
                    self._set_single_joint(ordered_keys[idx], float(val))
        elif isinstance(angles, dict):
            for raw_name, val in angles.items():
                canon = self.canonical_name(raw_name)
                if canon is not None:
                    self._set_single_joint(canon, float(val))

        return dict(self.current_angles)

    def _set_single_joint(self, joint_name: str, angle_rad: float) -> None:
        cfg = self.configs[joint_name]
        pulse_us = cfg.angle_to_pulse_us(angle_rad)
        self.driver.set_channel_pulse_us(cfg.channel, pulse_us)
        self.current_angles[joint_name] = cfg.pulse_us_to_angle(pulse_us)

    def check_watchdog(self) -> bool:
        """Trigger emergency cutoff if command stream is lost. Returns True if cutoff triggered."""
        if self.watchdog_timeout_sec <= 0:
            return False

        if not self.is_cutoff and (time.time() - self.last_command_time > self.watchdog_timeout_sec):
            self.emergency_stop()
            return True
        return self.is_cutoff

    def emergency_stop(self) -> None:
        """Trigger emergency cutoff to prevent servo strain."""
        self.is_cutoff = True
        self.driver.disable_all()

    def close(self) -> None:
        self.emergency_stop()
        self.driver.close()
