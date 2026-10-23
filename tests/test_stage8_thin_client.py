import math
import time
import pytest
import numpy as np

from manipulator3.hardware import (
    MockGPIOBackend,
    StepperDriverA4988,
    ServoConfig,
    ServoController,
    ToFReaderVL53L0X,
    PIN_STEP,
    PIN_DIR,
    PIN_SERVO_SHOULDER_A,
    PIN_SERVO_SHOULDER_B,
    PIN_SERVO_ELBOW,
    PIN_SERVO_WRIST_PITCH,
    PIN_SERVO_WRIST_ROLL,
    PIN_SERVO_GRIPPER,
)
from manipulator3.thin_client import ThinClient
from manipulator3.zenoh_client import HostClient


def test_stepper_a4988_step_calculation_and_direction():
    """8.4.3 Verify stepper angle-to-steps conversion and DIR pin logic."""
    gpio = MockGPIOBackend()
    stepper = StepperDriverA4988(gpio, steps_per_rev=200, microsteps=16)

    # 200 * 16 = 3200 steps per rev (2*pi rad) -> ~509.2958 steps/rad
    assert stepper.total_steps_per_rev == 3200
    assert abs(stepper.steps_per_rad - (3200.0 / (2.0 * math.pi))) < 1e-4

    # 1. Zero angle
    assert stepper.rad_to_steps(0.0) == 0

    # 2. Positive 90 deg (pi/2)
    steps_90 = stepper.rad_to_steps(math.pi / 2.0)
    assert steps_90 == 800

    # 3. Negative 90 deg (-pi/2)
    steps_neg_90 = stepper.rad_to_steps(-math.pi / 2.0)
    assert steps_neg_90 == -800

    # Test stepping forward
    stepper.set_target_angle(math.pi / 2.0)
    assert stepper.target_steps == 800
    stepper.step_once()
    assert gpio.digital_states[PIN_DIR] == 1
    assert stepper.current_steps == 1

    # Test stepping backward
    stepper.set_target_angle(-math.pi / 2.0)
    assert stepper.target_steps == -800
    stepper.step_once()
    assert gpio.digital_states[PIN_DIR] == 0
    assert stepper.current_steps == 0


def test_servo_pulse_mapping_and_shoulder_mechanical_reverse():
    """8.4.2 Verify radian-to-pulse conversion and dual shoulder servo opposed reverse."""
    gpio = MockGPIOBackend()
    servos = ServoController(gpio)

    # Center angle (0 rad): both should be 1500 us (7.5% duty at 50Hz)
    pulse_a, pulse_b = servos.set_shoulder_angle(0.0)
    assert abs(pulse_a - 1500.0) < 1.0
    assert abs(pulse_b - 1500.0) < 1.0
    duty_a = gpio.pwm_states[PIN_SERVO_SHOULDER_A]["duty"]
    duty_b = gpio.pwm_states[PIN_SERVO_SHOULDER_B]["duty"]
    assert abs(duty_a - 7.5) < 0.1
    assert abs(duty_b - 7.5) < 0.1

    # Positive shoulder rotation (+0.5 rad):
    # Shoulder A increases pulse, Shoulder B decreases pulse by the exact same amount
    pulse_a, pulse_b = servos.set_shoulder_angle(0.5)
    delta_a = pulse_a - 1500.0
    delta_b = 1500.0 - pulse_b
    assert delta_a > 100.0
    assert abs(delta_a - delta_b) < 1.0  # Perfectly opposed symmetric drive

    # Negative shoulder rotation (-0.5 rad):
    pulse_a_neg, pulse_b_neg = servos.set_shoulder_angle(-0.5)
    assert abs((pulse_a_neg - 1500.0) - (1500.0 - pulse_a)) < 1.0
    assert abs((pulse_b_neg - 1500.0) - (1500.0 - pulse_b)) < 1.0


def test_servo_angle_clamping():
    """8.4.2 Verify pulse clamping within safe servo limits (500-2500 us)."""
    cfg = ServoConfig(pin=10, min_angle_rad=-1.5708, max_angle_rad=1.5708)
    # Excessive angles
    assert cfg.angle_to_pulse_us(10.0) == 2500.0
    assert cfg.angle_to_pulse_us(-10.0) == 500.0


def test_pigpio_backend_mock_integration(monkeypatch):
    """Verify PiGPIOBackend DMA pulse setting with mocked pigpio library."""
    from unittest.mock import MagicMock
    from manipulator3.hardware import PiGPIOBackend

    mock_pi = MagicMock()
    mock_pi.connected = True
    mock_pigpio_mod = MagicMock()
    mock_pigpio_mod.pi.return_value = mock_pi
    mock_pigpio_mod.OUTPUT = 1

    monkeypatch.setattr("manipulator3.hardware._load_optional_module", lambda name: mock_pigpio_mod if name == "pigpio" else None)

    backend = PiGPIOBackend()
    backend.set_servo_pulse(18, 1500.0)
    mock_pi.set_servo_pulsewidth.assert_called_with(18, 1500)

    backend.stop_pwm(18)
    mock_pi.set_servo_pulsewidth.assert_called_with(18, 0)


def test_tof_reader_mock_and_range():
    """8.3.1 Verify ToF distance reader returns valid range data."""
    tof = ToFReaderVL53L0X(mock=True)
    tof.set_mock_distance(123.4)
    dist, valid = tof.read_distance_mm()
    assert abs(dist - 123.4) < 1e-3
    assert valid is True

    tof.set_mock_distance(-1.0)
    dist_inv, valid_inv = tof.read_distance_mm()
    assert valid_inv is False


def test_thin_client_watchdog_cutoff():
    """8.4.3 & 8.4.4 Verify emergency cutoff triggers when watchdog timeout expires."""
    gpio = MockGPIOBackend()
    # 0.1s watchdog timeout for fast unit testing
    client = ThinClient(gpio=gpio, force_mock_hardware=True, watchdog_timeout_s=0.1)
    client.start()

    try:
        # Initial command
        client.apply_command([0.1, 0.2, -0.3, 0.4, 0.0, -0.2])
        assert client._emergency_stop is False

        # Wait longer than watchdog timeout
        time.sleep(0.25)

        # Watchdog should have triggered
        assert client._emergency_stop is True
        # Servos should be stopped (duty = 0.0)
        assert gpio.pwm_states[PIN_SERVO_SHOULDER_A]["duty"] == 0.0
        assert gpio.pwm_states[PIN_SERVO_GRIPPER]["duty"] == 0.0

        # Resend command should clear emergency stop
        client.apply_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert client._emergency_stop is False
    finally:
        client.close()


def test_thin_client_zenoh_loop():
    """8.1.1 & 8.2.1 Verify ThinClient streams camera and ToF over Zenoh to HostClient."""
    gpio = MockGPIOBackend()
    thin = ThinClient(
        gpio=gpio,
        force_mock_hardware=True,
        camera_fps=25,
        tof_rate_hz=25.0,
        watchdog_timeout_s=5.0,
    )
    thin.tof.set_mock_distance(85.0)
    thin.start()

    host = HostClient()
    try:
        # Wait for data stream
        assert host.wait_for_data(timeout=3.0) is True

        # Read camera frame
        frame_data = host.get_camera_frame(decode=True)
        assert frame_data is not None
        bgr, _ = frame_data
        assert bgr.shape == (480, 640, 3)

        # Read ToF range with short poll
        tof_data = None
        for _ in range(30):
            tof_data = host.get_tof_range()
            if tof_data is not None:
                break
            time.sleep(0.05)
        assert tof_data is not None
        assert abs(tof_data.distance_mm - 85.0) < 1e-2

        # Send joint command from host to thin client
        cmd_angles = [0.15, -0.2, 0.3, -0.1, 0.0, -0.4]
        host.send_joints_command(cmd_angles)

        # Wait for thin client to receive and update
        for _ in range(30):
            if abs(thin.servos.current_angles["revolute_2"] - (-0.2)) < 1e-4:
                break
            time.sleep(0.05)
        assert thin.stepper.target_steps == thin.stepper.rad_to_steps(0.15)
        assert abs(thin.servos.current_angles["revolute_2"] - (-0.2)) < 1e-4

        # Verify joint state feedback
        state = None
        for _ in range(30):
            state = host.get_joints_state()
            if state is not None:
                break
            time.sleep(0.05)
        assert state is not None
        assert "revolute_1" in state.positions
        assert "revolute_7" in state.positions
    finally:
        host.close()
        thin.close()
