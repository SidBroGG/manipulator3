import math
import pytest
from manipulator3.hardware import (
    MockGPIOBackend,
    PIN_SERVO_SHOULDER_A,
    PIN_SERVO_SHOULDER_B,
    PIN_SERVO_ELBOW,
    PIN_SERVO_GRIPPER,
    PIN_STEP,
    PIN_DIR,
)
from manipulator3.servo_tester import ServoTester


def test_motor_alias_resolution():
    """Verify alias mapping for motor numbers, names, and case insensitivity."""
    tester = ServoTester(force_mock=True)

    assert tester.resolve_motor(1) == "shoulder_a"
    assert tester.resolve_motor("1") == "shoulder_a"
    assert tester.resolve_motor("SHOULDER_A") == "shoulder_a"
    assert tester.resolve_motor(2) == "shoulder_b"
    assert tester.resolve_motor("shoulder") == "shoulder"
    assert tester.resolve_motor("s") == "shoulder"
    assert tester.resolve_motor(3) == "elbow"
    assert tester.resolve_motor("elbow") == "elbow"
    assert tester.resolve_motor(4) == "wrist_pitch"
    assert tester.resolve_motor(5) == "wrist_roll"
    assert tester.resolve_motor(6) == "gripper"
    assert tester.resolve_motor("jaw") == "gripper"
    assert tester.resolve_motor(0) == "base"
    assert tester.resolve_motor("stepper") == "base"

    with pytest.raises(ValueError):
        tester.resolve_motor("unknown_joint_99")


def test_servo_angle_pulse_and_clamping():
    """Verify degree/radian conversion, PWM pulse, and limit clamping."""
    gpio = MockGPIOBackend()
    tester = ServoTester(gpio=gpio, force_mock=True)

    # 1. Center elbow (0 deg) -> 1500 us (7.5% duty at 50Hz)
    res = tester.set_motor_angle("elbow", 0.0, in_degrees=True)
    assert res["angle_deg"] == 0.0
    assert abs(res["pulse_us"] - 1500.0) < 1.0
    assert abs(res["duty_pct"] - 7.5) < 0.1
    assert not res["clamped"]
    assert gpio.pwm_states[PIN_SERVO_ELBOW]["duty"] == res["duty_pct"]

    # 2. Elbow +45 deg (pi/4 rad) -> 2000 us (10% duty)
    res_pos = tester.set_motor_angle("elbow", 45.0, in_degrees=True)
    assert abs(res_pos["pulse_us"] - 2000.0) < 2.0
    assert not res_pos["clamped"]

    # 3. Excessive angle: 120 deg should clamp to 90 deg
    res_clamped = tester.set_motor_angle("elbow", 120.0, in_degrees=True)
    assert res_clamped["clamped"]
    assert res_clamped["angle_deg"] == 90.0
    assert abs(res_clamped["pulse_us"] - 2500.0) < 1.0

    # 4. Gripper limits (-34.4 deg to +34.4 deg)
    res_grip_clamped = tester.set_motor_angle("gripper", 50.0, in_degrees=True)
    assert res_grip_clamped["clamped"]
    assert abs(res_grip_clamped["angle_deg"] - 34.38) < 0.1


def test_coupled_shoulder_vs_individual_shoulder_servos():
    """Verify coupled dual shoulder movement vs single servo calibration."""
    gpio = MockGPIOBackend()
    tester = ServoTester(gpio=gpio, force_mock=True)

    # Move dual shoulder: shoulder_a and shoulder_b move symmetrically opposed
    tester.set_motor_angle("shoulder", 30.0, in_degrees=True)
    pulse_a = gpio.pwm_states[PIN_SERVO_SHOULDER_A]["duty"]
    pulse_b = gpio.pwm_states[PIN_SERVO_SHOULDER_B]["duty"]
    assert pulse_a > 7.5
    assert pulse_b < 7.5
    assert abs((pulse_a - 7.5) - (7.5 - pulse_b)) < 0.01

    # Move individual shoulder_a for zero calibration without affecting shoulder_b
    prev_duty_b = gpio.pwm_states[PIN_SERVO_SHOULDER_B]["duty"]
    tester.set_motor_angle("shoulder_a", 0.0, in_degrees=True)
    assert abs(gpio.pwm_states[PIN_SERVO_SHOULDER_A]["duty"] - 7.5) < 0.01
    assert gpio.pwm_states[PIN_SERVO_SHOULDER_B]["duty"] == prev_duty_b


def test_base_stepper_movement():
    """Verify base stepper motor receives steps via A4988 driver."""
    gpio = MockGPIOBackend()
    tester = ServoTester(gpio=gpio, force_mock=True)

    res = tester.set_motor_angle("base", 90.0, in_degrees=True)
    assert res["angle_deg"] == 90.0
    assert tester.stepper.current_steps == 800
    assert gpio.digital_states[PIN_DIR] == 1


def test_relax_and_center_all():
    """Verify PWM duty is zeroed upon relax and centered upon center_all."""
    gpio = MockGPIOBackend()
    tester = ServoTester(gpio=gpio, force_mock=True)

    # Center all
    tester.center_all()
    assert abs(gpio.pwm_states[PIN_SERVO_ELBOW]["duty"] - 7.5) < 0.1

    # Relax single servo
    tester.relax("elbow")
    assert gpio.pwm_states[PIN_SERVO_ELBOW]["duty"] == 0.0
    assert gpio.pwm_states[PIN_SERVO_GRIPPER]["duty"] > 0.0

    # Relax all
    tester.relax()
    for pin in [PIN_SERVO_SHOULDER_A, PIN_SERVO_SHOULDER_B, PIN_SERVO_ELBOW, PIN_SERVO_GRIPPER]:
        assert gpio.pwm_states[pin]["duty"] == 0.0


def test_smooth_motion_and_from_angle():
    """Verify smooth interpolation moves motor through intermediate positions and accepts from_angle."""
    gpio = MockGPIOBackend()
    tester = ServoTester(gpio=gpio, force_mock=True)

    # 1. Move from explicit starting angle 0 to 30 smoothly
    res = tester.set_motor_angle("elbow", 30.0, in_degrees=True, smooth=True, speed_deg_per_s=300.0, from_angle=0.0)
    assert res["angle_deg"] == 30.0
    assert abs(res["pulse_us"] - 1833.3) < 2.0
    assert abs(gpio.pwm_states[PIN_SERVO_ELBOW]["duty"] - 9.16) < 0.1

    # 2. Smooth move back to 0 using tracked current position
    res_back = tester.set_motor_angle("elbow", 0.0, in_degrees=True, smooth=True, speed_deg_per_s=300.0)
    assert res_back["angle_deg"] == 0.0
    assert abs(res_back["pulse_us"] - 1500.0) < 1.0


def test_persisted_state_roundtrip(tmp_path, monkeypatch):
    """Verify state file maintains motor positions across independent process/tester instances."""
    state_file = str(tmp_path / "servo_state.json")
    monkeypatch.setattr("manipulator3.servo_tester.STATE_FILE_PATH", state_file)

    gpio1 = MockGPIOBackend()
    tester1 = ServoTester(gpio=gpio1, force_mock=True)
    tester1.set_motor_angle("wrist_pitch", 45.0, in_degrees=True)

    # New tester instance simulating subsequent CLI call
    gpio2 = MockGPIOBackend()
    tester2 = ServoTester(gpio=gpio2, force_mock=True)
    assert abs(tester2.current_angles_rad["wrist_pitch"] - math.radians(45.0)) < 1e-4

    # Smooth transition from remembered 45° to -15°
    res = tester2.set_motor_angle("wrist_pitch", -15.0, in_degrees=True, smooth=True, speed_deg_per_s=500.0)
    assert res["angle_deg"] == -15.0

