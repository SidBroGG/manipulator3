import time
import numpy as np
import pytest

from manipulator3.protocol import (
    ToFRange,
    JointsCommand,
    JointsState,
    RobotStatus,
)
from manipulator3.servo_controller import (
    ServoConfig,
    ServoController,
    PCA9685Driver,
    DEFAULT_SERVO_CONFIGS,
    pulse_us_to_pca_ticks,
)
from manipulator3.rpi_hardware import (
    CameraStreamer,
    VL53L0XReader,
    MockCamera,
)
from manipulator3.rpi_client import RPiThinClient
from manipulator3.zenoh_client import HostClient


def test_servo_config_angle_pulse_mapping():
    """Verify linear radian to microsecond pulse mapping, clamping, and inversion."""
    cfg = ServoConfig(
        channel=0,
        name="test_joint",
        min_angle_rad=-1.0,
        max_angle_rad=1.0,
        min_pulse_us=1000.0,
        max_pulse_us=2000.0,
        zero_pulse_us=1500.0,
    )

    # Center position
    assert cfg.angle_to_pulse_us(0.0) == pytest.approx(1500.0)
    assert cfg.pulse_us_to_angle(1500.0) == pytest.approx(0.0)

    # Positive limit
    assert cfg.angle_to_pulse_us(1.0) == pytest.approx(2000.0)
    assert cfg.pulse_us_to_angle(2000.0) == pytest.approx(1.0)

    # Negative limit
    assert cfg.angle_to_pulse_us(-1.0) == pytest.approx(1000.0)
    assert cfg.pulse_us_to_angle(1000.0) == pytest.approx(-1.0)

    # Clamping beyond limits
    assert cfg.angle_to_pulse_us(2.5) == pytest.approx(2000.0)
    assert cfg.angle_to_pulse_us(-3.0) == pytest.approx(1000.0)

    # Inverted configuration
    inv_cfg = ServoConfig(
        channel=1,
        name="inv_joint",
        min_angle_rad=-1.0,
        max_angle_rad=1.0,
        min_pulse_us=1000.0,
        max_pulse_us=2000.0,
        zero_pulse_us=1500.0,
        inverted=True,
    )
    assert inv_cfg.angle_to_pulse_us(1.0) == pytest.approx(1000.0)
    assert inv_cfg.angle_to_pulse_us(-1.0) == pytest.approx(2000.0)

    # Calibration offset
    cal_cfg = ServoConfig(
        channel=2,
        name="cal_joint",
        min_angle_rad=-1.0,
        max_angle_rad=1.0,
        min_pulse_us=500.0,
        max_pulse_us=2500.0,
        zero_pulse_us=1500.0,
        calibration_offset_us=50.0,
    )
    assert cal_cfg.angle_to_pulse_us(0.0) == pytest.approx(1550.0)


def test_pulse_us_to_pca_ticks():
    """Verify conversion of microsecond pulse to 12-bit PCA9685 ticks at 50Hz."""
    # 50 Hz -> 20,000 us period -> 4096 ticks
    # 1500 us -> 1500 / 20000 * 4096 = 307.2 -> 307
    assert pulse_us_to_pca_ticks(1500.0, 50.0) == 307
    # 1000 us -> 1000 / 20000 * 4096 = 204.8 -> 205
    assert pulse_us_to_pca_ticks(1000.0, 50.0) == 205
    # 2000 us -> 2000 / 20000 * 4096 = 409.6 -> 410
    assert pulse_us_to_pca_ticks(2000.0, 50.0) == 410
    # Clamping
    assert pulse_us_to_pca_ticks(-100.0, 50.0) == 0
    assert pulse_us_to_pca_ticks(25000.0, 50.0) == 4095


def test_pca9685_driver_registers():
    """Verify PCA9685 register writes using mock SMBus."""
    driver = PCA9685Driver(i2c_bus=1, address=0x40, frequency_hz=50.0, mock=True)
    try:
        # Check prescale write for 50Hz (approx 121 / 0x79)
        prescale_val = driver.bus.read_byte_data(0x40, driver.PRESCALE)
        assert prescale_val in (121, 122)

        # Set pulse on channel 0
        driver.set_channel_pulse_us(0, 1500.0)
        ticks = pulse_us_to_pca_ticks(1500.0, 50.0)
        off_l = driver.bus.read_byte_data(0x40, driver.LED0_ON_L + 2)
        off_h = driver.bus.read_byte_data(0x40, driver.LED0_ON_L + 3)
        read_ticks = off_l | (off_h << 8)
        assert read_ticks == ticks

        # Emergency cutoff disables all channels
        driver.disable_all()
        cutoff_val = driver.bus.read_byte_data(0x40, driver.ALL_LED_OFF_H)
        assert cutoff_val == 0x10
    finally:
        driver.close()


def test_servo_controller_commands_and_watchdog():
    """Verify multi-joint command parsing, alias handling, and safety watchdog."""
    controller = ServoController(mock=True, watchdog_timeout_sec=0.2)
    try:
        # Command with dict using actuator aliases
        cmd_dict = {
            "act_revolute_1": 0.5,
            "revolute_2": -0.4,
            "act_gripper": 0.25,
        }
        res = controller.apply_angles(cmd_dict)
        assert res["revolute_1"] == pytest.approx(0.5, abs=0.02)
        assert res["revolute_2"] == pytest.approx(-0.4, abs=0.02)
        assert res["act_gripper"] == pytest.approx(0.25, abs=0.02)
        assert controller.is_cutoff is False

        # Watchdog before timeout
        assert controller.check_watchdog() is False

        # Wait for watchdog timeout
        time.sleep(0.3)
        assert controller.check_watchdog() is True
        assert controller.is_cutoff is True

        # Recovery on fresh command
        controller.apply_angles({"revolute_1": 0.0})
        assert controller.is_cutoff is False
    finally:
        controller.close()


def test_camera_and_tof_hardware_mocks():
    """Verify MockCamera and VL53L0XReader in mock mode."""
    # Camera
    cam = CameraStreamer(fps=30.0, mock=True)
    cam.start()
    try:
        time.sleep(0.1)
        frame_info = cam.get_latest_frame()
        assert frame_info is not None
        frame, ts = frame_info
        assert frame.shape == (480, 640, 3)
        assert ts > 0
    finally:
        cam.stop()

    # ToF
    tof = VL53L0XReader(poll_rate_hz=30.0, mock=True)
    tof.set_mock_distance(75.5, valid=True)
    tof.start()
    try:
        time.sleep(0.1)
        dist_mm, valid, ts = tof.get_latest_range()
        assert dist_mm == pytest.approx(75.5)
        assert valid is True
        assert ts > 0
    finally:
        tof.stop()


def test_rpi_thin_client_zenoh_integration():
    """Verify full communication loop between RPiThinClient and HostClient."""
    rpi = RPiThinClient(
        mock=True,
        camera_fps=25.0,
        jpeg_quality=70,
        tof_rate_hz=30.0,
        watchdog_timeout=1.0,
    )
    host = HostClient()

    try:
        rpi.tof.set_mock_distance(42.0, valid=True)
        rpi.start()

        # 1. Host waits for sensor stream
        ready = host.wait_for_data(timeout=3.0)
        assert ready is True

        # 2. Host receives camera frame
        t0 = time.time()
        frame_data = None
        while time.time() - t0 < 3.0:
            frame_data = host.get_camera_frame(decode=True)
            if frame_data is not None:
                break
            time.sleep(0.05)
        assert frame_data is not None
        bgr, frame_ts = frame_data
        assert bgr.shape == (480, 640, 3)

        # 3. Host receives ToF range
        tof_data = host.get_tof_range()
        assert tof_data is not None
        assert tof_data.distance_mm == pytest.approx(42.0)
        assert tof_data.valid is True

        # 4. Host sends joint command
        test_angles = {
            "act_revolute_1": 0.35,
            "act_revolute_2": -0.25,
            "act_gripper": 0.15,
        }
        host.send_joints_command(test_angles)
        time.sleep(0.15)

        # 5. Host receives joint feedback state
        state = host.get_joints_state()
        assert state is not None
        assert state.positions["revolute_1"] == pytest.approx(0.35, abs=0.02)
        assert state.positions["act_gripper"] == pytest.approx(0.15, abs=0.02)

        # 6. Host receives robot status
        status = host.get_status()
        assert status is not None
        assert status.connected is True
        assert status.state in ("RUNNING", "ACTIVE")

        # 7. Test watchdog cutoff on idle
        t_wait_start = time.time()
        while time.time() - t_wait_start < 2.0:
            if rpi.servos.is_cutoff:
                break
            time.sleep(0.05)
        assert rpi.servos.is_cutoff is True

    finally:
        host.close()
        rpi.stop()
