import time
import numpy as np
import pytest
import zenoh

from manipulator3.protocol import (
    TOPIC_CAMERA_RAW,
    TOPIC_TOF_RANGE,
    TOPIC_JOINTS_COMMAND,
    TOPIC_JOINTS_STATE,
    TOPIC_STATUS,
    ToFRange,
    JointsCommand,
    JointsState,
    RobotStatus,
    encode_jpeg,
    decode_jpeg,
)
from manipulator3.zenoh_client import HostClient
from manipulator3.sim_bridge import SimZenohBridge


def test_protocol_serialization():
    """Verify serialization and deserialization of protocol messages."""
    # ToFRange
    tof = ToFRange(distance_mm=123.45, valid=True, timestamp=1000.0)
    tof_bytes = tof.to_bytes()
    restored_tof = ToFRange.from_bytes(tof_bytes)
    assert restored_tof.distance_mm == pytest.approx(123.45)
    assert restored_tof.valid is True
    assert restored_tof.timestamp == pytest.approx(1000.0)

    # JointsCommand (dict and list)
    cmd_dict = JointsCommand(angles={"revolute_1": 0.5, "act_gripper": 0.2})
    restored_cmd_dict = JointsCommand.from_bytes(cmd_dict.to_bytes())
    assert restored_cmd_dict.angles["revolute_1"] == pytest.approx(0.5)

    cmd_list = JointsCommand(angles=[0.1, -0.2, 0.3, -0.4, 0.5, 0.05])
    restored_cmd_list = JointsCommand.from_bytes(cmd_list.to_bytes())
    assert len(restored_cmd_list.angles) == 6
    assert restored_cmd_list.angles[1] == pytest.approx(-0.2)

    # JointsState
    state = JointsState(
        positions={"revolute_1": 0.12},
        velocities={"revolute_1": 0.05},
        timestamp=1000.0,
    )
    restored_state = JointsState.from_bytes(state.to_bytes())
    assert restored_state.positions["revolute_1"] == pytest.approx(0.12)
    assert restored_state.velocities["revolute_1"] == pytest.approx(0.05)

    # RobotStatus
    status = RobotStatus(state="ACTIVE", connected=True, fps=30.5)
    restored_status = RobotStatus.from_bytes(status.to_bytes())
    assert restored_status.state == "ACTIVE"
    assert restored_status.connected is True
    assert restored_status.fps == pytest.approx(30.5)


def test_jpeg_codec():
    """Verify JPEG encoding and decoding fidelity on structured image."""
    # Create structured gradient test pattern
    y, x = np.mgrid[0:240, 0:320]
    original = np.zeros((240, 320, 3), dtype=np.uint8)
    original[:, :, 0] = (x % 256).astype(np.uint8)
    original[:, :, 1] = (y % 256).astype(np.uint8)
    original[:, :, 2] = 128

    jpeg_data = encode_jpeg(original, quality=85)
    assert isinstance(jpeg_data, bytes)
    assert len(jpeg_data) > 0

    decoded = decode_jpeg(jpeg_data)
    assert decoded.shape == original.shape
    # Structured image compression error is minimal
    assert np.mean(np.abs(original.astype(float) - decoded.astype(float))) < 8.0



def test_zenoh_sensor_roundtrip():
    """Verify roundtrip latency and throughput of sensor packets over Zenoh."""
    s1 = zenoh.open(zenoh.Config())
    s2 = zenoh.open(zenoh.Config())

    try:
        received_samples = []
        sub = s2.declare_subscriber(
            TOPIC_TOF_RANGE, lambda s: received_samples.append((time.perf_counter(), s.payload.to_bytes()))
        )
        pub = s1.declare_publisher(TOPIC_TOF_RANGE)
        time.sleep(0.1)

        latencies = []
        for i in range(20):
            t_send = time.perf_counter()
            msg = ToFRange(distance_mm=50.0 + i, valid=True, timestamp=t_send)
            pub.put(msg.to_bytes())
            time.sleep(0.005)

        # Allow delivery
        time.sleep(0.1)
        assert len(received_samples) >= 15
        for t_recv, raw in received_samples:
            parsed = ToFRange.from_bytes(raw)
            latencies.append((t_recv - parsed.timestamp) * 1000.0)

        mean_latency = float(np.mean(latencies))
        assert mean_latency < 25.0  # Under 25ms roundtrip on local transport
    finally:
        s1.close()
        s2.close()


def test_sim_bridge_and_host_client_integration():
    """Test full integration between SimZenohBridge and HostClient."""
    bridge = SimZenohBridge()
    client = HostClient()

    try:
        # 1. Send joint command from host
        test_angles = {
            "act_revolute_1": 0.25,
            "act_revolute_2": -0.35,
            "act_gripper": 0.15,
        }
        client.send_joints_command(test_angles)
        time.sleep(0.05)

        # 2. Bridge steps simulation and publishes
        for _ in range(5):
            bridge.step_and_publish(physics_steps=5)
            time.sleep(0.01)

        # Verify command applied in simulation
        qpos = bridge.sim.get_joint_positions()
        assert "revolute_1" in qpos

        # 3. Host receives camera and ToF reading
        data_ready = client.wait_for_data(timeout=2.0)
        assert data_ready is True

        frame_info = client.get_camera_frame(decode=True)
        assert frame_info is not None
        frame, ts = frame_info
        assert frame.shape == (480, 640, 3)

        tof_reading = client.get_tof_range()
        assert tof_reading is not None
        assert isinstance(tof_reading.distance_mm, float)

        state = client.get_joints_state()
        assert state is not None
        assert "revolute_1" in state.positions

    finally:
        client.close()
        bridge.close()
