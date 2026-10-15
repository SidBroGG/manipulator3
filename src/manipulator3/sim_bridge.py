import argparse
import time
from typing import Dict, List, Optional, Union
import numpy as np
import zenoh

from manipulator3.model import ACTUATOR_NAMES, ALL_JOINTS
from manipulator3.simulation import SimulationEnvironment
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
)


class SimZenohBridge:
    """Zenoh bridge running MuJoCo simulation as a digital twin for RPi Zero 2 W."""

    def __init__(
        self,
        sim: Optional[SimulationEnvironment] = None,
        config: Optional[zenoh.Config] = None,
        camera_name: str = "gripper_camera",
        jpeg_quality: int = 75,
    ) -> None:
        self.sim = sim or SimulationEnvironment()
        self.camera_name = camera_name
        self.jpeg_quality = jpeg_quality

        self.config = config or zenoh.Config()
        self.session = zenoh.open(self.config)

        # Publishers
        self.pub_camera = self.session.declare_publisher(TOPIC_CAMERA_RAW)
        self.pub_tof = self.session.declare_publisher(TOPIC_TOF_RANGE)
        self.pub_state = self.session.declare_publisher(TOPIC_JOINTS_STATE)
        self.pub_status = self.session.declare_publisher(TOPIC_STATUS)

        # Subscriber for commands
        self.sub_cmd = self.session.declare_subscriber(
            TOPIC_JOINTS_COMMAND, self._on_joint_command
        )

        self.last_command: Optional[JointsCommand] = None

    def _on_joint_command(self, sample: zenoh.Sample) -> None:
        try:
            cmd = JointsCommand.from_bytes(sample.payload.to_bytes())
            self.last_command = cmd
            self.apply_command(cmd.angles)
        except Exception:
            pass

    def apply_command(self, angles: Union[Dict[str, float], List[float], np.ndarray]) -> None:
        """Apply joint angles to MuJoCo actuators."""
        if isinstance(angles, (list, tuple, np.ndarray)):
            for i, val in enumerate(angles):
                if i < len(ACTUATOR_NAMES):
                    self.sim.set_actuator_target(ACTUATOR_NAMES[i], float(val))
        elif isinstance(angles, dict):
            for k, v in angles.items():
                if k in self.sim.actuator_map:
                    self.sim.set_actuator_target(k, float(v))
                elif f"act_{k}" in self.sim.actuator_map:
                    self.sim.set_actuator_target(f"act_{k}", float(v))
                elif k in ("gripper", "revolute_7", "gripper_joint"):
                    self.sim.set_actuator_target("act_gripper", float(v))

    def step_and_publish(self, physics_steps: int = 10) -> None:
        """Step simulation physics and publish sensor packets over Zenoh."""
        now = time.time()
        self.sim.step(physics_steps)

        # 1. Camera
        bgr = self.sim.render_bgr(self.camera_name)
        jpeg_bytes = encode_jpeg(bgr, quality=self.jpeg_quality)
        self.pub_camera.put(jpeg_bytes)

        # 2. ToF Range
        tof_m = self.sim.get_tof_reading()
        tof_msg = ToFRange(
            distance_mm=tof_m * 1000.0 if tof_m > 0 else -1.0,
            valid=tof_m > 0,
            timestamp=now,
        )
        self.pub_tof.put(tof_msg.to_bytes())

        # 3. Joint States
        qpos_dict = self.sim.get_joint_positions()
        qvel_dict = {
            name: float(self.sim.data.qvel[self.sim.model.joint(name).dofadr[0]])
            for name in ALL_JOINTS
            if self.sim.model.joint(name).dofadr[0] >= 0
        }
        state_msg = JointsState(positions=qpos_dict, velocities=qvel_dict, timestamp=now)
        self.pub_state.put(state_msg.to_bytes())

    def run(self, fps: float = 30.0, duration: float = 0.0) -> None:
        """Run continuous simulation and publishing loop."""
        period = 1.0 / fps
        start_time = time.time()
        last_log_time = start_time
        frame_count = 0
        current_fps = 0.0

        print(f"[INFO] Simulation Zenoh bridge started at {fps:.1f} target FPS...")
        try:
            while True:
                t0 = time.time()
                elapsed = t0 - start_time
                if duration > 0 and elapsed >= duration:
                    break

                self.step_and_publish()
                frame_count += 1

                # Periodic logging & status publish
                if t0 - last_log_time >= 1.5:
                    current_fps = frame_count / (t0 - last_log_time)
                    frame_count = 0
                    last_log_time = t0
                    status = RobotStatus(state="SIMULATION", connected=True, fps=current_fps, timestamp=t0)
                    self.pub_status.put(status.to_bytes())
                    tof_mm = self.sim.get_tof_distance_mm()
                    tof_str = f"{tof_mm:.1f} mm" if tof_mm > 0 else "> 2000 mm (Clear)"
                    print(f"[INFO] Sim Bridge | FPS: {current_fps:.1f} | ToF: {tof_str}")

                # Sleep remaining cycle
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            print("[INFO] Sim Zenoh bridge stopped.")

    def close(self) -> None:
        """Close Zenoh session and simulation environment."""
        if not self.session.is_closed():
            self.session.close()
        self.sim.close()

    def __enter__(self) -> "SimZenohBridge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 MuJoCo Simulation Zenoh Bridge")
    parser.add_argument("--fps", type=float, default=30.0, help="Target simulation publishing FPS")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = infinite)")
    parser.add_argument("--quality", type=int, default=75, help="JPEG compression quality (1-100)")
    args = parser.parse_args()

    bridge = SimZenohBridge(jpeg_quality=args.quality)
    try:
        bridge.run(fps=args.fps, duration=args.duration)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
