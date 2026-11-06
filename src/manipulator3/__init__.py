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


def __getattr__(name: str):
    if name in (
        "MJCF_PATH",
        "SCENE_PATH",
        "URDF_PATH",
        "load_mujoco_model",
        "load_mujoco_scene",
        "load_ikpy_chain",
        "get_gripper_pose",
        "get_tof_reading",
    ):
        from manipulator3 import model
        return getattr(model, name)
    if name in ("SimulationEnvironment", "draw_sensor_overlay"):
        from manipulator3 import simulation
        return getattr(simulation, name)
    if name == "SimZenohBridge":
        from manipulator3 import sim_bridge
        return getattr(sim_bridge, name)
    if name == "HostClient":
        from manipulator3 import zenoh_client
        return getattr(zenoh_client, name)
    if name in (
        "KinematicsEngine",
        "SafetyLimitError",
        "TableCollisionError",
        "WorkspaceLimitError",
        "KinematicsConvergenceError",
        "POSE_HOME",
        "POSE_SCAN",
        "POSE_DROP_OFF",
        "BASELINE_POSES",
        "TABLE_Z_MIN",
    ):
        from manipulator3 import kinematics
        return getattr(kinematics, name)
    if name in ("TrajectoryPlanner", "Trajectory"):
        from manipulator3 import trajectory
        return getattr(trajectory, name)
    if name in (
        "PointingStatus",
        "PointingTarget",
        "TableBounds",
        "HandLandmarks",
        "HandTracker",
        "intersect_ray_with_table",
        "compute_pointing_vector",
        "transform_ray_camera_to_world",
        "is_pointing_gesture",
    ):
        from manipulator3 import pointing
        return getattr(pointing, name)
    if name in (
        "DetectedObject",
        "compute_detected_object",
        "YOLOObjectDetector",
        "ColorBlobDetector",
        "draw_detection_overlay",
        "DEFAULT_PICKABLE_CLASSES",
        "EXCLUDED_ROBOT_CLASSES",
        "is_gripper_finger",
    ):
        from manipulator3 import detector
        return getattr(detector, name)
    if name in (
        "TrackingState",
        "VisualServoController",
        "SectorScanner",
        "VisualSearchTracker",
    ):
        from manipulator3 import servoing
        return getattr(servoing, name)
    if name in (
        "ApproachStatus",
        "ToFApproachController",
        "GraspController",
        "GRIPPER_OPEN",
        "GRIPPER_GRASP_CMD",
        "GRIPPER_CLOSED_EMPTY",
        "TOF_TARGET_MIN_MM",
        "TOF_TARGET_MAX_MM",
        "TOF_TARGET_NOMINAL_MM",
    ):
        from manipulator3 import grasping
        return getattr(grasping, name)
    if name in (
        "ManipulatorState",
        "StateMachineOutput",
        "PickAndPlaceStateMachine",
    ):
        from manipulator3 import state_machine
        return getattr(state_machine, name)
    if name in (
        "GPIOBackend",
        "MockGPIOBackend",
        "LGPIOBackend",
        "RPiGPIOBackend",
        "get_default_gpio_backend",
        "StepperDriverA4988",
        "ServoConfig",
        "ServoController",
        "ToFReaderVL53L0X",
        "CameraStreamerOV5647",
        "PIN_STEP",
        "PIN_DIR",
        "PIN_SERVO_SHOULDER_A",
        "PIN_SERVO_SHOULDER_B",
        "PIN_SERVO_ELBOW",
        "PIN_SERVO_WRIST_PITCH",
        "PIN_SERVO_WRIST_ROLL",
        "PIN_SERVO_GRIPPER",
    ):
        from manipulator3 import hardware
        return getattr(hardware, name)
    if name == "ThinClient":
        from manipulator3 import thin_client
        return getattr(thin_client, name)
    if name == "ServoTester":
        from manipulator3 import servo_tester
        return getattr(servo_tester, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    print("Manipulator3 - 6-DOF Robotic Manipulator with Vision & ToF")
    print("Run interactive examples:")
    print("  uv run python examples/sim_preview.py")
    print("  uv run python examples/zenoh_preview.py")
    print("  uv run python examples/kinematics_preview.py")
    print("  uv run python examples/hand_preview.py")
    print("  uv run python examples/detector_preview.py")
    print("  uv run python examples/statemachine_preview.py")
    print("  uv run python examples/interactive_pointing_demo.py")

