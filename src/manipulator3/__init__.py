from manipulator3.model import (
    MJCF_PATH,
    SCENE_PATH,
    URDF_PATH,
    load_mujoco_model,
    load_mujoco_scene,
    load_ikpy_chain,
    get_gripper_pose,
    get_tof_reading,
)
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    from manipulator3.zenoh_preview import main as preview_main
    preview_main()
