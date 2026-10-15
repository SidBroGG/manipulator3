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
from manipulator3.simulation import SimulationEnvironment, draw_sensor_overlay


def main() -> None:
    from manipulator3.sim_preview import main as preview_main
    preview_main()
