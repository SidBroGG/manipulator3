from pathlib import Path
from typing import Tuple
import numpy as np
import mujoco
import ikpy.chain

# Repository paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MJCF_PATH = REPO_ROOT / "assets" / "manipulator.xml"
URDF_PATH = REPO_ROOT / "assets" / "root.urdf"

ARM_JOINTS = [
    "revolute_1",  # base yaw
    "revolute_2",  # shoulder pitch
    "revolute_3",  # elbow pitch
    "revolute_4",  # wrist pitch
    "revolute_5",  # wrist roll
]

GRIPPER_JOINTS = [
    "revolute_6",  # left jaw (mimicked)
    "revolute_7",  # right jaw (actuated)
]

ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS

ACTUATOR_NAMES = [
    "act_revolute_1",
    "act_revolute_2",
    "act_revolute_3",
    "act_revolute_4",
    "act_revolute_5",
    "act_gripper",
]


def load_mujoco_model(xml_path: Path | str | None = None) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Load MuJoCo MJCF model and initialize simulation data."""
    path = str(xml_path or MJCF_PATH)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def load_ikpy_chain(urdf_path: Path | str | None = None) -> ikpy.chain.Chain:
    """Load serial kinematic chain up to main__gripperbase."""
    path = str(urdf_path or URDF_PATH)
    return ikpy.chain.Chain.from_urdf_file(
        path,
        base_elements=["base_link"],
        active_links_mask=[False, True, True, True, True, True, False],
    )


def get_gripper_pose(model: mujoco.MjModel, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
    """Return position (3,) and orientation matrix (3, 3) of main__gripperbase."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main__gripperbase")
    if body_id < 0:
        raise ValueError("Body 'main__gripperbase' not found in model.")
    pos = data.xpos[body_id].copy()
    mat = data.xmat[body_id].reshape(3, 3).copy()
    return pos, mat


def get_tof_reading(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Return distance in meters from virtual VL53L0X ToF rangefinder (-1.0 if clear)."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tof_rangefinder")
    if sensor_id < 0:
        raise ValueError("Sensor 'tof_rangefinder' not found in model.")
    adr = model.sensor_adr[sensor_id]
    return float(data.sensordata[adr])
