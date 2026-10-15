import pytest
import numpy as np
import mujoco
from manipulator3.model import (
    load_mujoco_model,
    load_ikpy_chain,
    get_gripper_pose,
    get_tof_reading,
    ARM_JOINTS,
    GRIPPER_JOINTS,
    ACTUATOR_NAMES,
    URDF_PATH,
    MJCF_PATH,
)


def test_urdf_file_and_ikpy():
    """Verify URDF loads cleanly in both IKPy and MuJoCo without missing mesh errors."""
    assert URDF_PATH.exists(), f"URDF file not found at {URDF_PATH}"

    # Load in MuJoCo
    m_urdf = mujoco.MjModel.from_xml_path(str(URDF_PATH))
    assert m_urdf.nbody >= 8
    assert m_urdf.ngeom > 0
    assert m_urdf.nmesh == 8

    # Load in IKPy
    chain = load_ikpy_chain()
    assert len(chain.links) == 7  # base + 5 arm joints + gripperbase


def test_mujoco_mjcf_compilation():
    """Verify assets/manipulator.xml compiles and contains all required bodies and joints."""
    assert MJCF_PATH.exists(), f"MJCF file not found at {MJCF_PATH}"
    model, data = load_mujoco_model()

    expected_bodies = [
        "base_link",
        "base_spinning",
        "shoulder_link",
        "elbow_link",
        "wrist_link",
        "main__gripperbase",
        "main__gripper1",
        "main__gripper2",
    ]
    for b in expected_bodies:
        b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        assert b_id >= 0, f"Body {b} not found in MJCF model"

    for j in ARM_JOINTS + GRIPPER_JOINTS:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert j_id >= 0, f"Joint {j} not found in MJCF model"

    for act in ACTUATOR_NAMES:
        a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
        assert a_id >= 0, f"Actuator {act} not found in MJCF model"


def test_sensor_rig_attachment():
    """Verify camera and ToF sensor are mounted on main__gripperbase."""
    model, data = load_mujoco_model()
    gripper_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main__gripperbase")

    # Camera check
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_camera")
    assert cam_id >= 0, "Camera 'gripper_camera' missing"
    assert model.cam_bodyid[cam_id] == gripper_base_id, "Camera not mounted on main__gripperbase"
    assert np.isclose(model.cam_fovy[cam_id], 48.8, atol=0.5), "Camera FOV mismatch"

    # ToF Sensor check
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tof_rangefinder")
    assert sensor_id >= 0, "Sensor 'tof_rangefinder' missing"
    site_id = model.sensor_objid[sensor_id]
    assert model.site_bodyid[site_id] == gripper_base_id, "ToF site not on main__gripperbase"

    # Sightline check in free space
    tof_val = get_tof_reading(model, data)
    assert tof_val == -1.0, f"ToF sightline blocked internally (reading: {tof_val})"


def test_gripper_mimic_coupling():
    """Verify revolute_6 mimicked motion when revolute_7 is actuated."""
    model, data = load_mujoco_model()
    gripper_act_id = model.actuator("act_gripper").id

    # Drive gripper open
    data.ctrl[gripper_act_id] = 0.3
    for _ in range(300):
        mujoco.mj_step(model, data)

    q7 = data.qpos[model.joint("revolute_7").qposadr[0]]
    q6 = data.qpos[model.joint("revolute_6").qposadr[0]]
    assert np.isclose(q6, -q7, atol=0.03), f"Mimic coupling failed: q6={q6}, q7={q7}"


def test_forward_kinematics():
    """Validate forward kinematics of gripperbase at home and displaced poses."""
    model, data = load_mujoco_model()

    # Home pose
    pos_home, _ = get_gripper_pose(model, data)
    assert np.isclose(pos_home[0], 0.0, atol=0.03)
    assert np.isclose(pos_home[1], 0.0, atol=0.03)
    assert pos_home[2] > 0.3  # End-effector elevated vertically

    # Displaced pose: pitch shoulder forward
    data.ctrl[model.actuator("act_revolute_2").id] = 0.5
    for _ in range(500):
        mujoco.mj_step(model, data)

    pos_rot, _ = get_gripper_pose(model, data)
    assert abs(pos_rot[0] - pos_home[0]) > 0.05 or abs(pos_rot[2] - pos_home[2]) > 0.05
