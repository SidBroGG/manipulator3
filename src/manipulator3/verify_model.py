import logging
import mujoco
from manipulator3.model import (
    load_mujoco_model,
    load_ikpy_chain,
    get_gripper_pose,
    get_tof_reading,
    ARM_JOINTS,
    GRIPPER_JOINTS,
    ACTUATOR_NAMES,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("manipulator3")


def verify_geometry() -> bool:
    logger.info("[INFO] Starting Stage 1 geometry and model verification...")

    # 1. Verify IKPy Chain from URDF
    chain = load_ikpy_chain()
    logger.info(f"[INFO] URDF loaded via IKPy: {len(chain.links)} links in kinematic chain")

    # 2. Verify MuJoCo MJCF Model
    model, data = load_mujoco_model()
    logger.info(f"[INFO] MuJoCo model compiled successfully: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms")

    # Verify expected bodies
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
        if b_id < 0:
            logger.error(f"[ERROR] Required body '{b}' missing from model!")
            return False
    logger.info("[INFO] All 5 arm links and gripper bodies verified.")

    # 3. Verify joints and actuators
    for j in ARM_JOINTS + GRIPPER_JOINTS:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if j_id < 0:
            logger.error(f"[ERROR] Required joint '{j}' missing from model!")
            return False

    for act in ACTUATOR_NAMES:
        a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
        if a_id < 0:
            logger.error(f"[ERROR] Required actuator '{act}' missing from model!")
            return False
    logger.info("[INFO] All joints (revolute 1-7) and actuators verified.")

    # 4. Verify Camera attached to main__gripperbase
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_camera")
    if cam_id < 0:
        logger.error("[ERROR] Camera 'gripper_camera' missing from model!")
        return False
    cam_body_id = model.cam_bodyid[cam_id]
    gripper_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main__gripperbase")
    if cam_body_id != gripper_base_id:
        logger.error("[ERROR] Camera is not attached to 'main__gripperbase'!")
        return False
    logger.info("[INFO] Virtual camera 'gripper_camera' verified on 'main__gripperbase'.")

    # 5. Verify ToF Rangefinder attached to main__gripperbase
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tof_rangefinder")
    if sensor_id < 0:
        logger.error("[ERROR] Sensor 'tof_rangefinder' missing from model!")
        return False
    site_id = model.sensor_objid[sensor_id]
    site_body_id = model.site_bodyid[site_id]
    if site_body_id != gripper_base_id:
        logger.error("[ERROR] ToF sensor site is not attached to 'main__gripperbase'!")
        return False
    tof_val = get_tof_reading(model, data)
    logger.info(f"[INFO] ToF sensor verified on 'main__gripperbase' (Sightline reading: {tof_val}).")

    # 6. Verify Forward Kinematics
    pos, mat = get_gripper_pose(model, data)
    logger.info(f"[INFO] Gripperbase home position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")

    # 7. Test Gripper Mimic Coupling
    data.ctrl[model.actuator("act_gripper").id] = 0.4
    for _ in range(200):
        mujoco.mj_step(model, data)
    q7 = data.qpos[model.joint("revolute_7").qposadr[0]]
    q6 = data.qpos[model.joint("revolute_6").qposadr[0]]
    if abs(q6 + q7) > 0.02:
        logger.error(f"[ERROR] Mimic coupling failed: q7={q7:.4f}, q6={q6:.4f}")
        return False
    logger.info(f"[INFO] Gripper mimic verified: q7={q7:.4f}, q6={q6:.4f} (symmetric)")

    logger.info("[INFO] Stage 1 verification successfully completed!")
    return True


if __name__ == "__main__":
    verify_geometry()
