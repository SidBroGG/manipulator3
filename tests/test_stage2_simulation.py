import pytest
import numpy as np
import mujoco

from manipulator3.model import SCENE_PATH, load_mujoco_scene
from manipulator3.simulation import SimulationEnvironment, draw_sensor_overlay


def test_scene_compilation_and_entities():
    """Verify assets/scene.xml compiles and contains table, objects, and cameras."""
    assert SCENE_PATH.exists(), f"Scene file not found at {SCENE_PATH}"
    model, data = load_mujoco_scene()

    # Table and dropoff zone
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dropoff_zone") >= 0

    # Test objects on table
    expected_objects = ["target_cube", "target_cylinder", "target_box", "target_yellow"]
    for obj in expected_objects:
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
        assert obj_id >= 0, f"Object body '{obj}' missing in scene"

    # Cameras
    expected_cameras = ["overview_camera", "top_camera", "gripper_camera"]
    for cam in expected_cameras:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        assert cam_id >= 0, f"Camera '{cam}' missing in scene"

    # ToF sensor
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tof_rangefinder")
    assert sensor_id >= 0, "ToF sensor missing in scene"


def test_camera_render_pipeline():
    """Verify offscreen camera rendering output dimensions, format, and overlay."""
    with SimulationEnvironment(width=640, height=480) as sim:
        # 1. Gripper camera RGB
        rgb_gripper = sim.render("gripper_camera")
        assert rgb_gripper.shape == (480, 640, 3)
        assert rgb_gripper.dtype == np.uint8
        assert rgb_gripper.mean() > 10.0  # Scene is lit

        # 2. Gripper camera BGR
        bgr_gripper = sim.render_bgr("gripper_camera")
        assert bgr_gripper.shape == (480, 640, 3)
        assert bgr_gripper.dtype == np.uint8

        # 3. Overview and top cameras
        rgb_overview = sim.render("overview_camera")
        assert rgb_overview.shape == (480, 640, 3)
        assert rgb_overview.mean() > 10.0

        rgb_top = sim.render("top_camera")
        assert rgb_top.shape == (480, 640, 3)
        assert rgb_top.mean() > 10.0

        # 4. Sensor overlay drawing
        annotated = draw_sensor_overlay(
            bgr_gripper,
            tof_distance_m=0.125,
            fps=30.0,
            camera_name="gripper_camera",
            status_text="TEST",
        )
        assert annotated.shape == bgr_gripper.shape
        assert annotated.dtype == np.uint8


def test_tof_rangefinder_accuracy_and_approach():
    """Verify ToF reading unobstructed at HOME, and decreasing distance during approach."""
    with SimulationEnvironment() as sim:
        # Home pose: clear sightline
        home_tof = sim.get_tof_reading()
        assert home_tof == -1.0, f"Expected unobstructed ToF (-1.0), got {home_tof}"
        assert sim.get_tof_distance_mm() == -1.0

        # Pitch arm down towards table
        # Higher angles pitch shoulder and elbow down
        distances = []
        for phase in [0.4, 0.7, 1.0]:
            sim.data.ctrl[sim.actuator_map["act_revolute_2"]] = 0.2 + 0.65 * phase
            sim.data.ctrl[sim.actuator_map["act_revolute_3"]] = 0.1 + 0.45 * phase
            sim.data.ctrl[sim.actuator_map["act_revolute_4"]] = -0.2 - 1.25 * phase
            sim.step(150)
            dist = sim.get_tof_reading()
            if dist > 0:
                distances.append(dist)

        assert len(distances) >= 2, f"Expected multiple valid ToF detections during approach, got {distances}"
        # Verify distance decreases as arm pitches closer to the table
        assert distances[-1] < distances[0], (
            f"ToF reading should decrease on approach: {distances[0]} -> {distances[-1]}"
        )
        assert distances[-1] < 0.10, f"Approach should bring gripper within 100mm, got {distances[-1]}"


def test_physics_simulation_stability():
    """Verify objects rest stably on table surface without exploding or penetrating."""
    with SimulationEnvironment() as sim:
        sim.step(300)

        for obj in ["target_cube", "target_cylinder", "target_box", "target_yellow"]:
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, obj)
            z_pos = sim.data.xpos[bid][2]
            # Table top is at Z=0, objects have positive height and should stay on table
            assert z_pos > 0.005, f"Object '{obj}' penetrated table: z = {z_pos}"
            assert z_pos < 0.10, f"Object '{obj}' flew away: z = {z_pos}"
            vel = np.linalg.norm(sim.data.cvel[bid])
            assert vel < 0.1, f"Object '{obj}' not resting stably: velocity = {vel}"
