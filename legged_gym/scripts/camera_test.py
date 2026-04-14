# minimal_camera_test.py
import os
import sys
import imageio.v2 as imageio
import numpy as np

from isaacgym import gymapi

gym = gymapi.acquire_gym()

# ----------------------------
# Config
# ----------------------------


USE_GPU_PIPELINE = True
COMPUTE_DEVICE_ID = 0
GRAPHICS_DEVICE_ID = 0   # important for camera rendering in headless mode
HEADLESS = True          # keep True for remote SSH testing

OUT_PATH = "camera_test.png"

print("COMPUTE_DEVICE_ID =", COMPUTE_DEVICE_ID)
print("GRAPHICS_DEVICE_ID =", GRAPHICS_DEVICE_ID)
print("HEADLESS =", HEADLESS)

# ----------------------------
# Create sim
# ----------------------------
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.use_gpu_pipeline = USE_GPU_PIPELINE

sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1


# sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)

print("before create_sim")
sim = gym.create_sim(COMPUTE_DEVICE_ID, GRAPHICS_DEVICE_ID, gymapi.SIM_PHYSX, sim_params)
print("after create_sim")


if sim is None:
    raise RuntimeError("Failed to create sim")

# ----------------------------
# Ground
# ----------------------------
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# ----------------------------
# Environment
# ----------------------------
env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
env = gym.create_env(sim, env_lower, env_upper, 1)

# ----------------------------
# Add a simple box actor
# ----------------------------
box_asset = gym.create_box(sim, 0.4, 0.4, 0.4, gymapi.AssetOptions())

box_pose = gymapi.Transform()
box_pose.p = gymapi.Vec3(0.0, 0.0, 0.2)

box_actor = gym.create_actor(env, box_asset, box_pose, "box", 0, 0)

# ----------------------------
# Camera
# ----------------------------
cam_props = gymapi.CameraProperties()
cam_props.width = 640
cam_props.height = 480
cam_props.enable_tensors = False  # simplest path for first test

cam_handle = gym.create_camera_sensor(env, cam_props)
if cam_handle == -1:
    raise RuntimeError("Failed to create camera sensor")

cam_pos = gymapi.Vec3(1.5, 1.5, 1.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.2)
gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

# ----------------------------
# Optional viewer (disabled here)
# ----------------------------
viewer = None
if not HEADLESS:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

# ----------------------------
# Step sim a bit so physics/graphics settle
# ----------------------------
for _ in range(10):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)

    if viewer is not None:
        gym.draw_viewer(viewer, sim, False)

# ----------------------------
# Render camera and fetch image
# ----------------------------
gym.render_all_camera_sensors(sim)

color = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
if color is None:
    raise RuntimeError("Camera returned None")

# Isaac Gym returns RGBA uint8 image buffer
img = np.array(color, dtype=np.uint8).reshape(cam_props.height, cam_props.width, 4)
rgb = img[:, :, :3]

imageio.imwrite(OUT_PATH, rgb)
print(f"Saved image to: {os.path.abspath(OUT_PATH)}")

# ----------------------------
# Cleanup
# ----------------------------
if viewer is not None:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)