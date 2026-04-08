"""Headless policy playback with video recording.

Run on a remote machine (no display needed):
    python legged_gym/scripts/play_headless.py --task go2_wmp \
        --num_envs 1 --sim_device cuda:0 --rl_device cuda:0 \
        --output_video output.mp4 --fps 50

How it works:
- Creates a camera sensor attached to env 0 (GPU rendering, no viewer window).
- Captures RGB frames each step via gym.get_camera_image().
- Assembles frames into an MP4 using imageio-ffmpeg.
"""

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from isaacgym import gymapi, gymutil
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch
import imageio
import argparse


# ──────────────────────────────────────────────────────────────────
# Extra CLI args specific to this script
# ──────────────────────────────────────────────────────────────────
def get_video_args():
    """Parse extra args after get_args() has consumed its share."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_video", type=str, default="play_output.mp4",
                        help="Path for the output MP4 file")
    parser.add_argument("--fps", type=int, default=50,
                        help="Frames per second in the output video")
    parser.add_argument("--cam_width", type=int, default=1280,
                        help="Camera width in pixels")
    parser.add_argument("--cam_height", type=int, default=720,
                        help="Camera height in pixels")
    parser.add_argument("--cam_offset_x", type=float, default=-0.8,
                        help="Camera offset X from robot (behind)")
    parser.add_argument("--cam_offset_y", type=float, default=0.0,
                        help="Camera offset Y from robot (side)")
    parser.add_argument("--cam_offset_z", type=float, default=0.5,
                        help="Camera offset Z from robot (above)")
    known, _ = parser.parse_known_args()
    return known


def play_headless(args, video_args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ── Environment overrides ──────────────────────────────────────
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 20
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_length = 8
    env_cfg.terrain.terrain_width = 8
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.difficulty = 0.5
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.commands.ranges.lin_vel_x = [1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    train_cfg.runner.amp_num_preload_transitions = 1

    # ── CRITICAL: keep graphics device on GPU even in headless mode ─
    # base_task sets graphics_device_id = -1 when headless=True.
    # We need it on GPU to use camera sensors.
    # Patch: force headless=False so the graphics device stays on GPU,
    # but we will NOT create a viewer (we override that below).
    args.headless = False   # allow GPU graphics
    _orig_headless = True   # remember we still don't want a window

    # ── Make environment ───────────────────────────────────────────
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Close the auto-created viewer if one was made
    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
        env.viewer = None

    # ── Attach a chase camera to env 0 ────────────────────────────
    cam_props = gymapi.CameraProperties()
    cam_props.width = video_args.cam_width
    cam_props.height = video_args.cam_height
    cam_props.enable_tensors = False   # we use get_camera_image (CPU)
    cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)

    # Position camera offset relative to the robot root body
    robot_index = 0
    actor_handle = env.actor_handles[robot_index]
    root_body = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)

    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(
        video_args.cam_offset_x,
        video_args.cam_offset_y,
        video_args.cam_offset_z,
    )
    # Look forward (+X) and slightly downward
    local_transform.r = gymapi.Quat.from_euler_zyx(0, np.deg2rad(-15), 0)

    env.gym.attach_camera_to_body(
        cam_handle,
        env.envs[0],
        root_body,
        local_transform,
        gymapi.FOLLOW_TRANSFORM,
    )

    # ── Load policy ───────────────────────────────────────────────
    _, _ = env.reset()
    obs = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.use_wandb = False
    train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ── World-model bookkeeping (same as play.py) ──────────────────
    use_world_model = hasattr(ppo_runner, '_world_model')
    if use_world_model:
        history_length = 5
        traj_end_idx = env.num_obs - env.height_dim
        obs_without_command = torch.cat(
            (obs[:, env.privileged_dim:env.privileged_dim + 6],
             obs[:, env.privileged_dim + 9:traj_end_idx]), dim=1)
        trajectory_history = torch.zeros(
            env.num_envs, history_length, obs_without_command.shape[1],
            device=env.device)
        trajectory_history = torch.cat(
            (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        world_model = ppo_runner._world_model.to(env.device)
        wm_latent = wm_action = None
        wm_is_first = torch.ones(env.num_envs, device=env.device)
        wm_update_interval = env.cfg.depth.update_interval
        wm_action_history = torch.zeros(
            env.num_envs, wm_update_interval, env.num_actions, device=env.device)
        wm_obs = {
            "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
            "is_first": wm_is_first,
        }
        wm_feature = torch.zeros(
            (env.num_envs, ppo_runner.wm_feature_dim), device=env.device)
    else:
        wm_feature = None

    # ── Video writer ──────────────────────────────────────────────
    output_path = video_args.output_video
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=video_args.fps, codec="libx264",
                                quality=8)
    print(f"Recording to: {output_path}  ({video_args.cam_width}x{video_args.cam_height} @ {video_args.fps} fps)")

    # ── Rollout ───────────────────────────────────────────────────
    num_steps = int(env.max_episode_length) + 3
    for i in range(num_steps):
        # World-model step
        if use_world_model:
            if env.global_counter % wm_update_interval == 0:
                wm_embed = world_model.encoder(wm_obs)
                wm_latent, _ = world_model.dynamics.obs_step(
                    wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
                wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
                wm_is_first[:] = 0

            history = trajectory_history.flatten(1).to(env.device)
            actions = policy(obs.detach(), history.detach(), wm_feature.detach())
        else:
            actions = policy(obs.detach())

        # Step the simulation
        obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        # Render camera and capture frame
        env.gym.render_all_camera_sensors(env.sim)
        rgb = env.gym.get_camera_image(
            env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
        # get_camera_image returns a flat RGBA uint8 array
        frame = rgb.reshape(video_args.cam_height, video_args.cam_width, 4)
        writer.append_data(frame[:, :, :3])   # drop alpha

        # Update world-model state
        if use_world_model:
            wm_action_history = torch.cat(
                (wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
            wm_obs = {
                "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
                "is_first": wm_is_first,
            }
            reset_ids = reset_env_ids.cpu().numpy()
            if len(reset_ids) > 0:
                wm_action_history[reset_ids] = 0
                wm_is_first[reset_ids] = 1
            wm_action = wm_action_history.flatten(1)

            env_ids = dones.nonzero(as_tuple=False).flatten()
            trajectory_history[env_ids] = 0
            traj_end_idx = env.num_obs - env.height_dim
            obs_without_command = torch.cat(
                (obs[:, env.privileged_dim:env.privileged_dim + 6],
                 obs[:, env.privileged_dim + 9:traj_end_idx]), dim=1)
            trajectory_history = torch.cat(
                (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        if (i + 1) % 100 == 0:
            print(f"  step {i+1}/{num_steps}")

    writer.close()
    print(f"\nDone! Video saved to: {output_path}")


if __name__ == "__main__":
    args = get_args()
    args.rl_device = args.sim_device
    video_args = get_video_args()
    play_headless(args, video_args)
