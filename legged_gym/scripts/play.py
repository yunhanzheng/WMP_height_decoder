# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.difficulty = 0.1  # use 0.15 for stripe obstacle
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # env_cfg.terrain.difficulty = 1.0  # use 0.15 for stripe obstacle
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    env_cfg.noise.add_noise = False

    # Keep domain randomizations ENABLED but with fixed values to match training observation structure
    env_cfg.domain_rand.friction_range = [0.8, 0.8]  # Fixed value
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]  # Fixed value
    env_cfg.domain_rand.added_mass_range = [0., 0.]  # Fixed value (no added mass)
    env_cfg.domain_rand.com_x_pos_range = [0.0, 0.0]  # Fixed value (no offset)
    env_cfg.domain_rand.com_y_pos_range = [0.0, 0.0]  # Fixed value (no offset)
    env_cfg.domain_rand.com_z_pos_range = [0.0, 0.0]  # Fixed value (no offset)

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True  # Keep enabled with fixed values
    env_cfg.domain_rand.randomize_friction = True  # MUST be True to include in observations
    env_cfg.domain_rand.randomize_restitution = True  # MUST be True to include in observations
    env_cfg.domain_rand.randomize_base_mass = True  # MUST be True to include in observations
    env_cfg.domain_rand.randomize_com_pos = True  # MUST be True to include in observations
    env_cfg.domain_rand.randomize_link_mass = False  # Doesn't affect observations
    env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]

    env_cfg.commands.ranges.lin_vel_x = [0.8, 0.8]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.use_wandb = False


    train_cfg.runner.checkpoint = -1
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # Initialize world model components only if using WMPRunner
    use_world_model = hasattr(ppo_runner, '_world_model')

    # Verify observation dimensions match expected
    print(f"Observation shape: {obs.shape}, Expected: [{env.num_envs}, {env.num_obs}]")
    if obs.shape[-1] != env.num_obs:
        print(f"WARNING: Observation dimension mismatch! Got {obs.shape[-1]}, expected {env.num_obs}")
        print(f"This may cause incorrect behavior. Check domain_rand settings in play.py")

    if use_world_model:
        history_length = 5
        # Use env.num_obs for consistent trajectory dimension (needed for trained model compatibility)
        traj_end_idx = env.num_obs - env.height_dim
        obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                            obs[:, env.privileged_dim + 9:traj_end_idx]), dim=1)
        trajectory_history = torch.zeros(size=(env.num_envs, history_length, obs_without_command.shape[1]), device = env.device)
        trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        # Initialize world model
        world_model = ppo_runner._world_model.to(env.device)
        wm_latent = wm_action = None
        wm_is_first = torch.ones(env.num_envs, device=env.device)
        wm_update_interval = env.cfg.depth.update_interval
        wm_action_history = torch.zeros(size=(env.num_envs, wm_update_interval, env.num_actions),
                                        device=env.device)
        wm_obs = {
            "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
            "is_first": wm_is_first,
        }

        if (env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                          device=world_model.device)

        wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)
    else:
        wm_feature = None

    total_reward = 0
    not_dones = torch.ones((env.num_envs,), device=env.device)
    for i in range(1*int(env.max_episode_length) + 3):
        if use_world_model:
            if (env.global_counter % wm_update_interval == 0):
                if (env.cfg.depth.use_camera):
                    wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)

                wm_embed = world_model.encoder(wm_obs)
                wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
                wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
                wm_is_first[:] = 0

        if use_world_model:
            history = trajectory_history.flatten(1).to(env.device)
            actions = policy(obs.detach(), history.detach(), wm_feature.detach())
        else:
            # Extract actor observation (skip privileged_dim and lin_vel, take prop_dim + action_dim)
            start_idx = env.privileged_dim + 3
            end_idx = start_idx + env.cfg.env.prop_dim + env.cfg.env.action_dim
            actor_obs = obs[:, start_idx:end_idx]
            actions = policy(actor_obs.detach())


        obs, _, rews, dones, infos, reset_env_ids = env.step(actions.detach())

        not_dones *= (~dones)
        total_reward += torch.mean(rews * not_dones)

        # update world model input
        if use_world_model:
            wm_action_history = torch.concat(
                (wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
            wm_obs = {
                "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
                "is_first": wm_is_first,
            }
            if (env.cfg.depth.use_camera):
                wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                              device=world_model.device)

            reset_env_ids = reset_env_ids.cpu().numpy()
            if (len(reset_env_ids) > 0):
                wm_action_history[reset_env_ids, :] = 0
                wm_is_first[reset_env_ids] = 1

            wm_action = wm_action_history.flatten(1)

        # process trajectory history
        if use_world_model:
            env_ids = dones.nonzero(as_tuple=False).flatten()
            trajectory_history[env_ids] = 0
            # Use env.num_obs for consistent trajectory dimension
            traj_end_idx = env.num_obs - env.height_dim
            obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                                obs[:, env.privileged_dim + 9:traj_end_idx]),
                                               dim=1)
            trajectory_history = torch.concat(
                (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            lootat = env.root_states[robot_index, :3]
            camara_position = lootat.detach().cpu().numpy() + [0, 1, 0]
            env.set_camera(camara_position, lootat)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

    print('total reward:', total_reward)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    args.rl_device = args.sim_device
    play(args)
