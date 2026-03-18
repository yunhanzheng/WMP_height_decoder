# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import inspect
import time
import json
from datetime import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch


def evaluate(args):
    """Run evaluation with N parallel environments and report statistics"""
    # Evaluation parameters
    num_envs = NUM_ENVS
    difficulty = DIFFICULTY
    vel_x = VEL_X
    vel_y = VEL_Y
    vel_yaw = VEL_YAW
    randomize = RANDOMIZE
    add_noise = ADD_NOISE

    if SHOW_ALL:
        print(f"\n{'='*60}")
        print(f"Evaluation: {args.task}")
        print(f"Number of parallel environments: {num_envs}")
        print(f"{'='*60}\n")

    # Get configs
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override parameters
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = 20#4.6
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_length = 15
    env_cfg.terrain.terrain_width = 15
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.difficulty = difficulty
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    env_cfg.noise.add_noise = add_noise

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

    # Command settings
    env_cfg.commands.ranges.lin_vel_x = [vel_x, vel_x]
    env_cfg.commands.ranges.lin_vel_y = [vel_y, vel_y]
    env_cfg.commands.ranges.ang_vel_yaw = [vel_yaw, vel_yaw]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env_cfg.init_state.randomize_position = False

    train_cfg.runner.amp_num_preload_transitions = 1

    # Create environment
    if SHOW_ALL: print("Creating environment...")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()

    # Load policy
    if SHOW_ALL: print("Loading policy...")
    train_cfg.runner.resume = True
    train_cfg.runner.use_wandb = False
    train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Initialize world model components only if using WMPRunner
    use_world_model = hasattr(ppo_runner, '_world_model')

    # Check if using LongShortRunner
    use_long_short = isinstance(ppo_runner, type) and 'LongShortRunner' in str(type(ppo_runner)) or \
                     'LongShortRunner' in type(ppo_runner).__name__

    # Check if using HIMOnPolicyRunner
    use_him = 'HIMOnPolicyRunner' in type(ppo_runner).__name__
    if use_him:
        if SHOW_ALL: print("HIMOnPolicyRunner detected - using observation history directly")

    # Initialize history buffers for LongShortRunner
    if use_long_short:
        short_history_length = ppo_runner.short_history_length
        long_history_length = ppo_runner.long_history_length
        prop_dim = ppo_runner.prop_dim

        # Initialize history buffers
        short_history = torch.zeros(
            env.num_envs, short_history_length, prop_dim,
            device=env.device, dtype=torch.float32
        )
        long_history = torch.zeros(
            env.num_envs, long_history_length, prop_dim,
            device=env.device, dtype=torch.float32
        )

        def extract_proprio(obs_tensor):
            """Extract proprioceptive observation from full observation."""
            privileged_dim = getattr(env, 'privileged_dim', 0)
            start_idx = privileged_dim + 3  # Skip privileged + lin_vel
            end_idx = start_idx + prop_dim
            return obs_tensor[:, start_idx:end_idx]

        def extract_commands(obs_tensor):
            """Extract command from observation."""
            privileged_dim = getattr(env, 'privileged_dim', 0)
            start_idx = privileged_dim + 9  # privileged + lin_vel(3) + ang_vel(3) + gravity(3)
            return obs_tensor[:, start_idx:start_idx + 3]

        def update_history(proprio, dones, short_hist, long_hist):
            """Update history buffers with new proprioceptive observation."""
            # Shift history and add new observation
            short_hist = torch.roll(short_hist, shifts=-1, dims=1)
            short_hist[:, -1, :] = proprio

            long_hist = torch.roll(long_hist, shifts=-1, dims=1)
            long_hist[:, -1, :] = proprio

            # Reset history for done environments
            if dones is not None:
                done_mask = dones.bool()
                if done_mask.any():
                    short_hist[done_mask] = 0
                    long_hist[done_mask] = 0

            return short_hist, long_hist

        def construct_actor_obs(obs_tensor, short_hist, long_hist):
            """Construct actor observation with history."""
            batch_size = obs_tensor.shape[0]

            # Extract components
            commands = extract_commands(obs_tensor)
            proprio = extract_proprio(obs_tensor)

            # Current obs includes proprio + lin_vel (3)
            privileged_dim = getattr(env, 'privileged_dim', 0)
            lin_vel = obs_tensor[:, privileged_dim:privileged_dim + 3]
            current_obs = torch.cat([proprio, lin_vel], dim=-1)

            # Terrain one-hot placeholder
            terrain_one_hot = torch.zeros(batch_size, 1, device=env.device)

            # Flatten history buffers
            short_hist_flat = short_hist.view(batch_size, -1)
            long_hist_flat = long_hist.view(batch_size, -1)

            # Construct full actor observation
            actor_obs = torch.cat([
                commands,
                current_obs,
                terrain_one_hot,
                short_hist_flat,
                long_hist_flat
            ], dim=-1)

            return actor_obs

        # Initialize history with current proprio
        proprio = extract_proprio(obs)
        short_history, long_history = update_history(proprio, None, short_history, long_history)

        if SHOW_ALL:
            print(f"LongShortRunner eval mode initialized:")
            print(f"  prop_dim: {prop_dim}")
            print(f"  short_history_length: {short_history_length}")
            print(f"  long_history_length: {long_history_length}")

    if use_world_model:
        if SHOW_ALL: print("Initializing world model...")
        history_length = 5
        trajectory_history = torch.zeros(size=(env.num_envs, history_length, env.num_obs -
                                                env.privileged_dim - env.height_dim - 3), device=env.device)
        if env.height_dim > 0:
            obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                                obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
        else:
            obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                                obs[:, env.privileged_dim + 9:]), dim=1)
        trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

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

        if env.cfg.depth.use_camera:
            wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                          device=world_model.device)

        wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)
    else:
        wm_feature = None

    # Run evaluation
    if SHOW_ALL: print(f"Running evaluation for {env.max_episode_length} steps...\n")
    start_time = time.time()

    total_rewards = torch.zeros((env.num_envs,), device=env.device)
    env_dones = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    # Track specific metrics
    metrics = {
        'lin_vel_mse': torch.zeros((env.num_envs,), device=env.device),
        'ang_vel_mse': torch.zeros((env.num_envs,), device=env.device),
        'collision': torch.zeros((env.num_envs,), device=env.device),
        'termination': torch.zeros((env.num_envs,), device=env.device),
        'feet_stumble': torch.zeros((env.num_envs,), device=env.device),
        'total_reward': torch.zeros((env.num_envs,), device=env.device),
        'travel_distance': torch.zeros((env.num_envs,), device=env.device),
        'success': torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
    }
    step_counts = torch.zeros((env.num_envs,), device=env.device)

    # Track initial positions for displacement calculation
    initial_positions = env.root_states[:, :2].clone()  # XY positions

    # Terrain boundary thresholds (robot starts at center of its terrain cell)
    terrain_half_length = env_cfg.terrain.terrain_length / 2.0
    terrain_half_width = env_cfg.terrain.terrain_width / 2.0

    num_steps = int(env.max_episode_length)
    for i in range(num_steps + 3):
        if use_world_model:
            if env.global_counter % wm_update_interval == 0:
                if env.cfg.depth.use_camera:
                    wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)

                wm_embed = world_model.encoder(wm_obs)
                wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
                wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
                wm_is_first[:] = 0

        if use_world_model:
            history = trajectory_history.flatten(1).to(env.device)
            actions = policy(obs.detach(), history.detach(), wm_feature.detach())
        elif use_long_short:
            # Construct actor observation with history for LongShortRunner
            actor_obs = construct_actor_obs(obs, short_history, long_history)
            actions = policy(actor_obs.detach())
        elif use_him:
            # HIMOnPolicyRunner: obs from get_observations() already contains the history buffer
            # obs shape is (num_envs, num_one_step_obs * history_steps) = (num_envs, 270)
            actions = policy(obs.detach())
        else:
            # obs_buf is already constructed as actor observations in compute_observations
            # Just use it directly
            actor_obs = obs
            actions = policy(actor_obs.detach())

        # Save positions and contact forces before step — terminated envs are reset inside step()
        # so post-step contact_forces for those envs reflect the new initial state, not the collision
        pre_step_positions = env.root_states[:, :2].clone()
        pre_step_collision = torch.sum(
            1.0 * (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1
        ) if hasattr(env, 'penalised_contact_indices') else None
        pre_step_termination = torch.sum(
            1.0 * (torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1.0), dim=1
        ) if hasattr(env, 'termination_contact_indices') else None
        if hasattr(env, 'contact_forces'):
            _fc = env.contact_forces[:, env.feet_indices, :]
            pre_step_stumble = torch.any(
                torch.norm(_fc[:, :, :2], dim=2) > 5 * torch.abs(_fc[:, :, 2]), dim=1
            ).float()
        else:
            pre_step_stumble = None

        # Success: robot has exited its terrain boundary (checked before step to avoid post-reset position)
        pre_step_disp = pre_step_positions - initial_positions
        out_of_terrain = (
            (torch.abs(pre_step_disp[:, 0]) > terrain_half_length) |
            (torch.abs(pre_step_disp[:, 1]) > terrain_half_width)
        )
        metrics['success'] |= out_of_terrain & ~env_dones

        obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        # Accumulate rewards only for environments that haven't finished
        total_rewards += rews * (~env_dones).float()

        # Track metrics for active environments
        active_mask = ~env_dones

        # Linear velocity tracking MSE (mean of squared errors in x and y)
        lin_vel_error_vec = env.commands[:, :2] - env.base_lin_vel[:, :2]
        lin_vel_mse = torch.mean(lin_vel_error_vec ** 2, dim=1)
        metrics['lin_vel_mse'] += lin_vel_mse * active_mask.float()

        # Angular velocity tracking MSE (squared error in yaw)
        ang_vel_error = env.commands[:, 2] - env.base_ang_vel[:, 2]
        ang_vel_mse = ang_vel_error ** 2
        metrics['ang_vel_mse'] += ang_vel_mse * active_mask.float()

        # Total reward accumulation
        metrics['total_reward'] += rews * active_mask.float()

        # Collision, termination, stumble — use pre-step contact forces so terminated envs aren't missed
        if pre_step_collision is not None:
            metrics['collision'] += pre_step_collision * active_mask.float()
        if pre_step_termination is not None:
            metrics['termination'] += pre_step_termination * active_mask.float()
        if pre_step_stumble is not None:
            metrics['feet_stumble'] += pre_step_stumble * active_mask.float()

        # Travel distance (displacement from start position)
        # Record displacement for environments that just terminated (use pre-step position since env resets after done)
        newly_done = dones & ~env_dones  # Environments that are done this step but weren't before
        if newly_done.any():
            displacement = torch.norm(pre_step_positions - initial_positions, dim=1)
            metrics['travel_distance'] += displacement * newly_done.float()

        # Count steps for active environments
        step_counts += active_mask.float()

        env_dones |= dones

        # Update world model input
        if use_world_model:
            wm_action_history = torch.concat(
                (wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
            wm_obs = {
                "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
                "is_first": wm_is_first,
            }
            if env.cfg.depth.use_camera:
                wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                              device=world_model.device)

            reset_env_ids = reset_env_ids.cpu().numpy()
            if len(reset_env_ids) > 0:
                wm_action_history[reset_env_ids, :] = 0
                wm_is_first[reset_env_ids] = 1

            wm_action = wm_action_history.flatten(1)

        # Process trajectory history
        if use_world_model:
            env_ids = dones.nonzero(as_tuple=False).flatten()
            trajectory_history[env_ids] = 0
            if env.height_dim > 0:
                obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                                    obs[:, env.privileged_dim + 9:-env.height_dim]),
                                                   dim=1)
            else:
                obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                                    obs[:, env.privileged_dim + 9:]),
                                                   dim=1)
            trajectory_history = torch.concat(
                (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        # Update history for LongShortRunner
        if use_long_short:
            proprio = extract_proprio(obs)
            short_history, long_history = update_history(proprio, dones, short_history, long_history)

    # Capture displacement for environments that completed full episode without terminating
    still_active = ~env_dones
    if still_active.any():
        current_positions = env.root_states[:, :2]
        displacement = torch.norm(current_positions - initial_positions, dim=1)
        metrics['travel_distance'] += displacement * still_active.float()

    # Success rate: fraction of envs that exited their terrain at any point
    success_rate = float(metrics['success'].float().mean().item())

    elapsed_time = time.time() - start_time

    # Compute statistics
    rewards_cpu = total_rewards.cpu().numpy()
    best_reward = float(np.max(rewards_cpu))
    worst_reward = float(np.min(rewards_cpu))
    mean_reward = float(np.mean(rewards_cpu))
    std_reward = float(np.std(rewards_cpu))
    median_reward = float(np.median(rewards_cpu))

    # Print results
    if SHOW_ALL:
        print(f"{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Number of environments: {num_envs}")
        print(f"Evaluation time:        {elapsed_time:.2f}s")
        print(f"\n{'─'*60}")
        print(f"TOTAL REWARD STATISTICS")
        print(f"{'─'*60}")
        print(f"Best reward:            {best_reward:.2f}")
        print(f"Worst reward:           {worst_reward:.2f}")
        print(f"Mean reward:            {mean_reward:.2f}")
        print(f"Median reward:          {median_reward:.2f}")
        print(f"Std deviation:          {std_reward:.2f}")

    # Print per-step metrics
    if SHOW_ALL:
        print(f"\n{'─'*60}")
        print(f"METRICS (per step)")
        print(f"{'─'*60}")
        print(f"{'Metric':<30} {'Best':>10} {'Mean':>10} {'Worst':>10}")
        print(f"{'─'*60}")

    if SHOW_ALL:
        print(f"\n{'─'*60}")
        print(f"SUCCESS RATE")
        print(f"{'─'*60}")
        print(f"Success rate:           {success_rate*100:.1f}%  ({int(metrics['success'].sum())}/{env.num_envs} envs)")
        print(f"  (success = robot exited {terrain_half_length*2:.1f}m×{terrain_half_width*2:.1f}m terrain)")

    metric_stats = {}
    step_counts_cpu = step_counts.cpu().numpy()

    for key, values in metrics.items():
        if key == 'success':
            continue  # handled separately
        if key == 'travel_distance':
            # Travel distance is total (not per-step), higher is better
            total_values = values.cpu().numpy()
            metric_stats[key] = {
                'best': float(np.max(total_values)),
                'mean': float(np.mean(total_values)),
                'worst': float(np.min(total_values)),
                'std': float(np.std(total_values)),
            }
        else:
            # Compute per-step values by dividing by step count
            per_step_values = values.cpu().numpy() / np.maximum(step_counts_cpu, 1.0)

            if key == 'total_reward':
                # For reward, higher is better
                metric_stats[key] = {
                    'best': float(np.max(per_step_values)),
                    'mean': float(np.mean(per_step_values)),
                    'worst': float(np.min(per_step_values)),
                    'std': float(np.std(per_step_values)),
                }
            else:
                # For errors/penalties, lower is better
                metric_stats[key] = {
                    'best': float(np.min(per_step_values)),
                    'mean': float(np.mean(per_step_values)),
                    'worst': float(np.max(per_step_values)),
                    'std': float(np.std(per_step_values)),
                }

    # Print in fixed order
    metric_order = ['total_reward', 'lin_vel_mse', 'ang_vel_mse', 'collision', 'termination', 'feet_stumble', 'travel_distance']
    metric_names = {
        'total_reward': 'Total reward',
        'lin_vel_mse': 'Lin vel MSE (m²/s²)',
        'ang_vel_mse': 'Ang vel MSE (rad²/s²)',
        'collision': 'Collision count',
        'termination': 'Termination contact count',
        'feet_stumble': 'Feet stumble count',
        'travel_distance': 'Displacement (m)',
    }

    if SHOW_ALL:
        for key in metric_order:
            if key in metric_stats:
                stats = metric_stats[key]
                name = metric_names.get(key, key)
                print(f"{name:<30} {stats['best']:>10.4f} {stats['mean']:>10.4f} {stats['worst']:>10.4f}")

    # Compute per-displacement metrics (avg by displacement)
    if SHOW_ALL:
        print(f"\n{'─'*60}")
        print(f"METRICS (per meter displacement)")
        print(f"{'─'*60}")
        print(f"{'Metric':<30} {'Best':>10} {'Mean':>10} {'Worst':>10}")
        print(f"{'─'*60}")

    travel_dist_cpu = metrics['travel_distance'].cpu().numpy()
    per_disp_keys = ['collision', 'termination', 'feet_stumble']
    per_disp_names = {
        'collision': 'Collision / m',
        'termination': 'Termination / m',
        'feet_stumble': 'Stumble / m',
    }
    metric_per_disp_stats = {}
    for key in per_disp_keys:
        per_disp_values = metrics[key].cpu().numpy() / np.maximum(travel_dist_cpu, 0.01)
        metric_per_disp_stats[key] = {
            'best': float(np.min(per_disp_values)),
            'mean': float(np.mean(per_disp_values)),
            'worst': float(np.max(per_disp_values)),
            'std': float(np.std(per_disp_values)),
        }
        if SHOW_ALL:
            stats = metric_per_disp_stats[key]
            name = per_disp_names[key]
            print(f"{name:<30} {stats['best']:>10.4f} {stats['mean']:>10.4f} {stats['worst']:>10.4f}")

    if SHOW_ALL:
        print(f"{'='*60}\n")

    # Print Excel-friendly summary line (tab-separated)
    _sd = lambda s: s['std']
    print("EXCEL COPY (mean_reward, lin_vel_mse, ang_vel_mse, collision, termination, stumble, displacement, collision/m, termination/m, stumble/m, success_rate):")
    print(f"{mean_reward:.2f}±{std_reward:.2f}\n"
          f"{metric_stats['lin_vel_mse']['mean']:.4f}±{_sd(metric_stats['lin_vel_mse']):.4f}\n"
          f"{metric_stats['ang_vel_mse']['mean']:.4f}±{_sd(metric_stats['ang_vel_mse']):.4f}\n"
          f"{metric_stats['collision']['mean']:.4f}±{_sd(metric_stats['collision']):.4f}\n"
          f"{metric_stats['termination']['mean']:.4f}±{_sd(metric_stats['termination']):.4f}\n"
          f"{metric_stats['feet_stumble']['mean']:.4f}±{_sd(metric_stats['feet_stumble']):.4f}\n"
          f"{metric_stats['travel_distance']['mean']:.4f}±{_sd(metric_stats['travel_distance']):.4f}\n"
          f"{metric_per_disp_stats['collision']['mean']:.4f}±{_sd(metric_per_disp_stats['collision']):.4f}\n"
          f"{metric_per_disp_stats['termination']['mean']:.4f}±{_sd(metric_per_disp_stats['termination']):.4f}\n"
          f"{metric_per_disp_stats['feet_stumble']['mean']:.4f}±{_sd(metric_per_disp_stats['feet_stumble']):.4f}\n"
          f"{success_rate:.4f}")

    # Print all rewards if enabled
    if SHOW_ALL:
        print("All environment rewards:")
        for i, r in enumerate(rewards_cpu):
            print(f"  Env {i+1:3d}: {r:.2f}")
        print()

    # Prepare results dict
    results = {
        "task": args.task,
        "num_envs": num_envs,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "difficulty": difficulty,
            "vel_x": vel_x,
            "vel_y": vel_y,
            "vel_yaw": vel_yaw,
            "randomize": randomize,
            "add_noise": add_noise,
        },
        "statistics": {
            "total": {
                "best": best_reward,
                "worst": worst_reward,
                "mean": mean_reward,
                "median": median_reward,
                "std": std_reward,
            },
        },
        "rewards": rewards_cpu.tolist(),
        "elapsed_time": elapsed_time,
    }

    # Add per-step metrics to results
    results["statistics"]["metrics"] = {}
    for key in metric_order:
        if key in metric_stats:
            results["statistics"]["metrics"][key] = metric_stats[key]

    # Add per-displacement metrics to results
    results["statistics"]["metrics_per_displacement"] = {}
    for key in per_disp_keys:
        results["statistics"]["metrics_per_displacement"][key] = metric_per_disp_stats[key]

    results["statistics"]["success_rate"] = success_rate

    # Print full results as JSON
    if SHOW_ALL:
        print("FULL RESULTS (JSON):")
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    # ============================================
    # EVALUATION CONFIGURATION (Edit these values)
    # ============================================
    NUM_ENVS = 100          # Number of parallel environments
    DIFFICULTY = 0.8       # Terrain difficulty (0.0 - 1.0)
    VEL_X = 1.0           # Forward velocity command (m/s)
    VEL_Y = 0.0           # Lateral velocity command (m/s)
    VEL_YAW = 0.0         # Yaw velocity command (rad/s)
    RANDOMIZE = False     # Enable domain randomization
    ADD_NOISE = False     # Add observation noise
    SHOW_ALL = False       # Show all individual environment rewards
    # ============================================

    args = get_args()
    args.rl_device = args.sim_device
    evaluate(args)
