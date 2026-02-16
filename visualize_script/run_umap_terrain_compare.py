"""
UMAP visualization comparing world model states on different terrain types:
- Flat terrain (rough_flat, index 1)
- Discrete one obstacle terrain (index 7)

Uses a single environment with mixed terrain types, tracks which terrain each robot is on.

Usage:
    UMAP_NUM_STEPS=230 python run_umap_terrain_compare.py --task=go2_blind --load_run=Jan06_16-32-57_ --headless --num_envs=50
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL: Import isaacgym FIRST
import isaacgym

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict


def main():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import umap
    from rsl_rl.runners import WMPRunner
    from legged_gym.utils.helpers import get_load_path
    from legged_gym import LEGGED_GYM_ROOT_DIR

    env_args = get_args()

    num_steps = int(os.environ.get('UMAP_NUM_STEPS', '230'))
    save_path = os.environ.get('UMAP_SAVE_PATH', 'umap_terrain_compare.png')
    n_neighbors = int(os.environ.get('UMAP_N_NEIGHBORS', '30'))

    print(f"\nTerrain Comparison Visualization")
    print(f"  - num_steps: {num_steps}")
    print(f"  - state: compressed deterministic (wm_feature_encoder output)")
    print(f"  - save_path: {save_path}")

    # ============ SETUP ENVIRONMENT WITH MIXED TERRAIN ============
    print("\n" + "="*50)
    print("Setting up mixed terrain environment...")
    print("  - 50% Flat (rough_flat)")
    print("  - 50% Obstacle (discrete_one_obstacle)")
    print("="*50)

    env_cfg, train_cfg = task_registry.get_cfgs(name=env_args.task)

    # Use multiple rows with mixed terrain types
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_length = 4
    env_cfg.terrain.terrain_width = 4
    env_cfg.terrain.curriculum = False

    # Mixed terrain: 50% rough_flat (index 1), 50% obstacle (index 7)
    # terrain_proportions: [smooth, rough_flat, stairs_up, stairs_down, discrete, rough_stairs, stepping_stones, obstacles]
    env_cfg.terrain.terrain_proportions = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    env_cfg.terrain.difficulty = 0.3

    env_cfg.noise.add_noise = False
    env_cfg.depth.update_interval = 5
    env_cfg.domain_rand.friction_range = [0.8, 0.8]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0., 0.]
    env_cfg.domain_rand.com_x_pos_range = [0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [0.0, 0.0]
    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_restitution = True
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_com_pos = True
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_x = [0.8, 0.8]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    env_cfg.init_state.randomize_position = False
    train_cfg.runner.amp_num_preload_transitions = 1

    env, _ = task_registry.make_env(name=env_args.task, args=env_args, env_cfg=env_cfg)

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    load_run = env_args.load_run if env_args.load_run else -1
    if env_args.checkpoint is None:
        checkpoint_path = get_load_path(log_root, load_run=load_run, checkpoint=-1)
    else:
        checkpoint_path = get_load_path(log_root, load_run=load_run, checkpoint=env_args.checkpoint)

    train_cfg_dict = class_to_dict(train_cfg)
    train_cfg_dict['runner']['use_wandb'] = False

    runner = WMPRunner(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=os.path.dirname(checkpoint_path),
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    runner.load(checkpoint_path)
    runner.alg.actor_critic.eval()

    # ============ COLLECT DATA ============
    print(f"\nCollecting {num_steps} steps...")

    states = []
    terrain_labels = []

    # Initialize
    obs = runner.env.get_observations()
    privileged_obs = runner.env.get_privileged_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs, critic_obs = obs.to(runner.device), critic_obs.to(runner.device)

    # Initialize trajectory history
    trajectory_history = torch.zeros(
        size=(runner.env.num_envs, runner.history_length,
              runner.env.num_obs - runner.env.privileged_dim - runner.env.height_dim - 3),
        device=runner.device
    )

    # Initialize world model
    wm_latent = wm_action = None
    wm_is_first = torch.ones(runner.env.num_envs, device=runner._world_model.device)
    wm_obs = {
        "prop": obs[:, runner.env.privileged_dim: runner.env.privileged_dim + runner.env.cfg.env.prop_dim].to(runner._world_model.device),
        "is_first": wm_is_first,
    }

    wm_update_interval = runner.env.cfg.depth.update_interval
    wm_action_history = torch.zeros(
        size=(runner.env.num_envs, wm_update_interval, runner.env.num_actions),
        device=runner._world_model.device
    )

    wm_feature = torch.zeros((runner.env.num_envs, runner.wm_feature_dim), device=runner.device)

    with torch.no_grad():
        for step in range(num_steps):
            if step % wm_update_interval == 0:
                # World model obs step
                wm_embed = runner._world_model.encoder(wm_obs)
                wm_latent, _ = runner._world_model.dynamics.obs_step(
                    wm_latent, wm_action, wm_embed, wm_obs["is_first"]
                )
                wm_feature = runner._world_model.dynamics.get_deter_feat(wm_latent)
                wm_is_first[:] = 0

                # Compress deter state through wm_feature_encoder (same as actor-critic)
                compressed_deter = runner.alg.actor_critic.wm_feature_encoder(wm_feature)
                states.append(compressed_deter.detach().cpu().numpy())

                # Determine terrain type based on terrain_levels
                # terrain_levels tracks which row each robot is on
                # We use measured_heights to detect if there's an obstacle nearby
                heights = runner.env.measured_heights.detach().cpu().numpy()
                max_height = np.max(heights, axis=1)
                # If max height > threshold, robot is on obstacle terrain
                terrain_type = (max_height > 0.03).astype(int)  # 0=flat, 1=obstacle
                terrain_labels.append(terrain_type)

            # Take action
            history = trajectory_history.flatten(1).to(runner.device)
            actions = runner.alg.act(obs, critic_obs, history, wm_feature.to(runner.env.device))
            obs, privileged_obs, rewards, dones, infos, reset_env_ids, _ = runner.env.step(actions)

            # Update critic obs
            critic_obs = privileged_obs if privileged_obs is not None else obs

            # Update world model input
            wm_action_history = torch.concat(
                (wm_action_history[:, 1:], actions.unsqueeze(1).to(runner._world_model.device)),
                dim=1
            )
            wm_action = wm_action_history.flatten(1)

            wm_obs = {
                "prop": obs[:, runner.env.privileged_dim: runner.env.privileged_dim + runner.env.cfg.env.prop_dim].to(runner._world_model.device),
                "is_first": wm_is_first,
            }

            # Handle resets
            if len(reset_env_ids) > 0:
                wm_action_history[reset_env_ids, :] = 0
                wm_is_first[reset_env_ids] = 1
                trajectory_history[reset_env_ids] = 0

            # Update trajectory history
            traj_end_idx = runner.env.num_obs - runner.env.height_dim
            obs_without_command = torch.concat(
                (obs[:, runner.env.privileged_dim:runner.env.privileged_dim + 6],
                 obs[:, runner.env.privileged_dim + 9:traj_end_idx]),
                dim=1
            )
            trajectory_history = torch.concat(
                (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)),
                dim=1
            )

            if step % 100 == 0:
                print(f"Step {step}/{num_steps}")

    # ============ COMPUTE UMAP ============
    print("\n" + "="*50)
    print("Computing UMAP...")
    print("="*50)

    combined_states = np.concatenate(states, axis=0)
    combined_labels = np.concatenate(terrain_labels, axis=0)

    print(f"Total samples: {combined_states.shape[0]}")
    print(f"  - Flat terrain: {np.sum(combined_labels == 0)}")
    print(f"  - Obstacle terrain: {np.sum(combined_labels == 1)}")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='euclidean',
        n_components=2,
        random_state=42
    )
    umap_results = reducer.fit_transform(combined_states)

    # ============ PLOT ============
    print("Plotting...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot flat terrain (blue)
    flat_mask = combined_labels == 0
    ax.scatter(umap_results[flat_mask, 0], umap_results[flat_mask, 1],
              c='blue', alpha=0.5, s=10, label='Flat Terrain')

    # Plot obstacle terrain (red)
    obs_mask = combined_labels == 1
    ax.scatter(umap_results[obs_mask, 0], umap_results[obs_mask, 1],
              c='red', alpha=0.5, s=10, label='Obstacle Terrain')

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('UMAP of Compressed Deterministic State\n(Flat vs Obstacle Terrain)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Add statistics
    stats_text = (
        f'Flat: {np.sum(flat_mask)} samples\n'
        f'Obstacle: {np.sum(obs_mask)} samples'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
