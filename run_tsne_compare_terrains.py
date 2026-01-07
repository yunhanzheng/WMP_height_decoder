"""
Compare recurrent states from different terrain configurations using t-SNE

This script collects states from two terrain setups and visualizes them together
with different colors to see how terrain affects the world model's internal representations.
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL: Import isaacgym FIRST
import isaacgym

# Import in the same order as play.py to avoid circular imports
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict


def main():
    # Import numpy here
    import numpy as np

    # Import torch AFTER isaacgym
    import torch

    # Import WMPRunner after torch
    from rsl_rl.runners import WMPRunner

    # Get environment args
    env_args = get_args()

    # Set visualization parameters
    num_steps = int(os.environ.get('TSNE_NUM_STEPS', '200'))
    save_path = os.environ.get('TSNE_SAVE_PATH', 'tsne_terrain_comparison.png')
    perplexity = int(os.environ.get('TSNE_PERPLEXITY', '30'))

    print(f"\n{'='*80}")
    print(f"t-SNE Terrain Comparison")
    print(f"{'='*80}")
    print(f"Visualization settings:")
    print(f"  - num_steps per terrain: {num_steps}")
    print(f"  - save_path: {save_path}")
    print(f"  - perplexity: {perplexity}")
    print(f"{'='*80}\n")

    # Get configs
    env_cfg, train_cfg = task_registry.get_cfgs(name=env_args.task)

    # Find checkpoint
    if env_args.checkpoint is None:
        print("Searching for latest checkpoint...")
        from legged_gym.utils.helpers import get_load_path
        from legged_gym import LEGGED_GYM_ROOT_DIR

        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        checkpoint_path = get_load_path(log_root, load_run=-1, checkpoint=-1)
        print(f"Found: {checkpoint_path}\n")
    else:
        checkpoint_path = env_args.checkpoint

    # Import visualization module
    from visualize_recurrent_state import RecurrentStateVisualizer

    # ========================================================================
    # Create environment and runner once (Isaac Gym limitation)
    # ========================================================================
    print(f"\n{'='*80}")
    print("Initializing environment and model...")
    print(f"{'='*80}")

    # Start with terrain 1 config
    env_cfg.terrain.difficulty = 0.1
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    env, _ = task_registry.make_env(name=env_args.task, args=env_args, env_cfg=env_cfg)

    train_cfg_dict = class_to_dict(train_cfg)
    train_cfg_dict['runner']['use_wandb'] = False

    runner = WMPRunner(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=os.path.dirname(checkpoint_path),
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Loading checkpoint...")
    runner.load(checkpoint_path)
    runner.alg.actor_critic.eval()

    # ========================================================================
    # Rollout 1: First behavioral pattern
    # ========================================================================
    print(f"\n{'='*80}")
    print("ROLLOUT 1: Domino Terrain")
    print(f"{'='*80}")

    visualizer_1 = RecurrentStateVisualizer(use_deter_only=True)

    # Collect states from first rollout
    print(f"Collecting {num_steps} steps from domino terrain...")
    collect_states(runner, visualizer_1, num_steps)

    states_1 = np.concatenate(visualizer_1.states, axis=0)
    print(f"Collected {states_1.shape[0]} samples from domino terrain\n")

    # Reset environment for different behavior
    runner.env.reset()

    # ========================================================================
    # Rollout 2: Second behavioral pattern
    # ========================================================================
    print(f"{'='*80}")
    print("ROLLOUT 2: Stripes Terrain")
    print(f"{'='*80}")

    visualizer_2 = RecurrentStateVisualizer(use_deter_only=True)

    # Collect states from second rollout
    print(f"Collecting {num_steps} steps from stripes terrain...")
    collect_states(runner, visualizer_2, num_steps)

    states_2 = np.concatenate(visualizer_2.states, axis=0)
    print(f"Collected {states_2.shape[0]} samples from stripes terrain\n")

    # ========================================================================
    # Combine and visualize
    # ========================================================================
    print(f"{'='*80}")
    print("Computing t-SNE on combined data...")
    print(f"{'='*80}")

    # Combine states
    all_states = np.concatenate([states_1, states_2], axis=0)
    terrain_labels = np.concatenate([
        np.zeros(states_1.shape[0]),  # 0 for terrain 1
        np.ones(states_2.shape[0])     # 1 for terrain 2
    ])

    print(f"Total samples: {all_states.shape[0]}")
    print(f"  - Domino terrain: {states_1.shape[0]}")
    print(f"  - Stripes terrain: {states_2.shape[0]}")

    # Compute t-SNE
    from sklearn.manifold import TSNE

    # Adjust perplexity if necessary
    n_samples = all_states.shape[0]
    max_perplexity = (n_samples - 1) // 3
    if perplexity >= n_samples:
        perplexity = min(30, max_perplexity)
        print(f"Adjusted perplexity to {perplexity}")

    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(all_states)

    # Plot
    print(f"\nCreating visualization...")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot rollout 1 (domino) in blue
    mask_1 = terrain_labels == 0
    ax.scatter(tsne_results[mask_1, 0], tsne_results[mask_1, 1],
               c='blue', alpha=0.6, s=20, label='Domino Terrain')

    # Plot rollout 2 (stripes) in red
    mask_2 = terrain_labels == 1
    ax.scatter(tsne_results[mask_2, 0], tsne_results[mask_2, 1],
               c='red', alpha=0.6, s=20, label='Stripes Terrain')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('World Model Recurrent States: Domino vs Stripes Terrain', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")

    # Print statistics
    print(f"\n{'='*80}")
    print("Analysis:")
    print(f"{'='*80}")

    # Calculate separation
    from scipy.spatial.distance import cdist
    centroid_1 = tsne_results[mask_1].mean(axis=0)
    centroid_2 = tsne_results[mask_2].mean(axis=0)
    separation = np.linalg.norm(centroid_1 - centroid_2)

    print(f"Centroid separation: {separation:.2f}")
    print(f"Average distance within domino terrain: {np.mean(cdist(tsne_results[mask_1], tsne_results[mask_1])):.2f}")
    print(f"Average distance within stripes terrain: {np.mean(cdist(tsne_results[mask_2], tsne_results[mask_2])):.2f}")

    if separation > 10:
        print("\n✓ Strong separation: World model clearly distinguishes between terrains")
    elif separation > 5:
        print("\n~ Moderate separation: Some terrain-specific patterns detected")
    else:
        print("\n✗ Weak separation: World model uses similar representations for both terrains")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}\n")


def collect_states(runner, visualizer, num_steps):
    """Collect recurrent states from a runner"""
    import torch

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

    if runner.env.cfg.depth.use_camera:
        wm_obs["image"] = torch.zeros(
            ((runner.env.num_envs,) + runner.env.cfg.depth.resized + (1,)),
            device=runner._world_model.device
        )

    wm_update_interval = runner.env.cfg.depth.update_interval
    wm_action_history = torch.zeros(
        size=(runner.env.num_envs, wm_update_interval, runner.env.num_actions),
        device=runner._world_model.device
    )

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

                # Collect state
                visualizer.collect_state(wm_latent)

            # Take action
            history = trajectory_history.flatten(1).to(runner.device)
            actions = runner.alg.act(obs, critic_obs, history, wm_feature.to(runner.env.device))
            obs, privileged_obs, rewards, dones, infos, reset_env_ids = runner.env.step(actions)

            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs = obs.to(runner.device), critic_obs.to(runner.device)

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
            reset_env_ids = reset_env_ids.cpu().numpy()
            if len(reset_env_ids) > 0:
                wm_action_history[reset_env_ids, :] = 0
                wm_is_first[reset_env_ids] = 1

            # Update trajectory history
            env_ids = dones.nonzero(as_tuple=False).flatten()
            trajectory_history[env_ids] = 0
            obs_without_command = torch.concat((obs[:, runner.env.privileged_dim:runner.env.privileged_dim + 6],
                                                obs[:, runner.env.privileged_dim + 9:runner.env.num_obs - runner.env.height_dim]),
                                               dim=1)
            trajectory_history = torch.concat(
                (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

            if step % 50 == 0:
                print(f"  Step {step}/{num_steps}")


if __name__ == "__main__":
    main()
