"""
Extended visualizer that collects both reward and feet_stumble data in a single rollout
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from visualize_recurrent_state import RecurrentStateVisualizer


class ComparisonVisualizer(RecurrentStateVisualizer):
    """Collects reward and feet_stumble data in a single rollout"""

    def __init__(self, use_deter_only=True):
        super().__init__(use_deter_only=use_deter_only)
        # Add feet_stumble to metadata
        self.metadata['feet_stumble'] = []

    def collect_state_with_all_metadata(self, wm_latent, reward, feet_stumble,
                                        timestep=None, episode=None, action=None):
        """Collect state with both reward and feet_stumble"""
        self.collect_state(wm_latent, reward, timestep, episode, action)

        if isinstance(feet_stumble, torch.Tensor):
            feet_stumble = feet_stumble.detach().cpu().numpy()
        self.metadata['feet_stumble'].append(feet_stumble)

    def plot_comparison(self, tsne_results, save_path_reward='tsne_reward.png',
                       save_path_stumble='tsne_stumble.png', figsize=(20, 8)):
        """
        Plot both reward and feet_stumble side by side from the same t-SNE

        Args:
            tsne_results: Output from compute_tsne()
            save_path_reward: Path to save reward plot
            save_path_stumble: Path to save stumble plot
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left plot: colored by reward
        if len(self.metadata['rewards']) > 0:
            rewards = np.concatenate(self.metadata['rewards'])
            if len(rewards.shape) > 1:
                rewards = rewards.flatten()
            scatter1 = ax1.scatter(tsne_results[:, 0], tsne_results[:, 1],
                                  c=rewards, cmap='RdYlGn', alpha=0.6, s=10)
            plt.colorbar(scatter1, ax=ax1, label='Reward')
            ax1.set_title('t-SNE colored by Reward', fontsize=14)

        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Right plot: colored by feet stumble
        if len(self.metadata['feet_stumble']) > 0:
            feet_stumble = np.concatenate(self.metadata['feet_stumble'])

            # Plot non-stumble points first (in blue)
            no_stumble_mask = ~feet_stumble.astype(bool)
            if np.any(no_stumble_mask):
                ax2.scatter(tsne_results[no_stumble_mask, 0],
                           tsne_results[no_stumble_mask, 1],
                           c='blue', alpha=0.3, s=10, label='Normal')

            # Plot stumble points on top (in red)
            stumble_mask = feet_stumble.astype(bool)
            if np.any(stumble_mask):
                ax2.scatter(tsne_results[stumble_mask, 0],
                           tsne_results[stumble_mask, 1],
                           c='red', alpha=0.7, s=15, label='Feet Stumble')

            ax2.set_title('t-SNE colored by Feet Stumble', fontsize=14)
            ax2.legend(loc='best', fontsize=10)

            # Add statistics
            num_stumbles = np.sum(feet_stumble)
            total_samples = len(feet_stumble)
            stumble_pct = 100 * num_stumbles / total_samples
            stats_text = f'Stumble rate: {num_stumbles}/{total_samples} ({stumble_pct:.2f}%)'
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save individual plots
        if save_path_reward:
            extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(save_path_reward, dpi=300, bbox_inches=extent1.expanded(1.1, 1.1))
            print(f"Saved reward plot to {save_path_reward}")

        if save_path_stumble:
            extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(save_path_stumble, dpi=300, bbox_inches=extent2.expanded(1.1, 1.1))
            print(f"Saved stumble plot to {save_path_stumble}")

        # Save combined plot
        combined_path = 'tsne_comparison.png'
        fig.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to {combined_path}")

        plt.show()

    def clear(self):
        """Clear all collected states and metadata"""
        super().clear()
        self.metadata['feet_stumble'] = []


def compute_feet_stumble(contact_forces, feet_indices):
    """
    Compute feet stumble condition

    Args:
        contact_forces: Contact forces tensor [num_envs, num_bodies, 3]
        feet_indices: Indices of feet bodies

    Returns:
        feet_stumble: Boolean tensor [num_envs]
    """
    feet_forces = contact_forces[:, feet_indices, :]
    horizontal_forces = torch.norm(feet_forces[:, :, :2], dim=2)
    vertical_forces = torch.abs(feet_forces[:, :, 2])
    stumble = torch.any(horizontal_forces > 5 * vertical_forces, dim=1)
    return stumble


def visualize_comparison_from_checkpoint(checkpoint_path, runner, num_steps=1000,
                                        use_deter_only=True, perplexity=30):
    """
    Collect states in ONE rollout and create both visualizations from the same data

    Args:
        checkpoint_path: Path to model checkpoint
        runner: WMPRunner instance
        num_steps: Number of steps to collect
        use_deter_only: Whether to use only deterministic state
        perplexity: t-SNE perplexity parameter

    Returns:
        visualizer, tsne_results
    """
    # Load checkpoint
    runner.load(checkpoint_path)
    runner.alg.actor_critic.eval()

    visualizer = ComparisonVisualizer(use_deter_only=use_deter_only)

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

    print(f"Collecting {num_steps} steps with both reward and feet stumble data in a SINGLE rollout...")

    # Initialize rewards for first iteration
    rewards = torch.zeros(runner.env.num_envs, device=runner.device)

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

                # Compute feet stumble from CURRENT contact forces
                feet_stumble = compute_feet_stumble(
                    runner.env.contact_forces,
                    runner.env.feet_indices
                )

                # Collect state immediately (matching original script order)
                # Note: This collects state BEFORE taking the next action
                visualizer.collect_state_with_all_metadata(
                    wm_latent,
                    reward=rewards,
                    feet_stumble=feet_stumble,
                    timestep=step * torch.ones(runner.env.num_envs),
                    episode=torch.arange(runner.env.num_envs)
                )

            # Take action
            history = trajectory_history.flatten(1).to(runner.device)
            actions = runner.alg.act(obs, critic_obs, history, wm_feature.to(runner.env.device))
            obs, privileged_obs, rewards, dones, infos, reset_env_ids = runner.env.step(actions)

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

    print("Computing t-SNE (once for both visualizations)...")
    tsne_results = visualizer.compute_tsne(perplexity=perplexity)

    print("Creating comparison plots...")
    visualizer.plot_comparison(tsne_results)

    return visualizer, tsne_results
