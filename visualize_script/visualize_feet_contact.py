"""
Extended RecurrentStateVisualizer that tracks and visualizes the number of feet in contact with the ground
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from visualize_recurrent_state import RecurrentStateVisualizer


class FeetContactVisualizer(RecurrentStateVisualizer):
    """Extends RecurrentStateVisualizer to track number of feet in contact"""

    def __init__(self, use_deter_only=True):
        super().__init__(use_deter_only=use_deter_only)
        # Add feet_contact_count to metadata
        self.metadata['feet_contact_count'] = []

    def collect_state_with_contact(self, wm_latent, feet_contact_count, reward=None,
                                   timestep=None, episode=None, action=None):
        """
        Collect a recurrent state with feet contact count information

        Args:
            wm_latent: Dictionary containing 'deter' and 'stoch' keys
            feet_contact_count: Integer tensor indicating number of feet in contact [batch]
            reward: Optional reward at this timestep
            timestep: Optional timestep index
            episode: Optional episode index
            action: Optional action taken
        """
        # Use parent class method for state collection
        self.collect_state(wm_latent, reward, timestep, episode, action)

        # Add feet contact count data
        if isinstance(feet_contact_count, torch.Tensor):
            feet_contact_count = feet_contact_count.detach().cpu().numpy()
        self.metadata['feet_contact_count'].append(feet_contact_count)

    def plot_tsne_contact(self, tsne_results, save_path=None, figsize=(12, 8)):
        """
        Plot t-SNE results colored by number of feet in contact with the ground

        Args:
            tsne_results: Output from compute_tsne()
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if len(self.metadata['feet_contact_count']) == 0:
            raise ValueError("No feet contact data collected")

        fig, ax = plt.subplots(figsize=figsize)

        # Get feet contact count data
        contact_counts = np.concatenate(self.metadata['feet_contact_count'])

        # Define colors for different contact counts (0-4 feet)
        colors = {
            0: 'red',      # No feet in contact (airborne)
            1: 'orange',   # One foot in contact
            2: 'yellow',   # Two feet in contact
            3: 'lightgreen',  # Three feet in contact
            4: 'blue'      # All four feet in contact
        }
        labels = {
            0: '0 Feet (Airborne)',
            1: '1 Foot',
            2: '2 Feet',
            3: '3 Feet',
            4: '4 Feet (All contact)'
        }

        # Plot each contact count category
        for contact_num in range(5):  # 0 to 4 feet
            mask = contact_counts == contact_num
            if np.any(mask):
                ax.scatter(tsne_results[mask, 0],
                          tsne_results[mask, 1],
                          c=colors[contact_num],
                          alpha=0.6,
                          s=10,
                          label=labels[contact_num])

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE of Recurrent States (colored by Number of Feet in Contact)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Add statistics text
        total_samples = len(contact_counts)
        stats_lines = []
        for contact_num in range(5):
            count = np.sum(contact_counts == contact_num)
            pct = 100 * count / total_samples
            stats_lines.append(f'{labels[contact_num]}: {count} ({pct:.1f}%)')

        stats_text = '\n'.join(stats_lines)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def clear(self):
        """Clear all collected states and metadata"""
        super().clear()
        self.metadata['feet_contact_count'] = []


def compute_feet_contact_count(contact_forces, feet_indices, contact_threshold=1.0):
    """
    Compute the number of feet in contact with the ground

    Args:
        contact_forces: Contact forces tensor [num_envs, num_bodies, 3]
        feet_indices: Indices of feet bodies
        contact_threshold: Force threshold to consider a foot in contact (default: 1.0N)

    Returns:
        contact_count: Integer tensor [num_envs] indicating number of feet in contact
    """
    # Extract feet contact forces (z-component is vertical)
    feet_z_forces = contact_forces[:, feet_indices, 2]  # [num_envs, num_feet]

    # Check which feet are in contact (vertical force > threshold)
    feet_in_contact = feet_z_forces > contact_threshold  # [num_envs, num_feet]

    # Count number of feet in contact for each environment
    contact_count = torch.sum(feet_in_contact, dim=1)  # [num_envs]

    return contact_count


def visualize_feet_contact_from_checkpoint(checkpoint_path, runner, num_steps=1000,
                                          use_deter_only=True, save_path=None,
                                          perplexity=30, contact_threshold=1.0):
    """
    Collect states from a trained model and visualize with feet contact coloring

    Args:
        checkpoint_path: Path to model checkpoint
        runner: WMPRunner instance
        num_steps: Number of steps to collect
        use_deter_only: Whether to use only deterministic state
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        contact_threshold: Force threshold for foot contact detection (default: 1.0N)
    """
    # Load checkpoint
    runner.load(checkpoint_path)
    runner.alg.actor_critic.eval()

    visualizer = FeetContactVisualizer(use_deter_only=use_deter_only)

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

    print(f"Collecting {num_steps} steps of recurrent states and feet contact data...")

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

                # Compute number of feet in contact
                feet_contact_count = compute_feet_contact_count(
                    runner.env.contact_forces,
                    runner.env.feet_indices,
                    contact_threshold
                )

                # Collect state with feet contact count information
                visualizer.collect_state_with_contact(
                    wm_latent,
                    feet_contact_count=feet_contact_count,
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

    print("Computing t-SNE...")
    tsne_results = visualizer.compute_tsne(perplexity=perplexity)

    print("Plotting...")
    visualizer.plot_tsne_contact(tsne_results, save_path=save_path)

    return visualizer, tsne_results
