"""
Extended RecurrentStateVisualizer that tracks and visualizes high acceleration events
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from visualize_recurrent_state import RecurrentStateVisualizer


class HighAccelerationVisualizer(RecurrentStateVisualizer):
    """Extends RecurrentStateVisualizer to track high acceleration events"""

    def __init__(self, use_deter_only=True):
        super().__init__(use_deter_only=use_deter_only)
        # Add high_acceleration to metadata
        self.metadata['high_acceleration'] = []

    def collect_state_with_acceleration(self, wm_latent, high_acceleration, reward=None,
                                       timestep=None, episode=None, action=None):
        """
        Collect a recurrent state with high acceleration information

        Args:
            wm_latent: Dictionary containing 'deter' and 'stoch' keys
            high_acceleration: Boolean tensor indicating if acceleration exceeds threshold [batch]
            reward: Optional reward at this timestep
            timestep: Optional timestep index
            episode: Optional episode index
            action: Optional action taken
        """
        # Use parent class method for state collection
        self.collect_state(wm_latent, reward, timestep, episode, action)

        # Add high acceleration data
        if isinstance(high_acceleration, torch.Tensor):
            high_acceleration = high_acceleration.detach().cpu().numpy()
        self.metadata['high_acceleration'].append(high_acceleration)

    def plot_tsne_acceleration(self, tsne_results, save_path=None, figsize=(12, 8)):
        """
        Plot t-SNE results colored by high acceleration events

        Args:
            tsne_results: Output from compute_tsne()
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if len(self.metadata['high_acceleration']) == 0:
            raise ValueError("No acceleration data collected")

        fig, ax = plt.subplots(figsize=figsize)

        # Get high acceleration data
        high_accel = np.concatenate(self.metadata['high_acceleration'])

        # Plot low acceleration points first (in blue)
        low_mask = ~high_accel.astype(bool)
        if np.any(low_mask):
            ax.scatter(tsne_results[low_mask, 0],
                      tsne_results[low_mask, 1],
                      c='blue', alpha=0.3, s=10, label='Low Acceleration')

        # Plot high acceleration points on top (in red) so they're more visible
        high_mask = high_accel.astype(bool)
        if np.any(high_mask):
            ax.scatter(tsne_results[high_mask, 0],
                      tsne_results[high_mask, 1],
                      c='red', alpha=0.7, s=15, label='High Acceleration')

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE of Recurrent States (colored by High Acceleration)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Add statistics text
        num_high = np.sum(high_accel)
        total_samples = len(high_accel)
        high_pct = 100 * num_high / total_samples
        stats_text = f'High acceleration rate: {num_high}/{total_samples} ({high_pct:.2f}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def clear(self):
        """Clear all collected states and metadata"""
        super().clear()
        self.metadata['high_acceleration'] = []


def compute_high_acceleration(dof_vel, prev_dof_vel, dt, threshold=10.0):
    """
    Compute whether any joint has acceleration exceeding the threshold

    Args:
        dof_vel: Current joint velocities [num_envs, num_joints]
        prev_dof_vel: Previous joint velocities [num_envs, num_joints]
        dt: Time step
        threshold: Acceleration threshold in rad/s² (default: 10.0)

    Returns:
        high_acceleration: Boolean tensor [num_envs] indicating if any joint exceeds threshold
    """
    # Compute joint accelerations
    joint_acc = (dof_vel - prev_dof_vel) / dt  # [num_envs, num_joints]

    # Check if ANY joint exceeds threshold (using absolute value)
    max_acc = torch.max(torch.abs(joint_acc), dim=1)[0]  # [num_envs]
    high_acceleration = max_acc > threshold

    return high_acceleration


def visualize_high_acceleration_from_checkpoint(checkpoint_path, runner, num_steps=1000,
                                               use_deter_only=True, save_path=None,
                                               perplexity=30, acceleration_threshold=10.0):
    """
    Collect states from a trained model and visualize with high acceleration coloring

    Args:
        checkpoint_path: Path to model checkpoint
        runner: WMPRunner instance
        num_steps: Number of steps to collect
        use_deter_only: Whether to use only deterministic state
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        acceleration_threshold: Acceleration threshold in rad/s² for detection (default: 10.0)
    """
    # Load checkpoint
    runner.load(checkpoint_path)
    runner.alg.actor_critic.eval()

    visualizer = HighAccelerationVisualizer(use_deter_only=use_deter_only)

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

    print(f"Collecting {num_steps} steps of recurrent states and acceleration data...")

    # Track previous joint velocities for acceleration computation
    prev_joint_vel = runner.env.dof_vel.clone()
    dt = runner.env.dt  # timestep

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

                # Compute high acceleration
                high_acceleration = compute_high_acceleration(
                    runner.env.dof_vel,
                    prev_joint_vel,
                    dt,
                    acceleration_threshold
                )

                # Update previous velocity
                prev_joint_vel = runner.env.dof_vel.clone()

                # Collect state with high acceleration information
                visualizer.collect_state_with_acceleration(
                    wm_latent,
                    high_acceleration=high_acceleration,
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
                # Reset previous velocity for acceleration computation
                prev_joint_vel[reset_env_ids] = runner.env.dof_vel[reset_env_ids].clone()

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
    visualizer.plot_tsne_acceleration(tsne_results, save_path=save_path)

    return visualizer, tsne_results
