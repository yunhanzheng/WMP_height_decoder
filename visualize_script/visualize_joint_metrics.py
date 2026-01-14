"""
Extended visualizer that collects joint position, velocity, and acceleration data in a single rollout
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from visualize_recurrent_state import RecurrentStateVisualizer


class JointMetricsVisualizer(RecurrentStateVisualizer):
    """Collects joint position, velocity, and acceleration data in a single rollout"""

    def __init__(self, use_deter_only=True):
        super().__init__(use_deter_only=use_deter_only)
        # Add joint metrics to metadata
        self.metadata['joint_positions'] = []
        self.metadata['joint_velocities'] = []
        self.metadata['joint_accelerations'] = []

    def collect_state_with_joint_metrics(self, wm_latent, joint_pos, joint_vel, joint_acc,
                                        reward=None, timestep=None, episode=None, action=None):
        """Collect state with joint position, velocity, and acceleration"""
        self.collect_state(wm_latent, reward, timestep, episode, action)

        # Convert to numpy if needed
        if isinstance(joint_pos, torch.Tensor):
            joint_pos = joint_pos.detach().cpu().numpy()
        if isinstance(joint_vel, torch.Tensor):
            joint_vel = joint_vel.detach().cpu().numpy()
        if isinstance(joint_acc, torch.Tensor):
            joint_acc = joint_acc.detach().cpu().numpy()

        self.metadata['joint_positions'].append(joint_pos)
        self.metadata['joint_velocities'].append(joint_vel)
        self.metadata['joint_accelerations'].append(joint_acc)

    def plot_joint_metrics(self, tsne_results,
                          save_path_pos='tsne_joint_position.png',
                          save_path_vel='tsne_joint_velocity.png',
                          save_path_acc='tsne_joint_acceleration.png',
                          figsize=(24, 7)):
        """
        Plot three graphs: position, velocity, and acceleration

        Args:
            tsne_results: Output from compute_tsne()
            save_path_pos: Path to save position plot
            save_path_vel: Path to save velocity plot
            save_path_acc: Path to save acceleration plot
            figsize: Figure size
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Compute average metrics across all joints
        joint_positions = np.concatenate(self.metadata['joint_positions'])  # [samples, num_joints]
        joint_velocities = np.concatenate(self.metadata['joint_velocities'])
        joint_accelerations = np.concatenate(self.metadata['joint_accelerations'])

        # Compute average across joints (mean absolute value)
        avg_pos = np.mean(np.abs(joint_positions), axis=1)  # [samples]
        avg_vel = np.mean(np.abs(joint_velocities), axis=1)
        avg_acc = np.mean(np.abs(joint_accelerations), axis=1)

        # Plot 1: Average Joint Position
        scatter1 = ax1.scatter(tsne_results[:, 0], tsne_results[:, 1],
                              c=avg_pos, cmap='viridis', alpha=0.6, s=10)
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='Avg |Joint Position| (rad)')
        ax1.set_title('t-SNE colored by Avg Joint Position', fontsize=14)
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Add statistics
        stats_text1 = f'Mean: {np.mean(avg_pos):.3f}\nStd: {np.std(avg_pos):.3f}\nMin: {np.min(avg_pos):.3f}\nMax: {np.max(avg_pos):.3f}'
        ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Average Joint Velocity
        scatter2 = ax2.scatter(tsne_results[:, 0], tsne_results[:, 1],
                              c=avg_vel, cmap='plasma', alpha=0.6, s=10)
        cbar2 = plt.colorbar(scatter2, ax=ax2, label='Avg |Joint Velocity| (rad/s)')
        ax2.set_title('t-SNE colored by Avg Joint Velocity', fontsize=14)
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add statistics
        stats_text2 = f'Mean: {np.mean(avg_vel):.3f}\nStd: {np.std(avg_vel):.3f}\nMin: {np.min(avg_vel):.3f}\nMax: {np.max(avg_vel):.3f}'
        ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 3: Average Joint Acceleration
        scatter3 = ax3.scatter(tsne_results[:, 0], tsne_results[:, 1],
                              c=avg_acc, cmap='coolwarm', alpha=0.6, s=10)
        cbar3 = plt.colorbar(scatter3, ax=ax3, label='Avg |Joint Acceleration| (rad/s²)')
        ax3.set_title('t-SNE colored by Avg Joint Acceleration', fontsize=14)
        ax3.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax3.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Add statistics
        stats_text3 = f'Mean: {np.mean(avg_acc):.3f}\nStd: {np.std(avg_acc):.3f}\nMin: {np.min(avg_acc):.3f}\nMax: {np.max(avg_acc):.3f}'
        ax3.text(0.02, 0.98, stats_text3, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save individual plots
        if save_path_pos:
            extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(save_path_pos, dpi=300, bbox_inches=extent1.expanded(1.15, 1.15))
            print(f"Saved position plot to {save_path_pos}")

        if save_path_vel:
            extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(save_path_vel, dpi=300, bbox_inches=extent2.expanded(1.15, 1.15))
            print(f"Saved velocity plot to {save_path_vel}")

        if save_path_acc:
            extent3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(save_path_acc, dpi=300, bbox_inches=extent3.expanded(1.15, 1.15))
            print(f"Saved acceleration plot to {save_path_acc}")

        # Save combined plot
        combined_path = 'tsne_joint_metrics.png'
        fig.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to {combined_path}")

        plt.show()

    def clear(self):
        """Clear all collected states and metadata"""
        super().clear()
        self.metadata['joint_positions'] = []
        self.metadata['joint_velocities'] = []
        self.metadata['joint_accelerations'] = []


def visualize_joint_metrics_from_checkpoint(checkpoint_path, runner, num_steps=1000,
                                           use_deter_only=True, perplexity=30):
    """
    Collect states in ONE rollout and create joint metrics visualizations

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

    visualizer = JointMetricsVisualizer(use_deter_only=use_deter_only)

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

    print(f"Collecting {num_steps} steps with joint position, velocity, and acceleration data...")

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

                # Get current joint positions and velocities
                joint_pos = runner.env.dof_pos - runner.env.default_dof_pos  # Relative to default
                joint_vel = runner.env.dof_vel

                # Compute joint acceleration (change in velocity / dt)
                joint_acc = (joint_vel - prev_joint_vel) / dt
                prev_joint_vel = joint_vel.clone()

                # Collect state with joint metrics
                visualizer.collect_state_with_joint_metrics(
                    wm_latent,
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    joint_acc=joint_acc,
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

    print("Computing t-SNE (once for all three visualizations)...")
    tsne_results = visualizer.compute_tsne(perplexity=perplexity)

    print("Creating joint metrics plots...")
    visualizer.plot_joint_metrics(tsne_results)

    return visualizer, tsne_results
