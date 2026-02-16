"""
Visualizer that tracks and visualizes feet stumble / obstacle proximity events
using the compressed deterministic state from wm_feature_encoder (actor-critic input).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap


class FeetStumbleVisualizer:
    """Visualizer that tracks obstacle proximity events using compressed deter state"""

    def __init__(self):
        self.states = []
        self.metadata = {
            'feet_stumble': [],
            'timestep': [],
            'episode': [],
        }

    def collect_state(self, compressed_deter, feet_stumble,
                      timestep=None, episode=None):
        """
        Collect compressed deterministic state with obstacle proximity information

        Args:
            compressed_deter: Compressed deterministic state from wm_feature_encoder [batch, compressed_dim]
            feet_stumble: Boolean tensor indicating if near obstacle [batch]
            timestep: Optional timestep index
            episode: Optional episode index
        """
        if isinstance(compressed_deter, torch.Tensor):
            state = compressed_deter.detach().cpu().numpy()
        else:
            state = compressed_deter
        self.states.append(state)

        # Add feet stumble data
        if isinstance(feet_stumble, torch.Tensor):
            feet_stumble = feet_stumble.detach().cpu().numpy()
        self.metadata['feet_stumble'].append(feet_stumble)

        if timestep is not None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.detach().cpu().numpy()
            self.metadata['timestep'].append(timestep)

        if episode is not None:
            if isinstance(episode, torch.Tensor):
                episode = episode.detach().cpu().numpy()
            self.metadata['episode'].append(episode)

    def compute_umap(self, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        """Compute UMAP embedding of collected states"""
        if len(self.states) == 0:
            raise ValueError("No states collected")

        all_states = np.concatenate(self.states, axis=0)
        print(f"Computing UMAP for {all_states.shape[0]} samples with {all_states.shape[1]} dimensions...")

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=2,
            random_state=42
        )
        return reducer.fit_transform(all_states)

    def plot_umap_stumble(self, umap_results, save_path=None, figsize=(12, 8)):
        """
        Plot UMAP results colored by near-obstacle events (terrain scan height)

        Args:
            umap_results: Output from compute_umap()
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if len(self.metadata['feet_stumble']) == 0:
            raise ValueError("No obstacle proximity data collected")

        fig, ax = plt.subplots(figsize=figsize)

        # Get obstacle proximity data
        near_obstacle = np.concatenate(self.metadata['feet_stumble'])

        # Plot far-from-obstacle points first (in blue)
        far_mask = ~near_obstacle.astype(bool)
        if np.any(far_mask):
            ax.scatter(umap_results[far_mask, 0],
                      umap_results[far_mask, 1],
                      c='blue', alpha=0.3, s=10, label='Far from Obstacles')

        # Plot near-obstacle points on top (in red) so they're more visible
        near_mask = near_obstacle.astype(bool)
        if np.any(near_mask):
            ax.scatter(umap_results[near_mask, 0],
                      umap_results[near_mask, 1],
                      c='red', alpha=0.7, s=15, label='Near Obstacles')

        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title('UMAP of Compressed Deterministic State (colored by Obstacle Proximity)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Add statistics text
        num_near = np.sum(near_obstacle)
        total_samples = len(near_obstacle)
        near_pct = 100 * num_near / total_samples
        stats_text = f'Near obstacle rate: {num_near}/{total_samples} ({near_pct:.2f}%)'
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
        self.states = []
        self.metadata = {
            'feet_stumble': [],
            'timestep': [],
            'episode': [],
        }


def compute_near_obstacle(measured_heights, threshold=0.05, close_range_x=(-0.01, 0.01), close_range_y=(-0.5, 0.5)):
    """
    Compute whether the robot is near an obstacle based on terrain scan height in close range

    The scan grid is organized as:
    - X (forward): -0.8 to 0.8 meters (17 points, index 0-16)
    - Y (lateral): -0.5 to 0.5 meters (11 points, index 0-10)
    - Grid layout: heights[env_id, x_idx * 11 + y_idx]

    Args:
        measured_heights: Terrain height measurements [num_envs, 187]
        threshold: Height threshold in meters (default: 0.05m)
        close_range_x: Tuple (min_x, max_x) defining forward range to check (default: 0.0-0.4m)
        close_range_y: Tuple (min_y, max_y) defining lateral range to check (default: -0.3 to 0.3m)

    Returns:
        near_obstacle: Boolean tensor [num_envs] indicating if near obstacle
    """
    # Scan grid parameters from go2_base.py
    x_points = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    y_points = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Find indices of points in the close range
    close_indices = []
    for x_idx, x in enumerate(x_points):
        if close_range_x[0] <= x <= close_range_x[1]:
            for y_idx, y in enumerate(y_points):
                if close_range_y[0] <= y <= close_range_y[1]:
                    # Height array is organized as [x_idx * num_y + y_idx]
                    height_idx = x_idx * len(y_points) + y_idx
                    close_indices.append(height_idx)

    # Extract heights at close-range points
    close_heights = measured_heights[:, close_indices]  # [num_envs, num_close_points]

    # Check if ANY close point exceeds threshold (max height in close range)
    max_close_height = torch.max(close_heights, dim=1)[0]  # [num_envs]
    near_obstacle = max_close_height > threshold

    return near_obstacle


def visualize_feet_stumble_from_checkpoint(checkpoint_path, runner, num_steps=1000,
                                          save_path=None, n_neighbors=15,
                                          height_threshold=0.05):
    """
    Collect compressed deterministic states from a trained model and visualize
    with near-obstacle coloring.

    Uses the compressed deterministic state (output of wm_feature_encoder)
    that is actually given to the actor-critic.

    Args:
        checkpoint_path: Path to model checkpoint
        runner: WMPRunner instance
        num_steps: Number of steps to collect
        save_path: Path to save the plot
        n_neighbors: UMAP n_neighbors parameter
        height_threshold: Height threshold in meters for obstacle detection (default: 0.05m)
    """
    # Load checkpoint
    runner.load(checkpoint_path)
    runner.alg.actor_critic.eval()

    visualizer = FeetStumbleVisualizer()

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

    print(f"Collecting {num_steps} steps of recurrent states and obstacle proximity data...")

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

                # Compute near-obstacle based on terrain scan height
                near_obstacle = compute_near_obstacle(
                    runner.env.measured_heights,
                    threshold=height_threshold
                )

                # Collect compressed deter state with obstacle proximity information
                visualizer.collect_state(
                    compressed_deter,
                    feet_stumble=near_obstacle,
                    timestep=step * torch.ones(runner.env.num_envs),
                    episode=torch.arange(runner.env.num_envs)
                )

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

    print("Computing UMAP...")
    umap_results = visualizer.compute_umap(n_neighbors=n_neighbors)

    print("Plotting...")
    visualizer.plot_umap_stumble(umap_results, save_path=save_path)

    return visualizer, umap_results
