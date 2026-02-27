"""
World model recurrent state visualizer that tracks termination, collision, and stumble events
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap


class TerminationEventsVisualizer:
    """Visualizer that tracks termination, collision, and stumble events

    Uses the compressed deterministic state from wm_feature_encoder,
    which is the representation actually given to the actor-critic.
    """

    def __init__(self):
        self.states = []
        self.metadata = {
            'termination': [],
            'collision': [],
            'stumble': [],
            'timestep': [],
            'episode': [],
            'joint_avg': [],
            'joint_min': [],
            'joint_max': [],
            'obstacle_phase': [],  # 0=before, 1=close, 2=after
        }

    def collect_state(self, compressed_deter, termination, collision, stumble,
                      timestep=None, episode=None, joint_angles=None, obstacle_phase=None):
        """
        Collect the compressed deterministic state with event information

        Args:
            compressed_deter: Compressed deterministic state from wm_feature_encoder [batch, compressed_dim]
            termination: Boolean tensor indicating termination [batch]
            collision: Boolean tensor indicating collision [batch]
            stumble: Boolean tensor indicating stumble [batch]
            timestep: Optional timestep index
            episode: Optional episode index
            joint_angles: Optional joint angles tensor [batch, num_joints]
            obstacle_phase: Optional obstacle phase tensor [batch] (0=before, 1=close, 2=after)
        """
        if isinstance(compressed_deter, torch.Tensor):
            state = compressed_deter.detach().cpu().numpy()
        else:
            state = compressed_deter

        self.states.append(state)

        # Convert tensors to numpy
        if isinstance(termination, torch.Tensor):
            termination = termination.detach().cpu().numpy()
        if isinstance(collision, torch.Tensor):
            collision = collision.detach().cpu().numpy()
        if isinstance(stumble, torch.Tensor):
            stumble = stumble.detach().cpu().numpy()

        self.metadata['termination'].append(termination)
        self.metadata['collision'].append(collision)
        self.metadata['stumble'].append(stumble)

        if timestep is not None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.detach().cpu().numpy()
            self.metadata['timestep'].append(timestep)

        if episode is not None:
            if isinstance(episode, torch.Tensor):
                episode = episode.detach().cpu().numpy()
            self.metadata['episode'].append(episode)

        if joint_angles is not None:
            if isinstance(joint_angles, torch.Tensor):
                joint_angles = joint_angles.detach().cpu().numpy()
            # Compute avg, min, max across joints for each env
            self.metadata['joint_avg'].append(np.mean(joint_angles, axis=1))
            self.metadata['joint_min'].append(np.min(joint_angles, axis=1))
            self.metadata['joint_max'].append(np.max(joint_angles, axis=1))

        if obstacle_phase is not None:
            if isinstance(obstacle_phase, torch.Tensor):
                obstacle_phase = obstacle_phase.detach().cpu().numpy()
            self.metadata['obstacle_phase'].append(obstacle_phase)

    def compute_umap(self, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        """
        Compute UMAP embedding of collected states

        Args:
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            metric: Distance metric for UMAP

        Returns:
            umap_results: UMAP embedding [total_samples, 2]
        """
        if len(self.states) == 0:
            raise ValueError("No states collected")

        # Concatenate all states
        all_states = np.concatenate(self.states, axis=0)
        print(f"Computing UMAP for {all_states.shape[0]} samples with {all_states.shape[1]} dimensions...")

        # Compute UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=2,
            random_state=42
        )
        umap_results = reducer.fit_transform(all_states)

        return umap_results

    def plot_umap_events(self, umap_results, save_path=None, figsize=(16, 5)):
        """
        Plot UMAP results colored by termination, collision, and stumble events

        Args:
            umap_results: Output from compute_umap()
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if len(self.metadata['termination']) == 0:
            raise ValueError("No event data collected")

        # Concatenate all event data
        termination = np.concatenate(self.metadata['termination']).astype(bool)
        collision = np.concatenate(self.metadata['collision']).astype(bool)
        stumble = np.concatenate(self.metadata['stumble']).astype(bool)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Termination
        ax = axes[0]
        normal_mask = ~termination
        if np.any(normal_mask):
            ax.scatter(umap_results[normal_mask, 0], umap_results[normal_mask, 1],
                      c='blue', alpha=0.3, s=10, label='Normal')
        if np.any(termination):
            ax.scatter(umap_results[termination, 0], umap_results[termination, 1],
                      c='red', alpha=0.7, s=15, label='Termination')
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax.set_title('Colored by Termination', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=13)
        num_term = np.sum(termination)
        ax.text(0.02, 0.98, f'Termination: {num_term}/{len(termination)} ({100*num_term/len(termination):.2f}%)',
               transform=ax.transAxes, fontsize=13, verticalalignment='top')

        # Plot 2: Collision
        ax = axes[1]
        normal_mask = ~collision
        if np.any(normal_mask):
            ax.scatter(umap_results[normal_mask, 0], umap_results[normal_mask, 1],
                      c='blue', alpha=0.3, s=10, label='Normal')
        if np.any(collision):
            ax.scatter(umap_results[collision, 0], umap_results[collision, 1],
                      c='orange', alpha=0.7, s=15, label='Collision')
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax.set_title('Colored by Collision', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=13)
        num_coll = np.sum(collision)
        ax.text(0.02, 0.98, f'Collision: {num_coll}/{len(collision)} ({100*num_coll/len(collision):.2f}%)',
               transform=ax.transAxes, fontsize=13, verticalalignment='top')

        # Plot 3: Stumble
        ax = axes[2]
        normal_mask = ~stumble
        if np.any(normal_mask):
            ax.scatter(umap_results[normal_mask, 0], umap_results[normal_mask, 1],
                      c='blue', alpha=0.3, s=10, label='Normal')
        if np.any(stumble):
            ax.scatter(umap_results[stumble, 0], umap_results[stumble, 1],
                      c='green', alpha=0.7, s=15, label='Stumble')
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax.set_title('Colored by Stumble', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=13)
        num_stum = np.sum(stumble)
        ax.text(0.02, 0.98, f'Stumble: {num_stum}/{len(stumble)} ({100*num_stum/len(stumble):.2f}%)',
               transform=ax.transAxes, fontsize=13, verticalalignment='top')

        plt.suptitle('UMAP of Compressed Deterministic State (actor-critic input)', fontsize=17, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_umap_combined(self, umap_results, save_path=None, figsize=(10, 8)):
        """
        Plot UMAP results with combined event coloring

        Args:
            umap_results: Output from compute_umap()
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if len(self.metadata['termination']) == 0:
            raise ValueError("No event data collected")

        # Concatenate all event data
        termination = np.concatenate(self.metadata['termination']).astype(bool)
        collision = np.concatenate(self.metadata['collision']).astype(bool)
        stumble = np.concatenate(self.metadata['stumble']).astype(bool)

        fig, ax = plt.subplots(figsize=figsize)

        # Assign colors based on event priority: termination > collision > stumble > normal
        colors = np.full(len(termination), 'blue', dtype=object)
        colors[stumble] = 'green'
        colors[collision] = 'orange'
        colors[termination] = 'red'

        # Create category labels for legend
        categories = np.full(len(termination), 0)  # 0 = normal
        categories[stumble] = 1
        categories[collision] = 2
        categories[termination] = 3

        # Plot each category
        category_info = [
            (0, 'blue', 'Normal', 0.3, 10),
            (1, 'green', 'Stumble', 0.6, 12),
            (2, 'orange', 'Collision', 0.7, 14),
            (3, 'red', 'Termination', 0.8, 16),
        ]

        for cat_id, color, label, alpha, size in category_info:
            mask = categories == cat_id
            if np.any(mask):
                ax.scatter(umap_results[mask, 0], umap_results[mask, 1],
                          c=color, alpha=alpha, s=size, label=label)

        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax.set_title('UMAP of Compressed Deterministic State\n(colored by event type)', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=13)

        # Add statistics
        stats_text = (
            f'Termination: {np.sum(termination)}/{len(termination)} ({100*np.sum(termination)/len(termination):.2f}%)\n'
            f'Collision: {np.sum(collision)}/{len(collision)} ({100*np.sum(collision)/len(collision):.2f}%)\n'
            f'Stumble: {np.sum(stumble)}/{len(stumble)} ({100*np.sum(stumble)/len(stumble):.2f}%)'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=13, verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_umap_joint_angles(self, umap_results, save_path=None, figsize=(16, 5)):
        """
        Plot UMAP results colored by joint angle statistics (avg, min, max)

        Args:
            umap_results: Output from compute_umap()
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if len(self.metadata['joint_avg']) == 0:
            raise ValueError("No joint angle data collected")

        # Concatenate all joint angle data
        joint_avg = np.concatenate(self.metadata['joint_avg'])
        joint_min = np.concatenate(self.metadata['joint_min'])
        joint_max = np.concatenate(self.metadata['joint_max'])

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Average joint angle deviation
        ax = axes[0]
        sc = ax.scatter(umap_results[:, 0], umap_results[:, 1],
                       c=joint_avg, cmap='coolwarm', alpha=0.5, s=10)
        plt.colorbar(sc, ax=ax, label='Avg Deviation (rad)')
        ax.set_xlabel('UMAP Dimension 1', fontsize=10)
        ax.set_ylabel('UMAP Dimension 2', fontsize=10)
        ax.set_title('Colored by Avg Joint Deviation', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 2: Min joint angle deviation
        ax = axes[1]
        sc = ax.scatter(umap_results[:, 0], umap_results[:, 1],
                       c=joint_min, cmap='coolwarm', alpha=0.5, s=10)
        plt.colorbar(sc, ax=ax, label='Min Deviation (rad)')
        ax.set_xlabel('UMAP Dimension 1', fontsize=10)
        ax.set_ylabel('UMAP Dimension 2', fontsize=10)
        ax.set_title('Colored by Min Joint Deviation', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 3: Max joint angle deviation
        ax = axes[2]
        sc = ax.scatter(umap_results[:, 0], umap_results[:, 1],
                       c=joint_max, cmap='coolwarm', alpha=0.5, s=10)
        plt.colorbar(sc, ax=ax, label='Max Deviation (rad)')
        ax.set_xlabel('UMAP Dimension 1', fontsize=10)
        ax.set_ylabel('UMAP Dimension 2', fontsize=10)
        ax.set_title('Colored by Max Joint Deviation', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.suptitle('UMAP of Compressed Deterministic State (joint deviation from init)', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_umap_sequence(self, umap_results, save_path=None, figsize=(10, 8)):
        """
        Plot UMAP results colored by timestep/sequence

        Args:
            umap_results: Output from compute_umap()
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if len(self.metadata['timestep']) == 0:
            raise ValueError("No timestep data collected")

        # Concatenate all timestep data
        timesteps = np.concatenate(self.metadata['timestep'])

        fig, ax = plt.subplots(figsize=figsize)

        sc = ax.scatter(umap_results[:, 0], umap_results[:, 1],
                       c=timesteps, cmap='viridis', alpha=0.5, s=10)
        plt.colorbar(sc, ax=ax, label='Timestep')
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax.set_title('UMAP of Compressed Deterministic State\n(colored by sequence/timestep)', fontsize=16)
        ax.grid(True, alpha=0.3)

        # Add stats
        stats_text = f'Timestep range: {int(timesteps.min())} - {int(timesteps.max())}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=13, verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_umap_obstacle_phase(self, umap_results, save_path=None, figsize=(10, 8)):
        """
        Plot UMAP results colored by obstacle phase (before/close/after)

        Args:
            umap_results: Output from compute_umap()
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        if len(self.metadata['obstacle_phase']) == 0:
            raise ValueError("No obstacle phase data collected")

        # Concatenate all obstacle phase data
        obstacle_phase = np.concatenate(self.metadata['obstacle_phase']).astype(int)

        fig, ax = plt.subplots(figsize=figsize)

        # Define colors and labels for each phase
        phase_info = [
            (0, 'blue', 'Before Obstacle'),
            (1, 'red', 'Close to Obstacle'),
            (2, 'green', 'After Obstacle'),
        ]

        for phase_id, color, label in phase_info:
            mask = obstacle_phase == phase_id
            if np.any(mask):
                ax.scatter(umap_results[mask, 0], umap_results[mask, 1],
                          c=color, alpha=0.5, s=10, label=label)

        # ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        # ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        # ax.set_title('UMAP of Compressed Deterministic State\n(colored by obstacle phase)', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=20)

        # Add statistics
        before_count = np.sum(obstacle_phase == 0)
        close_count = np.sum(obstacle_phase == 1)
        after_count = np.sum(obstacle_phase == 2)
        total = len(obstacle_phase)
        stats_text = (
            f'Before: {before_count}/{total} ({100*before_count/total:.1f}%)\n'
            f'Close: {close_count}/{total} ({100*close_count/total:.1f}%)\n'
            f'After: {after_count}/{total} ({100*after_count/total:.1f}%)'
        )
        # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        #        fontsize=13, verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def clear(self):
        """Clear all collected states and metadata"""
        self.states = []
        self.metadata = {
            'termination': [],
            'collision': [],
            'stumble': [],
            'timestep': [],
            'episode': [],
            'joint_avg': [],
            'joint_min': [],
            'joint_max': [],
            'obstacle_phase': [],
        }


def compute_termination(env):
    """
    Compute termination condition from environment

    Args:
        env: LeggedRobot environment instance

    Returns:
        termination: Boolean tensor [num_envs]
    """
    # Check illegal contact (head/base touchdown)
    termination = torch.any(
        torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1.,
        dim=1
    )

    # Fall detection
    fall = (env.root_states[:, 9] < -3.) | (env.projected_gravity[:, 2] > 0.)
    termination |= fall

    return termination


def compute_collision(env):
    """
    Compute collision condition from environment (contact on thighs/calves)

    Args:
        env: LeggedRobot environment instance

    Returns:
        collision: Boolean tensor [num_envs]
    """
    collision = torch.any(
        torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1,
        dim=1
    )
    return collision


def compute_stumble(env):
    """
    Compute stumble condition from environment (feet hitting vertical surfaces)

    Args:
        env: LeggedRobot environment instance

    Returns:
        stumble: Boolean tensor [num_envs]
    """
    stumble = torch.any(
        torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2)
        > 5 * torch.abs(env.contact_forces[:, env.feet_indices, 2]),
        dim=1
    )
    return stumble


def compute_obstacle_phase(measured_heights, obstacle_state, threshold=0.05):
    """
    Compute obstacle phase: before (0), close (1), or after (2) obstacle

    Uses height scan grid:
    - X (forward): -0.8 to 0.8 meters (17 points)
    - Y (lateral): -0.5 to 0.5 meters (11 points)

    Args:
        measured_heights: Terrain height measurements [num_envs, 187]
        obstacle_state: Current state tensor [num_envs] tracking phase
        threshold: Height threshold for obstacle detection (default: 0.05m)

    Returns:
        obstacle_phase: Phase tensor [num_envs] (0=before, 1=close, 2=after)
        obstacle_state: Updated state tensor [num_envs]
    """
    # Scan grid parameters
    x_points = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    y_points = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    num_y = len(y_points)

    # Define ranges for obstacle detection
    # Close range: around robot center (-0.1 to 0.1 in x)
    close_x_range = (-0.01, 0.01)
    # Ahead range: in front of robot (0.2 to 0.8 in x)
    ahead_x_range = (0.1, 0.8)
    # Behind range: behind robot (-0.8 to -0.2 in x)
    behind_x_range = (-0.8, -0.1)

    def get_indices(x_range):
        indices = []
        for x_idx, x in enumerate(x_points):
            if x_range[0] <= x <= x_range[1]:
                for y_idx in range(num_y):
                    indices.append(x_idx * num_y + y_idx)
        return indices

    close_indices = get_indices(close_x_range)
    ahead_indices = get_indices(ahead_x_range)
    behind_indices = get_indices(behind_x_range)

    # Check for obstacles in each region
    close_heights = measured_heights[:, close_indices]
    ahead_heights = measured_heights[:, ahead_indices]
    behind_heights = measured_heights[:, behind_indices]

    obstacle_close = torch.max(close_heights, dim=1)[0] > threshold
    obstacle_ahead = torch.max(ahead_heights, dim=1)[0] > threshold
    obstacle_behind = torch.max(behind_heights, dim=1)[0] > threshold

    # State machine logic:
    # State 0 (before): stay until close to obstacle
    # State 1 (close): stay while close, transition to after when obstacle behind only
    # State 2 (after): stay until reset

    # Transition from before to close
    transition_to_close = (obstacle_state == 0) & obstacle_close
    obstacle_state = torch.where(transition_to_close, torch.ones_like(obstacle_state), obstacle_state)

    # Transition from close to after (obstacle no longer close, but behind)
    transition_to_after = (obstacle_state == 1) & ~obstacle_close & obstacle_behind
    obstacle_state = torch.where(transition_to_after, torch.full_like(obstacle_state, 2), obstacle_state)

    return obstacle_state.clone(), obstacle_state


def visualize_termination_events_from_checkpoint(checkpoint_path, runner, num_steps=1000,
                                                  save_path=None, n_neighbors=15):
    """
    Collect compressed deterministic states from a trained model and visualize
    with termination/collision/stumble coloring.

    Uses the compressed deterministic state (output of wm_feature_encoder)
    that is actually given to the actor-critic.

    Args:
        checkpoint_path: Path to model checkpoint
        runner: WMPRunner instance
        num_steps: Number of steps to collect
        save_path: Path to save the plot
        n_neighbors: UMAP n_neighbors parameter
    """
    # Load checkpoint
    runner.load(checkpoint_path)
    runner.alg.actor_critic.eval()

    visualizer = TerminationEventsVisualizer()

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

    # Initialize obstacle state tracker (0=before, 1=close, 2=after)
    obstacle_state = torch.zeros(runner.env.num_envs, device=runner.env.device, dtype=torch.long)

    print(f"Collecting {num_steps} steps of recurrent states and event data...")

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

                # Compute events
                termination = compute_termination(runner.env)
                collision = compute_collision(runner.env)
                stumble = compute_stumble(runner.env)

                # Get joint angle difference from default/init state
                joint_angles = runner.env.dof_pos - runner.env.default_dof_pos

                # Compute obstacle phase
                obstacle_phase, obstacle_state = compute_obstacle_phase(
                    runner.env.measured_heights, obstacle_state, threshold=0.05
                )

                # Collect compressed deter state with event information
                visualizer.collect_state(
                    compressed_deter,
                    termination=termination,
                    collision=collision,
                    stumble=stumble,
                    timestep=step * torch.ones(runner.env.num_envs),
                    episode=torch.arange(runner.env.num_envs),
                    joint_angles=joint_angles,
                    obstacle_phase=obstacle_phase
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
                obstacle_state[reset_env_ids] = 0  # Reset obstacle phase tracking

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
    # Save individual event plots
    if save_path:
        base_path = save_path.rsplit('.', 1)[0]
        visualizer.plot_umap_events(umap_results, save_path=f"{base_path}_separate.pdf")
        visualizer.plot_umap_combined(umap_results, save_path=f"{base_path}_combined.pdf")
        visualizer.plot_umap_joint_angles(umap_results, save_path=f"{base_path}_joint_angles.pdf")
        visualizer.plot_umap_sequence(umap_results, save_path=f"{base_path}_sequence.pdf")
        visualizer.plot_umap_obstacle_phase(umap_results, save_path=f"{base_path}_obstacle_phase.pdf")
    else:
        visualizer.plot_umap_events(umap_results)
        visualizer.plot_umap_combined(umap_results)
        visualizer.plot_umap_joint_angles(umap_results)
        visualizer.plot_umap_sequence(umap_results)
        visualizer.plot_umap_obstacle_phase(umap_results)

    return visualizer, umap_results
