import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os

class RecurrentStateVisualizer:
    def __init__(self, use_deter_only=True):
        """
        Initialize the visualizer

        Args:
            use_deter_only: If True, only use deterministic state for t-SNE.
                          If False, use both deterministic and stochastic states.
        """
        self.use_deter_only = use_deter_only
        self.states = []
        self.metadata = {
            'rewards': [],
            'timesteps': [],
            'episodes': [],
            'actions': []
        }

    def collect_state(self, wm_latent, reward=None, timestep=None, episode=None, action=None):
        """
        Collect a recurrent state during rollout

        Args:
            wm_latent: Dictionary containing 'deter' and 'stoch' keys
            reward: Optional reward at this timestep
            timestep: Optional timestep index
            episode: Optional episode index
            action: Optional action taken
        """
        if self.use_deter_only:
            # Only use deterministic state
            state = wm_latent['deter'].detach().cpu().numpy()
        else:
            # Use both deterministic and stochastic states
            deter = wm_latent['deter'].detach().cpu().numpy()
            stoch = wm_latent['stoch'].detach().cpu().numpy()

            # Flatten stochastic state if it's discrete (has shape [batch, stoch, discrete])
            if len(stoch.shape) == 3:
                stoch = stoch.reshape(stoch.shape[0], -1)

            state = np.concatenate([deter, stoch], axis=-1)

        self.states.append(state)

        if reward is not None:
            if isinstance(reward, torch.Tensor):
                reward = reward.detach().cpu().numpy()
            self.metadata['rewards'].append(reward)

        if timestep is not None:
            self.metadata['timesteps'].append(timestep)

        if episode is not None:
            self.metadata['episodes'].append(episode)

        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            self.metadata['actions'].append(action)

    def compute_tsne(self, n_components=2, perplexity=30, n_iter=1000, random_state=42):
        """
        Compute t-SNE on collected states

        Args:
            n_components: Number of dimensions for t-SNE (typically 2 or 3)
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations for optimization
            random_state: Random seed

        Returns:
            tsne_results: Array of shape [n_samples, n_components]
        """
        if len(self.states) == 0:
            raise ValueError("No states collected. Call collect_state() first.")

        # Concatenate all states
        all_states = np.concatenate(self.states, axis=0)
        print(f"Computing t-SNE on {all_states.shape[0]} samples with dimension {all_states.shape[1]}")

        # Adjust perplexity if necessary (must be less than n_samples)
        n_samples = all_states.shape[0]
        max_perplexity = (n_samples - 1) // 3  # Conservative upper bound
        if perplexity >= n_samples:
            adjusted_perplexity = min(30, max_perplexity)
            print(f"WARNING: Perplexity {perplexity} is too high for {n_samples} samples.")
            print(f"Adjusting perplexity to {adjusted_perplexity}")
            perplexity = adjusted_perplexity

        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity,
                   n_iter=n_iter, random_state=random_state, verbose=1)
        tsne_results = tsne.fit_transform(all_states)

        return tsne_results

    def plot_tsne(self, tsne_results, color_by='timestep', save_path=None, figsize=(10, 8)):
        """
        Plot t-SNE results

        Args:
            tsne_results: Output from compute_tsne()
            color_by: What to color points by ('timestep', 'reward', 'episode', or None)
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Determine colors
        if color_by == 'timestep' and len(self.metadata['timesteps']) > 0:
            timesteps = np.concatenate(self.metadata['timesteps'])
            scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1],
                               c=timesteps, cmap='viridis', alpha=0.6, s=10)
            plt.colorbar(scatter, ax=ax, label='Timestep')
            title = 't-SNE of Recurrent States (colored by timestep)'

        elif color_by == 'reward' and len(self.metadata['rewards']) > 0:
            rewards = np.concatenate(self.metadata['rewards'])
            if len(rewards.shape) > 1:
                rewards = rewards.flatten()
            scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1],
                               c=rewards, cmap='RdYlGn', alpha=0.6, s=10)
            plt.colorbar(scatter, ax=ax, label='Reward')
            title = 't-SNE of Recurrent States (colored by reward)'

        elif color_by == 'episode' and len(self.metadata['episodes']) > 0:
            episodes = np.concatenate(self.metadata['episodes'])
            scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1],
                               c=episodes, cmap='tab20', alpha=0.6, s=10)
            plt.colorbar(scatter, ax=ax, label='Episode')
            title = 't-SNE of Recurrent States (colored by episode)'

        else:
            ax.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.6, s=10)
            title = 't-SNE of Recurrent States'

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def clear(self):
        """Clear all collected states and metadata"""
        self.states = []
        self.metadata = {
            'rewards': [],
            'timesteps': [],
            'episodes': [],
            'actions': []
        }


def visualize_from_checkpoint(checkpoint_path, runner, num_steps=1000,
                              use_deter_only=True, color_by='timestep',
                              save_path=None):
    """
    Collect states from a trained model and visualize

    Args:
        checkpoint_path: Path to model checkpoint
        runner: WMPRunner instance
        num_steps: Number of steps to collect
        use_deter_only: Whether to use only deterministic state
        color_by: What to color points by
        save_path: Path to save the plot
    """
    # Load checkpoint
    runner.load(checkpoint_path)
    runner.alg.actor_critic.eval()

    visualizer = RecurrentStateVisualizer(use_deter_only=use_deter_only)

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

    print(f"Collecting {num_steps} steps of recurrent states...")

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
                visualizer.collect_state(
                    wm_latent,
                    timestep=step * torch.ones(runner.env.num_envs),
                    episode=torch.arange(runner.env.num_envs)
                )

            # Take action
            history = trajectory_history.flatten(1).to(runner.device)
            actions = runner.alg.act(obs, critic_obs, history, wm_feature.to(runner.env.device))
            obs, privileged_obs, rewards, dones, infos, reset_env_ids = runner.env.step(actions)

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

            if step % 100 == 0:
                print(f"Step {step}/{num_steps}")

    print("Computing t-SNE...")
    tsne_results = visualizer.compute_tsne()

    print("Plotting...")
    visualizer.plot_tsne(tsne_results, color_by=color_by, save_path=save_path)

    return visualizer, tsne_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize recurrent states with t-SNE')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of steps to collect')
    parser.add_argument('--use_full_state', action='store_true',
                       help='Use both deterministic and stochastic states (default: deter only)')
    parser.add_argument('--color_by', type=str, default='timestep',
                       choices=['timestep', 'reward', 'episode', 'none'],
                       help='What to color points by')
    parser.add_argument('--save_path', type=str, default='tsne_recurrent_state.png',
                       help='Path to save the plot')

    args = parser.parse_args()

    # Note: You need to initialize your runner here
    # This is just a template - you'll need to adapt it to your setup
    print("Note: You need to initialize your WMPRunner instance")
    print("Example usage in your training script:")
    print("""
    from visualize_recurrent_state import visualize_from_checkpoint

    visualizer, tsne_results = visualize_from_checkpoint(
        checkpoint_path='path/to/model.pt',
        runner=your_runner_instance,
        num_steps=1000,
        use_deter_only=True,
        color_by='timestep',
        save_path='tsne_visualization.png'
    )
    """)
