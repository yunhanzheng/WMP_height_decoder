"""
Example code showing how to integrate t-SNE visualization into the WMPRunner training loop

You can add this to your wmp_runner.py or use it as a reference.
"""

from visualize_recurrent_state import RecurrentStateVisualizer
import torch


def add_tsne_visualization_to_runner(runner_instance):
    """
    Add t-SNE visualization capability to WMPRunner

    Call this function to add visualization methods to your runner instance.

    Example usage:
        runner = WMPRunner(...)
        add_tsne_visualization_to_runner(runner)

        # During training
        if iteration % 100 == 0:
            runner.visualize_recurrent_states(
                num_steps=500,
                save_path=f'tsne_iter_{iteration}.png'
            )
    """

    def visualize_recurrent_states(self, num_steps=500, use_deter_only=True,
                                   color_by='timestep', save_path=None):
        """
        Collect and visualize recurrent states during training

        Args:
            num_steps: Number of steps to collect
            use_deter_only: Whether to use only deterministic state
            color_by: What to color points by ('timestep', 'reward', 'episode')
            save_path: Path to save the plot
        """
        print(f"Collecting {num_steps} steps for t-SNE visualization...")

        visualizer = RecurrentStateVisualizer(use_deter_only=use_deter_only)

        # Get current state
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        # Initialize trajectory history
        trajectory_history = torch.zeros(
            size=(self.env.num_envs, self.history_length,
                  self.env.num_obs - self.env.privileged_dim - self.env.height_dim - 3),
            device=self.device
        )

        # Initialize world model
        wm_latent = wm_action = None
        wm_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)
        wm_obs = {
            "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),
            "is_first": wm_is_first,
        }

        if self.env.cfg.depth.use_camera:
            wm_obs["image"] = torch.zeros(
                ((self.env.num_envs,) + self.env.cfg.depth.resized + (1,)),
                device=self._world_model.device
            )

        wm_action_history = torch.zeros(
            size=(self.env.num_envs, self.wm_update_interval, self.env.num_actions),
            device=self._world_model.device
        )
        wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)

        # Collect states
        with torch.no_grad():
            for step in range(num_steps):
                if step % self.wm_update_interval == 0:
                    # World model obs step
                    wm_embed = self._world_model.encoder(wm_obs)
                    wm_latent, _ = self._world_model.dynamics.obs_step(
                        wm_latent, wm_action, wm_embed, wm_obs["is_first"]
                    )
                    wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)
                    wm_is_first[:] = 0

                    # Collect state with metadata
                    visualizer.collect_state(
                        wm_latent,
                        reward=wm_reward.clone(),
                        timestep=step * torch.ones(self.env.num_envs),
                        episode=torch.arange(self.env.num_envs)
                    )

                # Take action
                history = trajectory_history.flatten(1).to(self.device)
                actions = self.alg.act(obs, critic_obs, history, wm_feature.to(self.env.device))
                obs, privileged_obs, rewards, dones, infos, reset_env_ids = self.env.step(actions)

                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(
                    self.device), rewards.to(self.device), dones.to(self.device)

                # Update world model input
                wm_action_history = torch.concat(
                    (wm_action_history[:, 1:], actions.unsqueeze(1).to(self._world_model.device)),
                    dim=1
                )
                wm_action = wm_action_history.flatten(1)
                wm_reward += rewards.to(self._world_model.device)

                wm_obs = {
                    "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),
                    "is_first": wm_is_first,
                }

                # Handle resets
                reset_env_ids = reset_env_ids.cpu().numpy()
                if len(reset_env_ids) > 0:
                    wm_action_history[reset_env_ids, :] = 0
                    wm_is_first[reset_env_ids] = 1
                    wm_reward[reset_env_ids] = 0

        # Compute and plot t-SNE
        print("Computing t-SNE...")
        tsne_results = visualizer.compute_tsne()

        print("Plotting...")
        visualizer.plot_tsne(tsne_results, color_by=color_by, save_path=save_path)

        return visualizer, tsne_results

    # Add method to runner instance
    import types
    runner_instance.visualize_recurrent_states = types.MethodType(
        visualize_recurrent_states, runner_instance
    )

    print("Added visualize_recurrent_states() method to runner")


# Example of how to modify the learn() function in WMPRunner
def example_modified_learn_loop():
    """
    Example showing where to add visualization in the training loop
    """
    example_code = '''
    # In WMPRunner.learn() method, add this after the main training loop:

    for it in range(self.current_learning_iteration, tot_iter):
        # ... existing training code ...

        # Add visualization every N iterations
        if it % 100 == 0 and it > 0:  # Visualize every 100 iterations
            save_path = os.path.join(self.log_dir, f'tsne_iter_{it}.png')
            self.visualize_recurrent_states(
                num_steps=500,
                use_deter_only=True,
                color_by='reward',  # or 'timestep', 'episode'
                save_path=save_path
            )

        # ... rest of training code ...
    '''
    print(example_code)


if __name__ == "__main__":
    print("This file shows how to integrate t-SNE visualization into WMPRunner")
    print("\n" + "="*80)
    print("Step 1: Import the function")
    print("="*80)
    print("from tsne_integration_example import add_tsne_visualization_to_runner")

    print("\n" + "="*80)
    print("Step 2: Add to your runner after initialization")
    print("="*80)
    print("runner = WMPRunner(env, train_cfg, log_dir, device)")
    print("add_tsne_visualization_to_runner(runner)")

    print("\n" + "="*80)
    print("Step 3: Call during training")
    print("="*80)
    example_modified_learn_loop()

    print("\n" + "="*80)
    print("Or use the standalone script:")
    print("="*80)
    print("python run_tsne_visualization.py --task=a1 --checkpoint=path/to/model.pt")
