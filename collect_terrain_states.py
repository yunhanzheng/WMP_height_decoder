"""
Collect recurrent states from a single terrain configuration
Saves to a numpy file for later comparison
"""

import argparse
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL: Import isaacgym FIRST
import isaacgym

# Import in the same order as play.py
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict


def main():
    import torch
    from rsl_rl.runners import WMPRunner
    from visualize_recurrent_state import RecurrentStateVisualizer

    # Get environment args (uses standard legged_gym arguments)
    env_args = get_args()

    # Get custom parameters from environment variables
    terrain_type = os.environ.get('TERRAIN_TYPE', 'domino')
    output_file = os.environ.get('OUTPUT_FILE', '/tmp/terrain_states.npz')
    num_steps = int(os.environ.get('NUM_STEPS', '200'))

    if terrain_type not in ['domino', 'stripes']:
        print(f"ERROR: Invalid TERRAIN_TYPE: {terrain_type}")
        print("Must be 'domino' or 'stripes'")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Collecting states from {terrain_type.upper()} terrain")
    print(f"{'='*80}\n")

    # Get configs
    env_cfg, train_cfg = task_registry.get_cfgs(name=env_args.task)

    # Set terrain based on type
    if terrain_type == 'domino':
        env_cfg.terrain.difficulty = 1.0
        env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        print("Terrain: Domino - index 5")
    else:  # stripes
        env_cfg.terrain.difficulty = 0.1
        env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        print("Terrain: Stripes - index 6")

    print(f"Terrain proportions: {env_cfg.terrain.terrain_proportions}\n")

    # Create environment
    env, _ = task_registry.make_env(name=env_args.task, args=env_args, env_cfg=env_cfg)

    # Find checkpoint
    if env_args.checkpoint is None:
        from legged_gym.utils.helpers import get_load_path
        from legged_gym import LEGGED_GYM_ROOT_DIR

        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        checkpoint_path = get_load_path(log_root, load_run=-1, checkpoint=-1)
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = env_args.checkpoint

    # Create runner
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

    # Collect states
    print(f"\nCollecting {num_steps} steps...")
    visualizer = RecurrentStateVisualizer(use_deter_only=True)
    collect_states(runner, visualizer, num_steps)

    states = np.concatenate(visualizer.states, axis=0)
    print(f"Collected {states.shape[0]} samples\n")

    # Save to file
    np.savez(output_file,
             states=states,
             terrain_type=terrain_type)
    print(f"Saved to: {output_file}")
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

            if step % 50 == 0 and step > 0:
                print(f"  Step {step}/{num_steps}")


if __name__ == "__main__":
    main()
