"""
Standalone script to generate UMAP visualization of world model recurrent states
colored by feet stumble events

Usage:
    UMAP_NUM_STEPS=1000 python run_tsne_feet_stumble.py --task=go2_blind --checkpoint=13000
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL: Import isaacgym FIRST, before ANY other imports
import isaacgym

# Import in the same order as play.py to avoid circular imports
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict

# All other imports that might use torch happen inside main()


def main():
    # Import numpy here (doesn't use torch)
    import numpy as np

    # Import torch AFTER isaacgym
    import torch

    # Import WMPRunner after torch
    from rsl_rl.runners import WMPRunner

    # Get environment args (uses standard legged_gym arguments)
    env_args = get_args()

    # Set visualization parameters
    num_steps = int(os.environ.get('UMAP_NUM_STEPS', '230'))
    save_path = os.environ.get('UMAP_SAVE_PATH', 'umap_near_obstacle.png')
    n_neighbors = int(os.environ.get('UMAP_N_NEIGHBORS', '20'))
    height_threshold = float(os.environ.get('UMAP_HEIGHT_THRESHOLD', '0.08'))

    print(f"\nVisualization settings:")
    print(f"  - num_steps: {num_steps}")
    print(f"  - state: compressed deterministic (wm_feature_encoder output)")
    print(f"  - color_by: near_obstacle (terrain scan height)")
    print(f"  - height_threshold: {height_threshold}m")
    print(f"  - save_path: {save_path}")
    print(f"  - n_neighbors: {n_neighbors}")
    print(f"\nTip: Set environment variables to customize (e.g., UMAP_HEIGHT_THRESHOLD=0.08)")

    # Get environment and training configs
    env_cfg, train_cfg = task_registry.get_cfgs(name=env_args.task)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_length = 2
    env_cfg.terrain.terrain_width = 2
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.difficulty = 0.1  # adjust difficulty (0.0-1.0)
    # terrain_proportions: [smooth, rough, stairs, stepping_stones, gaps, rough_stairs, obstacles]
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    env_cfg.noise.add_noise = False
    env_cfg.depth.update_interval = 2  # Can differ from training config
    # Keep domain randomizations ENABLED but with fixed values to match training observation structure
    env_cfg.domain_rand.friction_range = [0.8, 0.8]  # Fixed value
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]  # Fixed value
    env_cfg.domain_rand.added_mass_range = [0., 0.]  # Fixed value (no added mass)
    env_cfg.domain_rand.com_x_pos_range = [0.0, 0.0]  # Fixed value (no offset)
    env_cfg.domain_rand.com_y_pos_range = [0.0, 0.0]  # Fixed value (no offset)
    env_cfg.domain_rand.com_z_pos_range = [0.0, 0.0]  # Fixed value (no offset)

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True  # Keep enabled with fixed values
    env_cfg.domain_rand.randomize_friction = True  # MUST be True to include in observations
    env_cfg.domain_rand.randomize_restitution = True  # MUST be True to include in observations
    env_cfg.domain_rand.randomize_base_mass = True  # MUST be True to include in observations
    env_cfg.domain_rand.randomize_com_pos = True  # MUST be True to include in observations
    env_cfg.domain_rand.randomize_link_mass = False  # Doesn't affect observations
    env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]

    env_cfg.commands.ranges.lin_vel_x = [0.8, 0.8]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env_cfg.init_state.randomize_position = False

    # Create environment
    env, _ = task_registry.make_env(name=env_args.task, args=env_args, env_cfg=env_cfg)

    # Automatically find checkpoint if not provided (like play.py does)
    from legged_gym.utils.helpers import get_load_path
    from legged_gym import LEGGED_GYM_ROOT_DIR

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)

    # Use load_run from args if provided, otherwise use -1 (latest)
    load_run = env_args.load_run if env_args.load_run else -1

    if env_args.checkpoint is None:
        print("No checkpoint provided, searching for latest checkpoint...")
        checkpoint_path = get_load_path(log_root, load_run=load_run, checkpoint=-1)
        print(f"Found latest checkpoint: {checkpoint_path}")
    else:
        # env_args.checkpoint is an integer, convert to path using get_load_path
        checkpoint_path = get_load_path(log_root, load_run=load_run, checkpoint=env_args.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)

    # Create runner (convert config to dict as WMPRunner expects)
    log_dir = os.path.dirname(checkpoint_path)
    train_cfg_dict = class_to_dict(train_cfg)

    # Disable wandb logging for visualization
    train_cfg_dict['runner']['use_wandb'] = False

    runner = WMPRunner(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=log_dir,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nLoading checkpoint from: {checkpoint_path}")
    print(f"Collecting {num_steps} steps of compressed deterministic states...")
    print(f"Coloring by obstacle proximity (terrain scan height > {height_threshold}m)")

    # Import visualization module here (after isaacgym/torch imports are done)
    from visualize_feet_stumble import visualize_feet_stumble_from_checkpoint

    # Run visualization
    visualizer, umap_results = visualize_feet_stumble_from_checkpoint(
        checkpoint_path=checkpoint_path,
        runner=runner,
        num_steps=num_steps,
        save_path=save_path,
        n_neighbors=n_neighbors,
        height_threshold=height_threshold
    )

    print(f"\nVisualization saved to: {save_path}")
    print(f"UMAP results shape: {umap_results.shape}")

    # Print some statistics
    print(f"\nCollected {len(visualizer.states)} batches of states")
    total_samples = sum(s.shape[0] for s in visualizer.states)
    print(f"Total samples: {total_samples}")

    # Print near-obstacle statistics
    if len(visualizer.metadata['feet_stumble']) > 0:
        near_obstacle = np.concatenate(visualizer.metadata['feet_stumble'])
        num_near_obstacles = np.sum(near_obstacle)
        print(f"Near obstacle events (height > {height_threshold}m): {num_near_obstacles} / {len(near_obstacle)} ({100*num_near_obstacles/len(near_obstacle):.2f}%)")


if __name__ == "__main__":
    main()
