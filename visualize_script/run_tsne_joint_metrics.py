"""
Standalone script to generate t-SNE visualizations colored by joint metrics
(position, velocity, and acceleration) from a SINGLE rollout

Usage:
    TSNE_NUM_STEPS=1000 TSNE_PERPLEXITY=20 python run_tsne_joint_metrics.py --task=go2_blind --checkpoint=13000
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
    num_steps = int(os.environ.get('TSNE_NUM_STEPS', '1000'))
    use_full_state = os.environ.get('TSNE_USE_FULL_STATE', 'false').lower() == 'true'
    perplexity = int(os.environ.get('TSNE_PERPLEXITY', '30'))

    print(f"\nVisualization settings:")
    print(f"  - num_steps: {num_steps}")
    print(f"  - use_full_state: {use_full_state}")
    print(f"  - perplexity: {perplexity}")
    print(f"  - output: tsne_joint_metrics.png (combined)")
    print(f"            tsne_joint_position.png")
    print(f"            tsne_joint_velocity.png")
    print(f"            tsne_joint_acceleration.png")
    print(f"\nTip: Set environment variables to customize (e.g., TSNE_NUM_STEPS=500)")

    # Get environment and training configs
    env_cfg, train_cfg = task_registry.get_cfgs(name=env_args.task)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_length = 2
    env_cfg.terrain.terrain_width = 2
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.difficulty = 0.1  # adjust difficulty (0.0-1.0)
    # terrain_proportions: [smooth, rough, stairs, stepping_stones, gaps, rough_stairs, obstacles]
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # 100% obstacles

    env_cfg.noise.add_noise = False
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

    # Create environment
    env, _ = task_registry.make_env(name=env_args.task, args=env_args, env_cfg=env_cfg)

    # Automatically find checkpoint if not provided (like play.py does)
    from legged_gym.utils.helpers import get_load_path
    from legged_gym import LEGGED_GYM_ROOT_DIR

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)

    if env_args.checkpoint is None:
        print("No checkpoint provided, searching for latest checkpoint...")
        checkpoint_path = get_load_path(log_root, load_run=-1, checkpoint=-1)
        print(f"Found latest checkpoint: {checkpoint_path}")
    else:
        # env_args.checkpoint is an integer, convert to path using get_load_path
        checkpoint_path = get_load_path(log_root, load_run=-1, checkpoint=env_args.checkpoint)
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
    print(f"Collecting {num_steps} steps in a SINGLE rollout...")
    print(f"Using {'full state (deter + stoch)' if use_full_state else 'deterministic state only'}")
    print(f"Will generate visualizations colored by:")
    print(f"  1. Average joint position")
    print(f"  2. Average joint velocity")
    print(f"  3. Average joint acceleration")

    # Import visualization module here (after isaacgym/torch imports are done)
    from visualize_joint_metrics import visualize_joint_metrics_from_checkpoint

    # Run visualization
    visualizer, tsne_results = visualize_joint_metrics_from_checkpoint(
        checkpoint_path=checkpoint_path,
        runner=runner,
        num_steps=num_steps,
        use_deter_only=not use_full_state,
        perplexity=perplexity
    )

    print(f"\nt-SNE results shape: {tsne_results.shape}")

    # Print some statistics
    print(f"\nCollected {len(visualizer.states)} batches of states")
    total_samples = sum(s.shape[0] for s in visualizer.states)
    print(f"Total samples: {total_samples}")

    # Print joint metrics statistics
    if len(visualizer.metadata['joint_positions']) > 0:
        joint_positions = np.concatenate(visualizer.metadata['joint_positions'])
        joint_velocities = np.concatenate(visualizer.metadata['joint_velocities'])
        joint_accelerations = np.concatenate(visualizer.metadata['joint_accelerations'])

        # Compute averages
        avg_pos = np.mean(np.abs(joint_positions), axis=1)
        avg_vel = np.mean(np.abs(joint_velocities), axis=1)
        avg_acc = np.mean(np.abs(joint_accelerations), axis=1)

        print(f"\nJoint Position statistics:")
        print(f"  Mean: {np.mean(avg_pos):.4f} rad, Std: {np.std(avg_pos):.4f}, Range: [{np.min(avg_pos):.4f}, {np.max(avg_pos):.4f}]")

        print(f"\nJoint Velocity statistics:")
        print(f"  Mean: {np.mean(avg_vel):.4f} rad/s, Std: {np.std(avg_vel):.4f}, Range: [{np.min(avg_vel):.4f}, {np.max(avg_vel):.4f}]")

        print(f"\nJoint Acceleration statistics:")
        print(f"  Mean: {np.mean(avg_acc):.4f} rad/s², Std: {np.std(avg_acc):.4f}, Range: [{np.min(avg_acc):.4f}, {np.max(avg_acc):.4f}]")


if __name__ == "__main__":
    main()
