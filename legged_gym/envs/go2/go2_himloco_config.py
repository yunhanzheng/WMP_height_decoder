from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go2.go2_base import GO2BaseCfg


class GO2HIMLocoCfg(GO2BaseCfg):
    """
    GO2 HIM Locomotion Configuration

    Architecture:
    - HIM Estimator: Takes proprioceptive observations and their history,
                     outputs latent representation (16 dims) and velocity estimation (3 dims)
    - Actor: Takes current proprioceptive obs + latent + velocity estimation from HIM estimator
    - Critic: Takes full observations with privileged information (asymmetric training)
    """
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = 6  # History steps for HIM estimator

        # Proprioceptive observation dimensions
        prop_dim = 33  # ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12)
        action_dim = 12

        # Privileged information dimensions
        privileged_dim = 24 + 26 + 3  # friction(24) + domain_rand(26) + base_lin_vel(3)
        height_dim = 187  # heightmap

        # Single step observation for HIM estimator (proprioceptive only, no velocity)
        num_one_step_observations = 45  # ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12)

        # Actor observations: single step proprioceptive (history handled by buffer)
        # Total actor input after history buffer: 45 * 6 = 270
        num_observations = num_one_step_observations  # 45

        # Critic observations: full privileged observations
        # prop_dim(33) + privileged_dim(53) + height_dim(187) + action_dim(12) = 285
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim  # 285

        privileged_obs = True
        asymmetric_actor = True

    class depth:
        use_camera = False
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.33, 0.0, 0.08]  # front camera
        y_angle = [-5, 5]  # positive pitch down
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 5

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True


class GO2HIMLocoCfgPPO(LeggedRobotCfgPPO):
    """
    PPO Training Configuration for GO2 HIM Locomotion

    Uses:
    - HIMOnPolicyRunner: Handles history buffer and asymmetric observations
    - HIMPPO: PPO with HIM estimator loss (velocity estimation + contrastive learning)
    - HIMActorCritic: Actor-critic with integrated HIM estimator
    """
    runner_class_name = 'HIMOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        # Actor network: processes (current_obs + vel + latent) = 45 + 3 + 16 = 64 dims
        actor_hidden_dims = [256, 128, 64]
        # Critic network: processes full privileged observations (285 dims)
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # PPO hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_himloco'
        algorithm_class_name = 'HIMPPO'
        policy_class_name = 'HIMActorCritic'
        max_iterations = 20000
        num_steps_per_env = 24
        save_interval = 1000
        use_wandb = True
        resume = False
