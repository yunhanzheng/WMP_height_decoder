from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go2.go2_base import GO2BaseCfg


class GO2BaselineCfg(GO2BaseCfg):
    """
    Configuration for GO2 robot baseline with symmetric actor-critic.
    Both actor and critic receive full observations including privileged information.
    """

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None

        # Observation dimensions
        prop_dim = 33  # proprioception: ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12)
        action_dim = 12
        privileged_dim = 24 + 26 + 3  # friction, mass, com + DR params + base linear vel
        height_dim = 187  # heightmap

        # Full observation for both actor and critic
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim

        forward_height_dim = 0
        privileged_obs = True
        asymmetric_actor = False  # Symmetric: both actor and critic get full observations

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


class GO2BaselineCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_baseline'
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 15000
        save_interval = 1000
        use_wandb = True
