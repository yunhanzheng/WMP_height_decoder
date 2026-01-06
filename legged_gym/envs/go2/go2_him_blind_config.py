from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go2.go2_base import GO2BaseCfg

class GO2HIMBlindCfg(GO2BaseCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        prop_dim = 33 # proprioception
        action_dim = 12
        privileged_dim = 24 + 26 + 3  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        height_dim = 187  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim
        privileged_obs = True  # Disable extra privileged observations (domain randomization params)

        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187  # same as observations, no base_lin_vel, disturbance, or heights
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise

    class depth:
        use_camera = False
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.33, 0.0, 0.08]  # front camera
        y_angle = [-5, 5]  # positive pitch down
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

class GO2HIMBlindCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'HIMOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_him_blind'
        algorithm_class_name = 'HIMPPO'
        policy_class_name = 'HIMActorCritic'
        max_iterations = 20000  # number of policy updates
        save_interval = 1000
        use_wandb = True
        resume = False  # Changed to False to start training from scratch
        # resume_path = "logs/go2_baseline_blind/Dec21_14-38-22_/model_6000.pt"