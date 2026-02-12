from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go2.go2_base import GO2BaseCfg

class GO2BlindCfg(GO2BaseCfg):
    class asset(GO2BaseCfg.asset):
        visualize_ghost = False  # Show ghost robot from decoder output

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        prop_dim = 33 # proprioception
        action_dim = 12
        privileged_dim = 24 + 26 + 3  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        height_dim = 341  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim
        forward_height_dim = 0 # for depth image prediction
        privileged_obs = True  # Disable extra privileged observations (domain randomization params)
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

        update_interval = 2  # 5 works without retraining, 8 worse

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

class GO2BlindCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'WMPRunner'

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # vel_predict_coef = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_blind'
        algorithm_class_name = 'PPOWMP'
        policy_class_name = 'ActorCritic'
        max_iterations = 10000  # number of policy updates
        save_interval = 1000
        use_wandb = True

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100
