from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.g1.g1_base import G1BaseCfg


class G1BaseTaskCfg(G1BaseCfg):
    class asset(G1BaseCfg.asset):
        visualize_ghost = False

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None

        prop_dim = 33
        action_dim = 12
        privileged_dim = 24 + 20 + 3
        height_dim = 187
        footprint_dim = 33 * 21
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim

        forward_height_dim = 0
        privileged_obs = True
        asymmetric_actor = True

    class depth:
        use_camera = False
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.33, 0.0, 0.08]
        y_angle = [-5, 5]
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 2

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class commands(G1BaseCfg.commands):
        use_stop_and_go = False


class G1BaseCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'WMPRunner'

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [512, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = 'elu'
        use_prop_in_actor = True

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'g1_base'
        algorithm_class_name = 'PPOWMP'
        policy_class_name = 'ActorCritic'
        use_wandb = True
        wandb_project = 'yunhan_zheng_WMP'
        wandb_entity = None  # None = use your logged-in wandb account

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100
