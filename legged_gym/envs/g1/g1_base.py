from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

training_stage = 4
# 1 = flat terrain + full cmd_vel
# 2 = sparse stripes (3-5, 1.1-2.5 m) + forward-only, basic crossing rewards
# 3 = sparser stripes (2-3, 2.0-4.0 m) + clean step-over (all latest fixes)
# 4 = stripe terrain + mid-crossing pause: hold with one foot over, one not; 2nd foot clears cleanly


def _footprint_range(n_points, step=0.05):
    """Centered grid of n_points with given step size."""
    start = -step * (n_points // 2)
    return [round(start + step * i, 4) for i in range(n_points)]


class G1BaseCfg(LeggedRobotCfg):
    """Base configuration shared by Unitree G1 (12-DoF biped) variants.

    Mirrors the structure of GO2BaseCfg so it plugs into the same WMP world-model
    pipeline (proprioception + privileged + heightmap + footprint decoder), but
    with biped morphology, gains, default pose and termination contacts.
    """

    class terrain(LeggedRobotCfg.terrain):

        if training_stage == 1:
            # flat ground for basic locomotion (smooth slope at difficulty 0)
            terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            curriculum = False
            difficulty = 0.0  # required when curriculum=False (randomized_terrain path)
        elif training_stage == 2:
            # [8]: sparse stripes — learn basic forward crossing
            terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            curriculum = True
            stripe_num_rects_min = 3
            stripe_num_rects_max = 5
            stripe_min_gap = 1.1
            stripe_max_gap = 2.5
        elif training_stage == 3:
            # [8]: sparser stripes — clean step-over without stepping on top
            terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            curriculum = True
            stripe_num_rects_min = 2
            stripe_num_rects_max = 3
            stripe_min_gap = 2.0
            stripe_max_gap = 4.0
        elif training_stage == 4:
            # [8]: same sparser stripes as stage 3 — refine mid-cross pause + 2nd-foot clearance
            terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            curriculum = True
            stripe_num_rects_min = 2
            stripe_num_rects_max = 3
            stripe_min_gap = 2.0
            stripe_max_gap = 4.0

        mesh_type = "trimesh"
        max_init_terrain_level = 0  # start on flat ground (level 0); curriculum raises difficulty as robot succeeds
        border_size = 25
        terrain_length = 7.5
        terrain_width = 7.5
        num_rows = 10  # difficulty levels
        num_cols = 10  # terrain types
        measure_heights = True
        # 17 x 11 = 187 points -> height_dim = 187 (same grid as GO2)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        map_path = None
        num_points = len(measured_points_x) * len(measured_points_y)
        measured_forward_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        measured_forward_points_y = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
                                     -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                     1.0, 1.1, 1.2]
        # footprint grid for the binary height decoder (base-centered, 5 cm spacing)
        footprint_length_points = 33   # along x
        footprint_width_points = 21    # along y
        footprint_points_x = _footprint_range(footprint_length_points)
        footprint_points_y = _footprint_range(footprint_width_points)

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint': -0.1,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.1,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
        }

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 24
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # Stiffness/damping keyed by substring match against joint names.
        stiffness = {
            'hip_yaw': 100,
            'hip_roll': 100,
            'hip_pitch': 100,
            'knee': 150,
            'ankle': 40,
        }  # [N*m/rad]
        damping = {
            'hip_yaw': 2,
            'hip_roll': 2,
            'hip_pitch': 2,
            'knee': 4,
            'ankle': 2,
        }  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.0]

        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]  # kg
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_x_pos_range = [-0.05, 0.05]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.05, 0.05]

        push_robots = True
        push_interval_s = 5
        min_push_interval_s = 5
        max_push_vel_xy = 1.0

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = True
        latency_range = [0.00, 0.005]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            # privileged
            height_measurements = 5.0
            contact_force = 0.005
            com_pos = 20
            pd_gains = 5

        clip_observations = 100.
        clip_actions = 100.0
        base_height = 0.78  # used to normalize measured heights

    class rewards(LeggedRobotCfg.rewards):
        reward_curriculum = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        max_contact_force = 200.0
        tracking_sigma = 0.25
        only_positive_rewards = False

        class scales:
            # task tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # base regularization
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -2.0
            # effort / smoothness
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            action_rate = -0.01
            dof_pos_limits = -5.0
            # biped gait shaping (implemented in G1Robot)
            alive = 0.15
            contact = 0.18
            contact_no_vel = -0.2
            hip_pos = -1.0

            if training_stage == 1:
                feet_swing_height = -2.0
                yaw_alignment = -1.0
            elif training_stage == 2:
                feet_swing_height = -2.0
                feet_air_time = 0.05
                feet_stumble = -0.3
                collision = -1.0
                feet_step = -0.5
                yaw_alignment = -1.0
            elif training_stage == 3:
                feet_swing_height = -5.0
                feet_air_time = 0.1
                feet_stumble = -0.8
                feet_obstacle_contact = -2.0
                collision = -1.0
                feet_step = -0.5
                yaw_alignment = -1.0
            elif training_stage == 4:
                crossing_pause = 1.0
                feet_swing_height = -5.0
                feet_air_time = 0.1
                feet_stumble = -0.8
                feet_obstacle_contact = -2.0
                trailing_clearance = -2.0
                collision = -1.0
                feet_step = -0.5
                contact_no_vel = -0.5
                yaw_alignment = -1.0

    class noise:
        add_noise = False
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0  # lin_vel is privileged
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0

    class commands:
        curriculum = False
        max_lin_vel_forward_x_curriculum = 1.0
        max_lin_vel_backward_x_curriculum = 0.0
        max_lin_vel_y_curriculum = 0.0
        max_ang_vel_yaw_curriculum = 1.0

        num_commands = 4  # lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 10.
        if training_stage == 1:
            heading_command = True
        elif training_stage == 2:
            heading_command = False
        elif training_stage == 3 or training_stage == 4:
            heading_command = True  # lock heading to world +x

        use_stop_and_go = False
        moving_time_range = [3.0, 6.0]
        stop_time_range = [2.0, 4.0]
        crossing_pause_time_range = [1.5, 3.0]
        crossing_resume_cmd = 0.2       # soft forward speed after pause
        crossing_resume_max_time = 2.0  # force full cmd even if not both-past yet

        class ranges:
            if training_stage == 1:
                # full cmd_vel
                lin_vel_x = [-1.0, 1.0]  # min max [m/s]
                lin_vel_y = [-1.0, 1.0]  # min max [m/s]
                ang_vel_yaw = [-3.14, 3.14]  # min max [rad]
                heading = [-3.14, 3.14]  # min max [rad/s]
            elif training_stage == 2 or training_stage == 3 or training_stage == 4:
                # forward-only cmd_vel
                lin_vel_x = [0.0, 1.0]  # min max [m/s]
                lin_vel_y = [0.0, 0.0]  # min max [m/s]
                ang_vel_yaw = [0.0, 0.0]  # min max [rad]
                heading = [0.0, 0.0]  # min max [rad/s]
