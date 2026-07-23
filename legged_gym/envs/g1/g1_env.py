import numpy as np
import torch

from isaacgym import gymtorch

from legged_gym.envs.base.legged_robot import LeggedRobot


class G1Robot(LeggedRobot):
    """Unitree G1 (12-DoF biped) on top of the WMP base env.

    The base env is quadruped-centric. This subclass:
      * redefines ``hip_indices`` to the biped's hip-roll/yaw joints,
      * tracks per-foot state and a periodic gait phase,
      * adds biped gait-shaping rewards (alive / contact / feet_swing_height /
        contact_no_vel) used by the G1 reward scales.

    The gait phase is used only for rewards, not appended to the observation,
    so the WMP observation layout (prop / privileged / height / action) is
    unchanged.
    """

    # ----- buffers -------------------------------------------------------
    def _init_buffers(self):
        super()._init_buffers()
        # G1 DoF order (from URDF): per leg = hip_pitch, hip_roll, hip_yaw, knee,
        # ankle_pitch, ankle_roll. Penalize lateral hip motion (roll + yaw).
        self.hip_indices = torch.tensor([1, 2, 7, 8], device=self.device, dtype=torch.long)
        self._init_foot()

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        # phase buffers (filled in _post_physics_step_callback)
        self.leg_phase = torch.zeros(self.num_envs, self.feet_num, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.cmd_was_moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Elevated swing after lateral stripe hit; cleared once both feet have crossed
        self.swing_clearance_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.active_stripe_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.active_stripe_x = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.phase[env_ids] = 0.0
        self.cmd_was_moving[env_ids] = False
        self.swing_clearance_active[env_ids] = False
        self.active_stripe_height[env_ids] = 0.0
        self.active_stripe_x[env_ids] = 0.0

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _cmd_is_moving(self):
        return torch.norm(self.commands[:, :2], dim=1) > 0.1

    def _sample_terrain_height_world(self, world_x, world_y):
        """Terrain height [m] at world (x, y). Uses max of neighboring cells."""
        if self.cfg.terrain.mesh_type == "plane" or self.height_samples is None:
            return torch.zeros_like(world_x, dtype=torch.float, device=self.device)
        border = self.terrain.cfg.border_size
        hs = self.terrain.cfg.horizontal_scale
        vs = self.terrain.cfg.vertical_scale
        px = torch.clip(((world_x + border) / hs).long(), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(((world_y + border) / hs).long(), 0, self.height_samples.shape[1] - 2)
        h1 = self.height_samples[px, py]
        h2 = self.height_samples[px + 1, py]
        h3 = self.height_samples[px, py + 1]
        h4 = self.height_samples[px + 1, py + 1]
        return torch.max(torch.max(h1, h2), torch.max(h3, h4)).float() * vs

    def _get_base_terrain_height(self):
        """World-frame terrain height [m] under the base."""
        return self._sample_terrain_height_world(
            self.root_states[:, 0], self.root_states[:, 1]
        )

    def _update_swing_clearance(self):
        """On lateral stripe hit: target = stripe_h+0.08; clear after both feet cross & are low."""
        xy_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        z_force = torch.abs(self.contact_forces[:, self.feet_indices, 2])
        hit = xy_force > 5.0 * z_force

        if hit.any():
            ahead_x = self.feet_pos[:, :, 0] + 0.1
            stripe_h = torch.maximum(
                self._sample_terrain_height_world(self.feet_pos[:, :, 0], self.feet_pos[:, :, 1]),
                self._sample_terrain_height_world(ahead_x, self.feet_pos[:, :, 1]),
            )
            hit_h = stripe_h.clone()
            hit_h[~hit] = -1.0
            new_h, hit_idx = hit_h.max(dim=1)
            new_x = self.feet_pos[:, :, 0].gather(1, hit_idx.unsqueeze(1)).squeeze(1)
            new_h = torch.where(new_h > 0.03, new_h, torch.full_like(new_h, 0.10))
            activating = hit.any(dim=1)
            self.swing_clearance_active |= activating
            self.active_stripe_height = torch.where(
                activating, new_h, self.active_stripe_height
            )
            self.active_stripe_x = torch.where(activating, new_x, self.active_stripe_x)
            # Reset contact gait phase on lateral stripe hit
            self.phase[activating] = 0.0

        both_past = (self.feet_pos[:, :, 0] > (self.active_stripe_x.unsqueeze(1) + 0.1)).all(dim=1)
        both_low = (self.feet_pos[:, :, 2] <= 0.08).all(dim=1)
        cleared = self.swing_clearance_active & both_past & both_low
        self.swing_clearance_active &= ~cleared
        self.active_stripe_height = torch.where(
            cleared, torch.zeros_like(self.active_stripe_height), self.active_stripe_height
        )

    def _post_physics_step_callback(self):
        self.update_feet_state()
        self._update_swing_clearance()

        moving = self._cmd_is_moving()
        # Restart gait phase when command resumes after a stop
        resumed = moving & ~self.cmd_was_moving
        if resumed.any():
            self.phase[resumed] = 0.0
        self.cmd_was_moving[:] = moving

        period = 0.8
        offset = 0.5
        # Freeze phase while stopped so resume starts from a clean cycle
        self.phase = torch.where(
            moving,
            (self.phase + self.dt / period) % 1.0,
            self.phase,
        )
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat(
            [self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1
        )
        return super()._post_physics_step_callback()

    # ----- biped gait rewards -------------------------------------------
    def _forward_only_commands(self):
        y_range = self.command_ranges["lin_vel_y"]
        yaw_range = self.command_ranges["ang_vel_yaw"]
        return (
            y_range[0] == 0.0
            and y_range[1] == 0.0
            and yaw_range[0] == 0.0
            and yaw_range[1] == 0.0
        )

    def _reward_tracking_lin_vel(self):
        # Track linear velocity in world frame (not body frame).
        # Body-frame tracking allows turning 90° and still scoring high on cmd_x.
        world_vel_xy = self.root_states[:, 7:9]
        if self._forward_only_commands():
            lin_vel_error = (
                torch.square(self.commands[:, 0] - world_vel_xy[:, 0])
                + torch.square(world_vel_xy[:, 1])
            )
        else:
            # Stage 1: commands[:, :2] are world-frame vx, vy
            lin_vel_error = torch.sum(
                torch.square(self.commands[:, :2] - world_vel_xy), dim=1
            )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_contact(self):
        # Phase-matched stance/swing; disabled while cmd ≈ 0 (stop-and-go / stand still)
        moving = self._cmd_is_moving()
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            res += ~(contact ^ is_stance)
        return res * moving.float()

    def _reward_base_height(self):
        # Target = terrain height under base + configured clearance (0.78 m)
        target = self._get_base_terrain_height() + self.cfg.rewards.base_height_target
        return torch.square(self.root_states[:, 2] - target)

    def _reward_feet_swing_height(self):
        # Default 0.08 m; after lateral stripe hit use stripe_h+0.08 until both feet have crossed
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        target_z = torch.where(
            self.swing_clearance_active,
            self.active_stripe_height + 0.08,
            torch.full_like(self.active_stripe_height, 0.08),
        )
        pos_error = torch.square(self.feet_pos[:, :, 2] - target_z.unsqueeze(1)) * ~contact
        return torch.sum(pos_error, dim=1)

    def _reward_feet_step(self):
        # Biped version of obstacle-stepping penalty (base uses .view(-1, 4)).
        # Penalize vertical contact force when the foot is raised above flat ground.
        feet_heights = self.feet_pos[:, :, 2].reshape(-1)
        z_forces = torch.abs(self.contact_forces[:, self.feet_indices, 2]).reshape(-1)
        z_forces[feet_heights < 0.05] = 0
        z_ans = z_forces.view(-1, self.feet_num).sum(dim=1)
        z_ans[z_ans > 2] = 1
        return z_ans

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))
