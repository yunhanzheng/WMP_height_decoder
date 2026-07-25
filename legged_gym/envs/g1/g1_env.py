import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.g1.g1_base import training_stage
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi


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
        # Elevated swing while crossing a stripe; cleared when both feet are past & low.
        self.swing_clearance_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.active_stripe_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.active_stripe_x = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if training_stage == 4:
            self.crossing_pause_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.crossing_pause_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            # After one pause, block re-entry until the robot leaves the straddle.
            self.crossing_pause_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.crossing_cmd_stored = torch.zeros(self.num_envs, device=self.device)
            # After pause: lead stance + rear swing until rear clears and lands.
            self.crossing_clear_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.crossing_stripe_x = torch.zeros(self.num_envs, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.swing_clearance_active[env_ids] = False
        self.active_stripe_height[env_ids] = 0.0
        self.active_stripe_x[env_ids] = 0.0
        if training_stage == 4:
            self.crossing_pause_active[env_ids] = False
            self.crossing_pause_timer[env_ids] = 0.0
            self.crossing_pause_done[env_ids] = False
            self.crossing_cmd_stored[env_ids] = 0.0
            self.crossing_clear_active[env_ids] = False
            self.crossing_stripe_x[env_ids] = 0.0

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat(
            [self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1
        )
        super()._post_physics_step_callback()
        if training_stage in (3, 4):
            self._update_swing_clearance()
        if training_stage == 4:
            self._apply_crossing_pause_commands()

    # ----- stage helpers ------------------------------------------------
    def _forward_only_commands(self):
        y_range = self.command_ranges["lin_vel_y"]
        yaw_range = self.command_ranges["ang_vel_yaw"]
        return y_range[0] == 0.0 and y_range[1] == 0.0 and yaw_range[0] == 0.0 and yaw_range[1] == 0.0

    def _swing_clearance(self):
        """Desired clearance [m] above local terrain / stripe top."""
        return 0.08

    def _sample_terrain_height_world(self, world_x, world_y):
        """Bilinear max terrain height [m] at world (x, y) positions."""
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

    def _is_straddling_obstacle(self):
        """One foot past the stripe, one not; neither planted on the stripe top."""
        terrain_h = self._get_feet_terrain_heights()
        on_flat = terrain_h < 0.03

        foot_x = self.feet_pos[:, :, 0]
        foot_y = self.feet_pos[:, :, 1]
        rear_x = foot_x.min(dim=1).values
        lead_x = foot_x.max(dim=1).values
        mid_y = foot_y.mean(dim=1)
        foot_spread = lead_x - rear_x

        lead_idx = foot_x.argmax(dim=1)
        rear_idx = foot_x.argmin(dim=1)
        lead_flat = on_flat.gather(1, lead_idx.unsqueeze(1)).squeeze(1)
        rear_flat = on_flat.gather(1, rear_idx.unsqueeze(1)).squeeze(1)

        num_samples = 5
        alphas = torch.linspace(0.0, 1.0, num_samples, device=self.device)
        sample_x = rear_x.unsqueeze(1) + alphas.unsqueeze(0) * foot_spread.unsqueeze(1)
        sample_y = mid_y.unsqueeze(1).expand(-1, num_samples)
        between_h = self._sample_terrain_height_world(sample_x.reshape(-1), sample_y.reshape(-1))
        between_h = between_h.view(self.num_envs, num_samples).max(dim=1).values

        return (
            lead_flat
            & rear_flat
            & (foot_spread > 0.20)
            & (between_h > 0.03)
        )

    def _lateral_foot_hit(self):
        """Per-foot lateral contact (same criterion as feet_stumble)."""
        xy_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        z_force = torch.abs(self.contact_forces[:, self.feet_indices, 2])
        return xy_force > 5.0 * z_force

    def _feet_near_stripe(self, dist=0.1):
        """Per-foot: raised terrain (h>3cm) within ``dist`` meters ahead of the foot."""
        foot_x = self.feet_pos[:, :, 0]
        foot_y = self.feet_pos[:, :, 1]
        max_h = torch.zeros_like(foot_x)
        for d in (0.0, 0.05, dist):
            h = self._sample_terrain_height_world(foot_x + d, foot_y)
            max_h = torch.maximum(max_h, h)
        return max_h > 0.03, max_h

    def _update_swing_clearance(self):
        """Raise swing target after near-stripe lateral hit until both feet past & low."""
        hit = self._lateral_foot_hit()
        near_stripe, near_h = self._feet_near_stripe(dist=0.1)
        hit = hit & near_stripe
        # Rising edge only — do not refresh stripe_x every frame while still rubbing.
        activating = hit.any(dim=1) & ~self.swing_clearance_active
        if activating.any():
            hit_h = near_h.clone()
            hit_h[~hit] = -1.0
            new_h, hit_idx = hit_h.max(dim=1)
            new_x = self.feet_pos[:, :, 0].gather(1, hit_idx.unsqueeze(1)).squeeze(1)
            self.swing_clearance_active |= activating
            self.active_stripe_height = torch.where(
                activating, new_h, self.active_stripe_height
            )
            self.active_stripe_x = torch.where(activating, new_x, self.active_stripe_x)

        both_past = (
            self.feet_pos[:, :, 0] > (self.active_stripe_x.unsqueeze(1) + 0.1)
        ).all(dim=1)
        both_low = (self.feet_pos[:, :, 2] <= 0.08).all(dim=1)
        cleared = self.swing_clearance_active & both_past & both_low
        self.swing_clearance_active &= ~cleared
        self.active_stripe_height = torch.where(
            cleared, torch.zeros_like(self.active_stripe_height), self.active_stripe_height
        )

    def _stripe_x_between_feet(self):
        """Approx world x of raised terrain between rear and lead foot."""
        foot_x = self.feet_pos[:, :, 0]
        foot_y = self.feet_pos[:, :, 1]
        rear_x = foot_x.min(dim=1).values
        lead_x = foot_x.max(dim=1).values
        mid_y = foot_y.mean(dim=1)
        foot_spread = (lead_x - rear_x).clamp(min=1e-3)
        num_samples = 5
        alphas = torch.linspace(0.0, 1.0, num_samples, device=self.device)
        sample_x = rear_x.unsqueeze(1) + alphas.unsqueeze(0) * foot_spread.unsqueeze(1)
        sample_y = mid_y.unsqueeze(1).expand(-1, num_samples)
        between_h = self._sample_terrain_height_world(
            sample_x.reshape(-1), sample_y.reshape(-1)
        ).view(self.num_envs, num_samples)
        _, idx = between_h.max(dim=1)
        return sample_x.gather(1, idx.unsqueeze(1)).squeeze(1)

    def _apply_crossing_pause_commands(self):
        """Auto-pause when straddling; exit only after both feet clear the stripe."""
        straddling = self._is_straddling_obstacle()

        entering = (
            straddling
            & ~self.crossing_pause_active
            & ~self.crossing_pause_done
            & ~self.crossing_clear_active
            & (self.commands[:, 0] > 0.05)
        )
        if entering.any():
            lo, hi = self.cfg.commands.crossing_pause_time_range
            self.crossing_pause_timer[entering] = (
                torch.rand(entering.sum(), device=self.device) * (hi - lo) + lo
            )
            self.crossing_cmd_stored[entering] = torch.clamp(
                self.commands[entering, 0], min=0.3
            )
            stripe_x = torch.where(
                self.swing_clearance_active,
                self.active_stripe_x,
                self._stripe_x_between_feet(),
            )
            self.crossing_stripe_x[entering] = stripe_x[entering]
            self.crossing_pause_active[entering] = True
            self.commands[entering, :3] = 0.0

        if self.crossing_pause_active.any():
            self.crossing_pause_timer[self.crossing_pause_active] -= self.dt
            finished = self.crossing_pause_active & (self.crossing_pause_timer <= 0.0)
            if finished.any():
                self.crossing_pause_active[finished] = False
                self.crossing_pause_done[finished] = True
                self.commands[finished, 0] = self.crossing_cmd_stored[finished]
                self.commands[finished, 1:3] = 0.0

                # Immediately latch elevated swing for the trailing foot.
                stripe_x = torch.where(
                    self.swing_clearance_active,
                    self.active_stripe_x,
                    self._stripe_x_between_feet(),
                )
                mid_y = self.feet_pos[:, :, 1].mean(dim=1)
                sampled_h = self._sample_terrain_height_world(stripe_x, mid_y)
                stripe_h = torch.where(
                    self.swing_clearance_active,
                    self.active_stripe_height,
                    torch.where(
                        sampled_h > 0.03, sampled_h, torch.full_like(sampled_h, 0.10)
                    ),
                )
                self.swing_clearance_active[finished] = True
                self.active_stripe_x[finished] = stripe_x[finished]
                self.active_stripe_height[finished] = stripe_h[finished]
                self.crossing_stripe_x[finished] = stripe_x[finished]
                self.crossing_clear_active[finished] = True

            self.commands[self.crossing_pause_active, :3] = 0.0

        # Restore normal gait / allow next pause only after both feet are past.
        both_past = (
            self.feet_pos[:, :, 0] > (self.crossing_stripe_x.unsqueeze(1) + 0.1)
        ).all(dim=1)
        cleared = (self.crossing_clear_active | self.crossing_pause_done) & both_past
        self.crossing_clear_active &= ~cleared
        self.crossing_pause_done &= ~cleared

    # ----- biped gait rewards -------------------------------------------
    def _reward_tracking_lin_vel(self):
        # World-frame tracking from stage 1. Forward-only stages penalize lateral drift.
        world_vel_xy = self.root_states[:, 7:9]
        if self._forward_only_commands():
            lin_vel_error = (
                torch.square(self.commands[:, 0] - world_vel_xy[:, 0])
                + torch.square(world_vel_xy[:, 1])
            )
        else:
            cmd_world_xy = quat_apply_yaw(
                self.base_quat,
                torch.cat(
                    [self.commands[:, :2], torch.zeros(self.num_envs, 1, device=self.device)],
                    dim=1,
                ),
            )[:, :2]
            lin_vel_error = torch.sum(torch.square(cmd_world_xy - world_vel_xy), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_yaw_alignment(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        if self._forward_only_commands():
            return torch.square(heading)
        if self.cfg.commands.heading_command:
            return torch.square(wrap_to_pi(self.commands[:, 3] - heading))
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_contact(self):
        # Encourage the expected stance/swing foot per gait phase.
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            res += ~(contact ^ is_stance)
        if training_stage == 4:
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
            # Pause: double support.
            double_support = contact.all(dim=1).float() * float(self.feet_num)
            res = torch.where(self.crossing_pause_active, double_support, res)
            # After pause: lead stance + rear swing until rear clears and lands.
            lead_idx = self.feet_pos[:, :, 0].argmax(dim=1)
            rear_idx = self.feet_pos[:, :, 0].argmin(dim=1)
            lead_contact = contact.gather(1, lead_idx.unsqueeze(1)).squeeze(1)
            rear_contact = contact.gather(1, rear_idx.unsqueeze(1)).squeeze(1)
            clear_contact = lead_contact.float() + (~rear_contact).float()
            res = torch.where(self.crossing_clear_active, clear_contact, res)
        return res

    def _get_feet_terrain_heights(self):
        """World-frame terrain height [m] under each foot."""
        foot_xy = self.feet_pos[:, :, :2] + self.terrain.cfg.border_size
        foot_px = torch.clip(
            (foot_xy[:, :, 0] / self.terrain.cfg.horizontal_scale).long(),
            0, self.height_samples.shape[0] - 2,
        )
        foot_py = torch.clip(
            (foot_xy[:, :, 1] / self.terrain.cfg.horizontal_scale).long(),
            0, self.height_samples.shape[1] - 2,
        )
        h1 = self.height_samples[foot_px, foot_py]
        h2 = self.height_samples[foot_px + 1, foot_py]
        h3 = self.height_samples[foot_px, foot_py + 1]
        h4 = self.height_samples[foot_px + 1, foot_py + 1]
        terrain_h = torch.max(torch.max(h1, h2), torch.max(h3, h4))
        return terrain_h.float() * self.terrain.cfg.vertical_scale

    def _reward_feet_obstacle_contact(self):
        # Stages 3-4: penalize planting feet on raised stripes.
        if training_stage not in (3, 4):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        terrain_h = self._get_feet_terrain_heights()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        on_stripe = contact & (terrain_h > 0.03)
        return on_stripe.float().sum(dim=1)

    def _reward_crossing_pause(self):
        # Stage 4: reward holding still only during the pause hold (not soft resume).
        if training_stage != 4:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        world_speed_sq = torch.sum(torch.square(self.root_states[:, 7:9]), dim=1)
        stillness = torch.exp(-world_speed_sq / 0.04)
        return self.crossing_pause_active.float() * stillness

    def _reward_trailing_clearance(self):
        # Stage 4: penalize swing foot passing over stripe without enough height.
        if training_stage != 4:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        terrain_h = self._get_feet_terrain_heights()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        swing = ~contact
        over_stripe = terrain_h > 0.03
        clearance_margin = 0.08
        too_low = swing & over_stripe & (self.feet_pos[:, :, 2] < terrain_h + clearance_margin)
        return too_low.float().sum(dim=1)

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        clearance = self._swing_clearance()
        if training_stage == 1:
            target_z = torch.full(
                (self.num_envs, self.feet_num), clearance, device=self.device
            )
            height_err = self.feet_pos[:, :, 2] - target_z
            pos_error = torch.square(height_err) * ~contact
        else:
            terrain_h = self._get_feet_terrain_heights()
            target_z = terrain_h + clearance
            if training_stage in (3, 4):
                # While crossing: track stripe_h+0.08; restore to terrain+0.08 after
                # both feet are past the stripe and foot z <= 0.08.
                high_z = (self.active_stripe_height + clearance).unsqueeze(1).expand_as(target_z)
                target_z = torch.where(
                    self.swing_clearance_active.unsqueeze(1), high_z, target_z
                )
            height_err = self.feet_pos[:, :, 2] - target_z
            # Crossing: only penalize under-clearance (too high is OK). Otherwise squared error.
            if training_stage in (3, 4):
                err = torch.where(
                    self.swing_clearance_active.unsqueeze(1),
                    torch.square(torch.clamp(-height_err, min=0.0)),
                    torch.square(height_err),
                )
            else:
                err = torch.square(height_err)
            pos_error = err * ~contact
        return torch.sum(pos_error, dim=1)

    def _reward_feet_step(self):
        feet_heights = self.feet_pos[:, :, 2].reshape(-1)
        z_forces = torch.abs(self.contact_forces[:, self.feet_indices, 2].reshape(-1))
        z_forces[feet_heights < 0.05] = 0
        z_ans = z_forces.view(-1, self.feet_num).sum(dim=1)
        z_ans[z_ans > 2] = 1
        return z_ans

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))
