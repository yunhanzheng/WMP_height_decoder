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
        return super()._post_physics_step_callback()

    # ----- biped gait rewards -------------------------------------------
    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_contact(self):
        # Encourage the expected stance/swing foot per gait phase.
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=1)

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))
