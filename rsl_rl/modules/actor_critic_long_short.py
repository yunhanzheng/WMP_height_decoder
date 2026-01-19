# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
from torch.distributions import Normal
from enum import Enum


class PolicyMode(str, Enum):
    CURRENT = "current"
    CUR_SHORT = "current+short"
    CUR_LONG = "current+long"
    CUR_SHORT_LONG = "current+short+long"
    SHORT_LONG = "short+long"


class CNNHistoryEncoder(nn.Module):
    """
    CNN encoder for long-term history.
    Input:  x ∈ [B, T, P]  (e.g., [B, 100, 42])
    Output: z ∈ [B, output_size]
    """
    def __init__(self, input_dim, history_length, output_size=32, activation=nn.ELU()):
        super().__init__()
        self.input_dim = input_dim
        self.history_length = history_length
        self.output_size = output_size
        self.act = activation

        # Conv layers: reduce temporal dimension while extracting features
        # Input: [B, input_dim, history_length]
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=6, stride=3, padding=0)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=4, stride=2, padding=0)

        # Calculate output size after convolutions
        # After conv1: (history_length - 6) / 3 + 1
        # After conv2: (conv1_out - 4) / 2 + 1
        conv1_out = (history_length - 6) // 3 + 1
        conv2_out = (conv1_out - 4) // 2 + 1
        flatten_size = 16 * conv2_out

        self.flatten = nn.Flatten()
        self.head = nn.Linear(flatten_size, output_size)

    def forward(self, x):
        # x: [B, T, P] -> [B, P, T] for Conv1d
        x = x.permute(0, 2, 1)
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.flatten(y)
        z = self.head(y)
        return z


class Actor(nn.Module):
    """
    Actor network with long and short term memory encoding.

    Observation layout expected:
    [command(3), current_obs(prop_dim+3), terrain_one_hot(1), short_hist(short_len*prop_dim), long_hist(long_len*prop_dim)]
    """
    def __init__(self,
                 num_proprio_obs,
                 num_actions,
                 num_long_history_length,
                 num_short_history_length,
                 dim_long_history_latent,
                 actor_hidden_dims,
                 activation,
                 policy_mode="current+short+long"):
        super().__init__()

        self.policy_mode = PolicyMode(policy_mode)

        self.long_history_length = num_long_history_length
        self.short_history_length = num_short_history_length
        self.dim_long_history_latent = dim_long_history_latent
        self.proprio_obs_dim = num_proprio_obs
        self.num_actions = num_actions

        # Flags for which components to use
        self.use_current = self.policy_mode in {
            PolicyMode.CURRENT, PolicyMode.CUR_SHORT, PolicyMode.CUR_LONG, PolicyMode.CUR_SHORT_LONG
        }
        self.use_short = self.policy_mode in {
            PolicyMode.CUR_SHORT, PolicyMode.CUR_SHORT_LONG, PolicyMode.SHORT_LONG
        }
        self.use_long = self.policy_mode in {
            PolicyMode.CUR_LONG, PolicyMode.CUR_SHORT_LONG, PolicyMode.SHORT_LONG
        }

        # Long-term memory encoder (CNN)
        if self.use_long:
            self.long_encoder = CNNHistoryEncoder(
                input_dim=self.proprio_obs_dim,
                history_length=self.long_history_length,
                output_size=self.dim_long_history_latent,
                activation=activation
            )
        else:
            self.long_encoder = None

        # Short-term memory encoder (MLP projection of short history frames)
        if self.use_short:
            short_input_dim = self.short_history_length * self.proprio_obs_dim
            H_1, H_2, L = 128, 64, self.dim_long_history_latent
            self.short_proj = nn.Sequential(
                nn.Linear(short_input_dim, H_1),
                activation,
                nn.Linear(H_1, H_2),
                activation,
                nn.Linear(H_2, L)
            )
        else:
            self.short_proj = None

        # Compute actor backbone input dimension
        input_dim = 3  # command
        if self.use_current:
            input_dim += (self.proprio_obs_dim + 3)  # current_obs
        if self.use_short:
            input_dim += self.dim_long_history_latent
        if self.use_long:
            input_dim += self.dim_long_history_latent

        # Actor MLP
        layers = [nn.Linear(input_dim, actor_hidden_dims[0]), activation]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                layers.append(nn.Linear(actor_hidden_dims[l], self.num_actions))
            else:
                layers += [nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l+1]), activation]
        self.actor_backbone = nn.Sequential(*layers)

    def _compute_latents(self, obs):
        """Parse observation and compute latent representations."""
        B = obs.shape[0]
        P = self.proprio_obs_dim
        terrain_one_hot_dim = 1

        expected_obs_dim = 3 + P + terrain_one_hot_dim + 3 + \
                           self.short_history_length * P + \
                           self.long_history_length * P
        assert obs.shape[1] == expected_obs_dim, \
            f"Expected obs.shape[1]={expected_obs_dim}, got {obs.shape[1]}"

        command_obs = obs[:, :3]
        current_obs = obs[:, 3: 3 + P + 3]

        flat_short = obs[:, 3 + P + terrain_one_hot_dim + 3:
                            3 + P + terrain_one_hot_dim + 3 + self.short_history_length * P]
        short_seq = flat_short.view(B, self.short_history_length, P)

        flat_long = obs[:, 3 + P + terrain_one_hot_dim + 3 + self.short_history_length * P:]
        long_seq = flat_long.view(B, self.long_history_length, P)

        out = {"command_obs": command_obs}

        if self.use_current:
            out["current_obs"] = current_obs

        if self.use_short:
            # Use all frames from short history
            short_flat = short_seq.contiguous().view(B, -1)
            out["short_proj"] = self.short_proj(short_flat)

        if self.use_long:
            out["latent_long"] = self.long_encoder(long_seq)

        return out

    def forward(self, obs, return_latent=False):
        lat = self._compute_latents(obs)

        x_parts = [lat["command_obs"]]

        if self.policy_mode == PolicyMode.CURRENT:
            x_parts += [lat["current_obs"]]
        elif self.policy_mode == PolicyMode.CUR_SHORT:
            x_parts += [lat["current_obs"], lat["short_proj"]]
        elif self.policy_mode == PolicyMode.CUR_LONG:
            x_parts += [lat["current_obs"], lat["latent_long"]]
        elif self.policy_mode == PolicyMode.CUR_SHORT_LONG:
            x_parts += [lat["current_obs"], lat["short_proj"], lat["latent_long"]]
        elif self.policy_mode == PolicyMode.SHORT_LONG:
            x_parts += [lat["short_proj"], lat["latent_long"]]
        else:
            raise ValueError(f"Unhandled policy_mode: {self.policy_mode}")

        x = torch.cat(x_parts, dim=-1)
        actions = self.actor_backbone(x)

        if return_latent:
            return actions, lat.get("short_proj", None), lat.get("latent_long", None)
        return actions


class ActorCriticLongShort(nn.Module):
    """
    Actor-Critic with long and short term memory.
    - Actor: current proprioceptive obs + encoded history (CNN for long, MLP for short)
    - Critic: full privileged observations
    """
    is_recurrent = False

    def __init__(self,
                 num_proprio_obs,
                 num_critic_obs,
                 num_actions,
                 num_long_history_length,
                 num_short_history_length,
                 dim_long_history_latent,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 policy_mode="current+short+long",
                 **kwargs):
        if kwargs:
            print("ActorCriticLongShort.__init__ got unexpected arguments: " + str(list(kwargs.keys())))
        super().__init__()

        activation_fn = get_activation(activation)

        self.long_history_length = num_long_history_length
        self.short_history_length = num_short_history_length
        self.dim_long_history_latent = dim_long_history_latent
        self.proprio_obs_dim = num_proprio_obs
        self.num_actions = num_actions

        print(f"ActorCriticLongShort | Policy mode: {policy_mode}")
        print(f"  Proprio obs dim: {num_proprio_obs}, Critic obs dim: {num_critic_obs}")
        print(f"  Long history: {num_long_history_length}, Short history: {num_short_history_length}")
        print(f"  Latent dim: {dim_long_history_latent}")

        # Actor with long-short memory
        self.actor = Actor(
            num_proprio_obs,
            num_actions,
            num_long_history_length,
            num_short_history_length,
            dim_long_history_latent,
            actor_hidden_dims,
            activation_fn,
            policy_mode=policy_mode
        )

        # Critic: simple MLP on full privileged observations
        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation_fn]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers += [nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]), activation_fn]
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        mean = torch.clamp(mean, -100, 100)
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"[NaN/Inf Mean] obs stats: mean={observations.mean().item():.4f}, std={observations.std().item():.4f}")
            raise ValueError("Actor mean has NaNs or Infs!")
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.sample()
        return torch.clamp(actions, -100, 100)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        """Critic forward pass on full privileged observations."""
        return self.critic(critic_observations)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
