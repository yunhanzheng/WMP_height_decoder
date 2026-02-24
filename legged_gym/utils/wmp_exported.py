"""
WMP export utilities – export actor + world model for MuJoCo deployment.

Public API
----------
export_wmp(out_dir, actor_critic, world_model, env, runner)
    → (actor_path: str, wm_path: str)

WMPWorldModelStep
    Stateful wrapper around the Dreamer world model (training/eval side).
    wm = WMPWorldModelStep.from_file("world_model_step.pt", num_envs=1, device="cpu")
    wm_feature = wm(prop, action_flat, is_first)
    wm.reset(env_ids)
"""

import os
import copy
import shutil
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# JIT-scriptable actor wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ActorWMPPolicy(nn.Module):
    """Packages history_encoder + wm_feature_encoder + actor for TorchScript."""

    privileged_dim: int  # TorchScript needs explicit type annotation for int attrs

    def __init__(self, actor_critic, privileged_dim: int):
        super().__init__()
        self.history_encoder    = copy.deepcopy(actor_critic.history_encoder).cpu()
        self.wm_feature_encoder = copy.deepcopy(actor_critic.wm_feature_encoder).cpu()
        self.actor              = copy.deepcopy(actor_critic.actor).cpu()
        self.privileged_dim     = privileged_dim

    def forward(
        self,
        obs:        torch.Tensor,
        history:    torch.Tensor,
        wm_feature: torch.Tensor,
    ) -> torch.Tensor:
        latent    = self.history_encoder(history)
        command   = obs[:, self.privileged_dim + 6 : self.privileged_dim + 9]
        wm_latent = self.wm_feature_encoder(wm_feature)
        concat    = torch.cat([latent, command, wm_latent], dim=-1)
        return self.actor(concat)


# ─────────────────────────────────────────────────────────────────────────────
# Deployment-compatible weight extractor
# Mirrors the submodule structure of the deployment WMPWorldModelStep.__init__
# so that state_dict() produces keys the deployment from_file() can load.
# ─────────────────────────────────────────────────────────────────────────────

class _DeployStep(nn.Module):
    def __init__(self, world_model, batch_size: int = 1):
        super().__init__()
        dyn = world_model.dynamics
        enc = world_model.encoder

        stoch    = int(dyn._stoch)
        deter    = int(dyn._deter)
        discrete = int(dyn._discrete)

        self._enc_mlp         = copy.deepcopy(enc._mlp).cpu() if hasattr(enc, '_mlp') else None
        self._img_in_layers   = copy.deepcopy(dyn._img_in_layers).cpu()
        self._cell            = copy.deepcopy(dyn._cell).cpu()
        self._img_out_layers  = copy.deepcopy(dyn._img_out_layers).cpu()
        self._obs_out_layers  = copy.deepcopy(dyn._obs_out_layers).cpu()
        self._obs_stat_layer  = copy.deepcopy(dyn._obs_stat_layer).cpu()
        self._imgs_stat_layer = copy.deepcopy(dyn._imgs_stat_layer).cpu()

        learned_init = hasattr(dyn, 'W') and getattr(dyn, '_initial', '') == 'learned'
        if learned_init:
            self.register_buffer('_W', copy.deepcopy(dyn.W).cpu().detach())
        else:
            self.register_buffer('_W', torch.zeros(1, deter))

        self.register_buffer('_deter_buf', torch.zeros(batch_size, deter))
        if discrete:
            self.register_buffer('_stoch_buf', torch.zeros(batch_size, stoch, discrete))
            self.register_buffer('_logit_buf', torch.zeros(batch_size, stoch, discrete))
        else:
            self.register_buffer('_stoch_buf', torch.zeros(batch_size, stoch))
            self.register_buffer('_logit_buf', torch.zeros(batch_size, stoch))


# ─────────────────────────────────────────────────────────────────────────────
# Stateful world-model step wrapper (training / eval side, requires dreamer)
# ─────────────────────────────────────────────────────────────────────────────

class WMPWorldModelStep(nn.Module):
    """Stateful Dreamer world-model wrapper for deployment / MuJoCo sim.

    The latent state is stored internally; call reset() on episode boundaries.
    """

    def __init__(self, encoder, dynamics, num_envs: int = 1, device: str = "cpu"):
        super().__init__()
        self.encoder   = encoder
        self.dynamics  = dynamics
        self._latent   = None          # initialised on first forward pass
        self._num_envs = num_envs
        self._device   = device

    # ------------------------------------------------------------------
    def forward(
        self,
        prop:        torch.Tensor,
        action_flat: torch.Tensor,
        is_first:    torch.Tensor,
    ) -> torch.Tensor:
        """Run one world-model obs step and return the deter feature."""
        wm_obs = {"prop": prop, "is_first": is_first}
        embed  = self.encoder(wm_obs)
        prev_action = action_flat if self._latent is not None else None
        self._latent, _ = self.dynamics.obs_step(
            self._latent, prev_action, embed, is_first, sample=True
        )
        return self.dynamics.get_deter_feat(self._latent)

    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Zero-out the latent state for the given environment indices."""
        if self._latent is None or len(env_ids) == 0:
            return
        for key, val in self._latent.items():
            if isinstance(val, torch.Tensor):
                self._latent[key][env_ids] = 0.0

    # ------------------------------------------------------------------
    @classmethod
    def from_file(
        cls,
        path:      str,
        num_envs:  int = 1,
        device:    str = "cpu",
    ) -> "WMPWorldModelStep":
        """Reconstruct the world model from a saved checkpoint file."""
        from dreamer.models import WorldModel

        ckpt        = torch.load(path, map_location=device)
        wm_config   = ckpt["wm_config"]
        obs_shape   = ckpt["obs_shape"]
        use_camera  = ckpt["use_camera"]

        world_model = WorldModel(wm_config, obs_shape, use_camera=use_camera)
        world_model.load_state_dict(ckpt["world_model_state"])
        world_model = world_model.to(device)

        obj = cls(
            encoder   = world_model.encoder,
            dynamics  = world_model.dynamics,
            num_envs  = num_envs,
            device    = device,
        )
        obj.to(device)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Main export entry point
# ─────────────────────────────────────────────────────────────────────────────

def export_wmp(out_dir: str, actor_critic, world_model, env, runner):
    """Export the complete WMP actor + world model for MuJoCo deployment.

    Writes to *out_dir*:
      actor_policy.pt      – TorchScript actor (torch.jit.load)
      world_model_step.pt  – world-model checkpoint (WMPWorldModelStep.from_file)
      export_wmp_policy.py – copy of this module (needed by demo.py)
      demo.py              – usage example

    Returns
    -------
    actor_path : str
    wm_path    : str
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Actor (TorchScript) ────────────────────────────────────────────────
    actor_module = ActorWMPPolicy(actor_critic, privileged_dim=env.privileged_dim)
    actor_module.eval()
    scripted_actor = torch.jit.script(actor_module)
    actor_path = os.path.join(out_dir, "actor_policy.pt")
    scripted_actor.save(actor_path)

    # ── 2. World-model checkpoint ─────────────────────────────────────────────
    prop_dim    = env.num_obs - env.privileged_dim - env.height_dim - env.num_actions
    image_shape = env.cfg.depth.resized + (1,)
    obs_shape   = {"prop": (prop_dim,), "image": image_shape}

    dyn = world_model.dynamics
    enc = world_model.encoder

    enc_cnn_dim  = enc._cnn.outdim if hasattr(enc, '_cnn') else 0
    learned_init = hasattr(dyn, 'W') and getattr(dyn, '_initial', '') == 'learned'

    # Scalar config dict expected by deployment wmp_world_model.py from_file()
    deploy_config = {
        'stoch':        int(dyn._stoch),
        'deter':        int(dyn._deter),
        'discrete':     int(dyn._discrete),
        'num_actions':  int(dyn._num_actions),
        'batch_size':   1,
        'enc_cnn_dim':  enc_cnn_dim,
        'learned_init': learned_init,
        'use_camera':   env.cfg.depth.use_camera,
    }

    # State dict with key names matching the deployment WMPWorldModelStep structure
    deploy_step = _DeployStep(world_model, batch_size=1)

    wm_ckpt = {
        # ── deployment side (wmp_world_model.py from_file) ────────────────────
        "config":            deploy_config,
        "state_dict":        deploy_step.state_dict(),
        # ── training/eval side (WMPWorldModelStep.from_file above) ───────────
        "wm_config":         runner.wm_config,
        "obs_shape":         obs_shape,
        "use_camera":        env.cfg.depth.use_camera,
        "world_model_state": world_model.state_dict(),
    }
    wm_path = os.path.join(out_dir, "world_model_step.pt")
    torch.save(wm_ckpt, wm_path)

    # ── 3. Copy this file as export_wmp_policy.py ─────────────────────────────
    shutil.copy2(os.path.abspath(__file__), os.path.join(out_dir, "export_wmp_policy.py"))

    # ── 4. Write demo.py ──────────────────────────────────────────────────────
    priv_dim           = env.privileged_dim
    prop_end           = priv_dim + env.cfg.env.prop_dim
    traj_end           = env.num_obs - env.height_dim
    history_length     = runner.history_length
    history_obs_dim    = env.num_obs - env.privileged_dim - env.height_dim - 3
    wm_update_interval = env.cfg.depth.update_interval
    num_actions        = env.num_actions
    wm_feature_dim     = runner.wm_feature_dim

    demo_src = (
        "import torch\n"
        "from export_wmp_policy import WMPWorldModelStep\n"
        "\n"
        'actor = torch.jit.load("actor_policy.pt").eval()\n'
        f'wm    = WMPWorldModelStep.from_file("world_model_step.pt", num_envs=1, device="cpu").eval()\n'
        "\n"
        "# Buffers managed by caller (async update rates)\n"
        f"history    = torch.zeros(1, {history_length}, {history_obs_dim})   # trajectory history\n"
        f"act_hist   = torch.zeros(1, {wm_update_interval}, {num_actions})   # action window for WM\n"
        f"wm_feature = torch.zeros(1, {wm_feature_dim})                      # last deter state\n"
        "\n"
        "step = 0\n"
        "is_first = torch.ones(1)\n"
        "\n"
        "while running:\n"
        f"    # ── world model (every {wm_update_interval} steps) ────────────────────────────────\n"
        f"    if step % {wm_update_interval} == 0:\n"
        f"        prop        = obs[:, {priv_dim}:{prop_end}]\n"
        "        action_flat = act_hist.flatten(1)\n"
        "        wm_feature  = wm(prop, action_flat, is_first)\n"
        "\n"
        "    # ── trajectory history ────────────────────────────────────────────────\n"
        f"    obs_slice_t = torch.cat([obs[:, {priv_dim}:{priv_dim + 6}], obs[:, {priv_dim + 9}:{traj_end}]], dim=1)\n"
        "    history     = torch.roll(history, -1, dims=1)\n"
        "    history[:, -1] = obs_slice_t\n"
        "\n"
        "    # ── actor (every step) ───────────────────────────────────────────────\n"
        "    with torch.no_grad():\n"
        "        actions = actor(obs, history.flatten(1), wm_feature)\n"
        "\n"
        "    obs, done, *_ = env.step(actions)\n"
        "    is_first = done.float()\n"
        "\n"
        "    # ── update action window ─────────────────────────────────────────────\n"
        "    act_hist = torch.roll(act_hist, -1, dims=1)\n"
        "    act_hist[:, -1] = actions\n"
        "\n"
        "    # reset history on episode end\n"
        "    history[done.bool()]  = 0.0\n"
        "    act_hist[done.bool()] = 0.0\n"
        "    wm.reset(done.nonzero(as_tuple=False).flatten())\n"
        "    step += 1\n"
    )

    with open(os.path.join(out_dir, "demo.py"), "w") as fh:
        fh.write(demo_src)

    return actor_path, wm_path
