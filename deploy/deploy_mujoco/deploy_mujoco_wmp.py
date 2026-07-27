#!/usr/bin/env python3
"""MuJoCo sim2sim for WMP policies (Isaac Gym → MuJoCo).

Usage
-----
  # 1) Export checkpoint (headless Isaac Gym, once per checkpoint)
  python deploy/deploy_mujoco/export_wmp_checkpoint.py \\
      --task=g1_blind --load_run=Jul25_23-32-07_ --checkpoint=100000

  # 2) Run in MuJoCo
  python deploy/deploy_mujoco/deploy_mujoco_wmp.py Jul25_23-32-07_
  python deploy/deploy_mujoco/deploy_mujoco_wmp.py configs/g1_blind.yaml
"""

import argparse
import importlib.util
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

# Load WMPWorldModelStep without legged_gym.utils (that pulls in isaacgym).
_wmp_exported_path = os.path.join(REPO_ROOT, "legged_gym", "utils", "wmp_exported.py")
_spec = importlib.util.spec_from_file_location("wmp_exported", _wmp_exported_path)
_wmp_exported = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wmp_exported)
WMPWorldModelStep = _wmp_exported.WMPWorldModelStep

import torch  # after wmp_exported (dreamer); no isaacgym in this script

_stripe_path = os.path.join(os.path.dirname(__file__), "stripe_scene.py")
_stripe_spec = importlib.util.spec_from_file_location("stripe_scene", _stripe_path)
_stripe_mod = importlib.util.module_from_spec(_stripe_spec)
_stripe_spec.loader.exec_module(_stripe_mod)
resolve_scene_xml = _stripe_mod.resolve_scene_xml

_pause_path = os.path.join(os.path.dirname(__file__), "foot_cross_pause.py")
_pause_spec = importlib.util.spec_from_file_location("foot_cross_pause", _pause_path)
_pause_mod = importlib.util.module_from_spec(_pause_spec)
_pause_spec.loader.exec_module(_pause_mod)
FootCrossPause = _pause_mod.FootCrossPause


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3, dtype=np.float64)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def load_config(arg: str) -> dict:
    if arg.endswith(".yaml"):
        cfg_path = arg if os.path.isabs(arg) else os.path.join(
            os.path.dirname(__file__), "configs", os.path.basename(arg)
        )
    else:
        cfg_path = os.path.join(os.path.dirname(__file__), "configs", "g1_blind.yaml")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["load_run"] = arg
        return cfg
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def resolve_export_dir(cfg: dict) -> str:
    task = cfg.get("task", "g1_blind")
    experiment = task if task.endswith("_blind") or task.endswith("_base") else task
    if task == "g1_blind":
        experiment = "g1_blind"
    return os.path.join(
        REPO_ROOT, "logs", experiment, cfg["load_run"], "exported", "mujoco"
    )


def build_history_step(ang_vel, gravity, dof_pos_scaled, dof_vel_scaled, action):
    return np.concatenate([ang_vel, gravity, dof_pos_scaled, dof_vel_scaled, action], axis=0)


def main():
    parser = argparse.ArgumentParser(description="WMP MuJoCo sim2sim")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/g1_blind.yaml",
        help="YAML config or load_run name (uses configs/g1_blind.yaml)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    export_dir = resolve_export_dir(cfg)
    actor_path = os.path.join(export_dir, "actor_policy.pt")
    wm_path = os.path.join(export_dir, "world_model_step.pt")
    if not os.path.isfile(actor_path) or not os.path.isfile(wm_path):
        raise FileNotFoundError(
            f"Missing export in {export_dir}. Run export first:\n"
            f"  python deploy/deploy_mujoco/export_wmp_checkpoint.py "
            f"--task={cfg.get('task', 'g1_blind')} "
            f"--load_run={cfg['load_run']} --checkpoint={cfg.get('checkpoint', 100000)}"
        )

    unitree_root = os.environ.get("UNITREE_RL_GYM_ROOT", "/home/ubuntu22/data/unitree_rl_gym")
    xml_path, stripes = resolve_scene_xml(cfg, REPO_ROOT, unitree_root)
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(
            f"MuJoCo scene not found: {xml_path}\n"
            "Set UNITREE_RL_GYM_ROOT or edit terrain/xml_path in the YAML config."
        )

    privileged_dim = int(cfg["privileged_dim"])
    prop_dim = int(cfg["prop_dim"])
    height_dim = int(cfg["height_dim"])
    num_actions = int(cfg["num_actions"])
    history_length = int(cfg["history_length"])
    wm_update_interval = int(cfg["wm_update_interval"])
    wm_feature_dim = int(cfg["wm_feature_dim"])
    history_step_dim = int(cfg.get("history_step_dim", prop_dim + num_actions - 3))

    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    simulation_duration = float(cfg["simulation_duration"])

    default_angles = np.array(cfg["default_angles"], dtype=np.float32)
    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    cmd_nominal = np.array(cfg["cmd"], dtype=np.float32)
    use_stop_and_go = bool(cfg.get("use_stop_and_go", True))
    stop_time_range = tuple(cfg.get("stop_time_range", [2.0, 4.0]))
    control_dt = simulation_dt * control_decimation

    ang_vel_scale = float(cfg["ang_vel_scale"])
    dof_pos_scale = float(cfg["dof_pos_scale"])
    dof_vel_scale = float(cfg["dof_vel_scale"])
    action_scale = float(cfg["action_scale"])
    init_base_z = float(cfg.get("init_base_z", 0.793))

    num_obs = privileged_dim + prop_dim + height_dim + num_actions

    print(f"[sim2sim] export_dir={export_dir}")
    print(f"[sim2sim] xml={xml_path}")
    print(f"[sim2sim] cmd={cmd_nominal.tolist()}  duration={simulation_duration}s")
    if use_stop_and_go and stripes:
        print(f"[sim2sim] stop-and-go ON (foot_cross, pause {stop_time_range[0]}-{stop_time_range[1]}s)")

    device = "cpu"
    actor = torch.jit.load(actor_path, map_location=device).eval()
    wm = WMPWorldModelStep.from_file(wm_path, num_envs=1, device=device).eval()
    wm.to(device)
    if hasattr(wm.dynamics, "_device"):
        wm.dynamics._device = device

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    left_foot_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    pause_ctrl = None
    if use_stop_and_go and stripes:
        pause_ctrl = FootCrossPause(stripes, stop_time_range=stop_time_range)

    cmd = cmd_nominal.copy()

    d.qpos[2] = init_base_z
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:7 + num_actions] = default_angles
    mujoco.mj_forward(m, d)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    history = torch.zeros(1, history_length, history_step_dim)
    wm_action_history = torch.zeros(1, wm_update_interval, num_actions)
    wm_feature = torch.zeros(1, wm_feature_dim)
    wm_is_first = torch.ones(1)
    obs_t = torch.zeros(1, num_obs)

    counter = 0
    ctrl_step = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(
                target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds
            )
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter += 1

            if counter % control_decimation != 0:
                viewer.sync()
                _sleep(step_start, simulation_dt)
                continue

            ctrl_step += 1
            qj = d.qpos[7:7 + num_actions].copy()
            dqj = d.qvel[6:6 + num_actions].copy()
            quat = d.qpos[3:7].copy()
            omega = d.qvel[3:6].copy()

            ang_vel = omega * ang_vel_scale
            gravity = get_gravity_orientation(quat)
            dof_pos_scaled = (qj - default_angles) * dof_pos_scale
            dof_vel_scaled = dqj * dof_vel_scale

            if pause_ctrl is not None:
                foot_xy = [
                    (float(d.xpos[left_foot_bid][0]), float(d.xpos[left_foot_bid][2])),
                    (float(d.xpos[right_foot_bid][0]), float(d.xpos[right_foot_bid][2])),
                ]
                cmd = pause_ctrl.update(foot_xy, control_dt, cmd_nominal)

            obs[:] = 0.0
            obs[privileged_dim:privileged_dim + 3] = ang_vel
            obs[privileged_dim + 3:privileged_dim + 6] = gravity
            obs[privileged_dim + 6:privileged_dim + 9] = cmd
            obs[privileged_dim + 9:privileged_dim + 9 + num_actions] = dof_pos_scaled
            obs[privileged_dim + 9 + num_actions:privileged_dim + 9 + 2 * num_actions] = dof_vel_scaled
            obs[-num_actions:] = action

            hist_step = build_history_step(
                ang_vel, gravity, dof_pos_scaled, dof_vel_scaled, action
            )
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1] = torch.from_numpy(hist_step).float().unsqueeze(0)
            current_prop = history[:, -1]
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)

            if ctrl_step % wm_update_interval == 0:
                prop = obs_t[:, privileged_dim:privileged_dim + prop_dim].to(device)
                wm_action_flat = wm_action_history.reshape(1, -1).to(device)
                wm_is_first_dev = wm_is_first.to(device)
                with torch.no_grad():
                    wm_feature = wm(prop, wm_action_flat, wm_is_first_dev).cpu()
                wm_is_first[:] = 0.0

            with torch.no_grad():
                actions = actor(
                    obs_t, history.reshape(1, -1), wm_feature, current_prop
                )
            action = actions.squeeze(0).cpu().numpy().astype(np.float32)
            target_dof_pos = action * action_scale + default_angles

            wm_action_history = torch.roll(wm_action_history, shifts=-1, dims=1)
            wm_action_history[:, -1] = torch.from_numpy(action).float().unsqueeze(0)

            viewer.sync()
            _sleep(step_start, simulation_dt)

    print(f"[sim2sim] finished ({ctrl_step} control steps)")


def _sleep(step_start, simulation_dt):
    delay = simulation_dt - (time.time() - step_start)
    if delay > 0:
        time.sleep(delay)


if __name__ == "__main__":
    main()
