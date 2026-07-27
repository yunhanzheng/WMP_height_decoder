#!/usr/bin/env python3
"""Export a WMP checkpoint (actor + world model) for MuJoCo sim2sim.

Requires Isaac Gym (headless). Example:

  python deploy/deploy_mujoco/export_wmp_checkpoint.py \\
      --task=g1_blind --load_run=Jul25_23-32-07_ --checkpoint=100000
"""

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
repo_root = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, repo_root)

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import export_wmp, get_args, task_registry


def main():
    argv = sys.argv[1:]
    if "--headless" not in argv:
        argv = ["--headless"] + argv
    sys.argv = [sys.argv[0]] + argv
    args = get_args()
    args.rl_device = args.sim_device

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()

    train_cfg.runner.resume = True
    train_cfg.runner.use_wandb = False
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )

    run_dir = os.path.join(
        repo_root, "logs", train_cfg.runner.experiment_name, args.load_run
    )
    export_dir = os.path.join(run_dir, "exported", "mujoco")
    actor_path, wm_path = export_wmp(
        out_dir=export_dir,
        actor_critic=ppo_runner.alg.actor_critic,
        world_model=ppo_runner._world_model,
        env=env,
        runner=ppo_runner,
    )
    print(f"Exported actor → {actor_path}")
    print(f"Exported WM    → {wm_path}")
    print(f"Run sim2sim: python deploy/deploy_mujoco/deploy_mujoco_wmp.py {args.load_run}")


if __name__ == "__main__":
    main()
