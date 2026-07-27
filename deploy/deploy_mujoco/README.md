# WMP MuJoCo Sim2Sim

Deploy Isaac Gym–trained WMP policies in MuJoCo (same workflow as [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)).

**Train (Isaac Gym)** → **Play** → **Sim2Sim (MuJoCo)** → Sim2Real

## Prerequisites

- Conda env `wmp` with Isaac Gym + PyTorch (for export)
- `pip install mujoco` (for sim2sim)
- Unitree G1 MuJoCo assets (default path: `/home/ubuntu22/data/unitree_rl_gym`)

## 1. Export checkpoint

```bash
cd /home/ubuntu22/data/WMP_height_decoder

python deploy/deploy_mujoco/export_wmp_checkpoint.py \
    --task=g1_blind \
    --load_run=Jul25_23-32-07_ \
    --checkpoint=100000
```

Writes to `logs/g1_blind/<load_run>/exported/mujoco/`:
- `actor_policy.pt` — TorchScript actor
- `world_model_step.pt` — RSSM checkpoint
- `export_wmp_policy.py`, `demo.py`

## 2. Run MuJoCo sim2sim

```bash
python deploy/deploy_mujoco/deploy_mujoco_wmp.py Jul25_23-32-07_
# or
python deploy/deploy_mujoco/deploy_mujoco_wmp.py configs/g1_blind.yaml
```

Edit `configs/g1_blind.yaml` for `cmd`, duration, or `xml_path`.

## Notes

- MuJoCo uses the **Unitree `g1_description`** robot model.
- Default **`terrain: stripe`** adds gray full-width bars (sparse, stage-2 style). Use `terrain: flat` for plain floor.
- Default **`use_stop_and_go: true`**: pause 2–4s when a foot crosses a stripe and lands (`foot_cross`, same as G1 stage-2 training).
- Control rate: 50 Hz (`dt=0.002`, `decimation=10`), matching Isaac `dt=0.005` × `decimation=4`.
- If `xml_path` is missing, set `UNITREE_RL_GYM_ROOT` or clone [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym).
