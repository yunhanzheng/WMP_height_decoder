#!/bin/bash

# Train G1 WMP (World Model Planner) — two-stage curriculum
#
# Stage 1 (g1_base.py: training_stage = 1):
#   Flat terrain + full cmd_vel, learn basic walking.
#   bash train.sh
#
# Stage 2 (g1_base.py: training_stage = 2):
#   Stripe terrain + forward-only + plant-then-hold until nearly stopped.
#   bash train.sh --resume --load_run <stage1_run> --checkpoint 15000 --max_iterations=45000
#
# Override any variable below or pass extra args as: bash train.sh --resume

TASK="g1_blind"
NUM_ENVS=4096
MAX_ITER=15000
SEED=1
DEVICE="cuda:0"

echo "Starting training for task: ${TASK}"
python ./legged_gym/scripts/train.py \
    --task=${TASK} \
    --headless \
    --num_envs=${NUM_ENVS} \
    --max_iterations=${MAX_ITER} \
    --seed=${SEED} \
    --sim_device=${DEVICE} \
    --rl_device=${DEVICE} \
    "$@"
