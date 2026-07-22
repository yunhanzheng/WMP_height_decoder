#!/bin/bash

# Train G1 WMP (World Model Planner)
# Override any variable below or pass extra args as: bash train.sh --resume

TASK="g1_blind"
NUM_ENVS=4096
MAX_ITER=30000
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
