#!/bin/bash

# Evaluate GO2 Blind model
# WMP (World Model Planner) Evaluation Script

TASK="go2_blind"
HEADLESS="--headless"

echo "Starting evaluation for task: ${TASK}"

python ./legged_gym/scripts/eval.py \
    --task=${TASK} \
    ${HEADLESS}

# To change evaluation parameters:
# Edit the constants at the top of eval.py (lines 259-267):
#   NUM_ENVS      - Number of parallel environments (default: 10)
#   DIFFICULTY    - Terrain difficulty (default: 0.1)
#   VEL_X         - Forward velocity (default: 0.5)
#   VEL_Y         - Lateral velocity (default: 0.0)
#   VEL_YAW       - Yaw velocity (default: 0.0)
#   RANDOMIZE     - Enable domain randomization (default: False)
#   ADD_NOISE     - Add observation noise (default: False)
#   SHOW_ALL      - Show all individual environment rewards (default: True)
