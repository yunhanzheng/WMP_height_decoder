#!/bin/bash

# Train GO2 Blind model
# WMP (World Model Planner) Training Script

TASK="go2_blind"
HEADLESS="--headless"

echo "Starting training for task: ${TASK}"
python ./legged_gym/scripts/train.py --task=${TASK} ${HEADLESS}
