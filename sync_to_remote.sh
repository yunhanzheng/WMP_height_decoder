#!/bin/bash

# Sync WMP directory to remote server
# Remote: yulong@131.159.60.170

REMOTE_USER="yulong"
REMOTE_HOST="131.159.60.170"
REMOTE_PATH="~/student_projects/wenshuo_rl/WMP"
LOCAL_PATH="/home/hsu/WMP"

echo "Syncing ${LOCAL_PATH} to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude '.idea' \
    --exclude '.vscode' \
    --exclude '*.egg-info' \
    --exclude 'logs' \
    --exclude 'wandb' \
    --exclude '.ipynb_checkpoints' \
    --exclude 'build' \
    --exclude 'dist' \
    --exclude '*.mkv' \
    "${LOCAL_PATH}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

echo "Sync complete!"
