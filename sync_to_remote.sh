#!/bin/bash

# Sync project to remote 5090 server
# Run from your LOCAL machine (e.g. ubuntu22)

REMOTE_USER="yulong"
REMOTE_HOST="131.159.60.170"
REMOTE_PATH="~/student_projects/yunhan_zheng_WMP"
LOCAL_PATH="${HOME}/student_projects/yunhan_zheng_WMP"

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
    --exclude 'IsaacGym_Preview_4_Package' \
    "${LOCAL_PATH}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

echo "Sync complete!"
