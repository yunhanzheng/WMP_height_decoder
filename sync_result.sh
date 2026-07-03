#!/bin/bash

# Pull training logs from remote 5090 server to local machine
# Run from your LOCAL machine (e.g. ubuntu22)

REMOTE_USER="yulong"
REMOTE_HOST="131.159.60.170"
REMOTE_PATH="~/student_projects/yunhan_zheng_WMP/logs"
LOCAL_LOGS="logs"

echo "Syncing ${LOCAL_LOGS} from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.ipynb_checkpoints' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" "${LOCAL_LOGS}"

echo "Sync complete!"
