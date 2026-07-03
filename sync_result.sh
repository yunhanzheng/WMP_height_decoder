#!/bin/bash

# Pull training logs from remote 5090 server to local machine
# Run from your LOCAL machine (e.g. ubuntu22)
#
# If sync fails, SSH to the server and find the real path:
#   ssh yulong@131.159.60.170
#   find ~ -type d -name 'Jul02_20-22-20_*' 2>/dev/null
#   find ~ -path '*/logs/g1_base/*' 2>/dev/null
# Then override: REMOTE_LOGS=/actual/path/to/logs bash sync_result.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REMOTE_USER="${REMOTE_USER:-yulong}"
REMOTE_HOST="${REMOTE_HOST:-131.159.60.170}"
REMOTE_LOGS="${REMOTE_LOGS:-/home/yulong/student_projects/yunhan_zheng_WMP/logs}"
LOCAL_LOGS="${SCRIPT_DIR}/logs"

echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_LOGS}"
echo "Local:  ${LOCAL_LOGS}/"
echo ""

echo "Checking remote (you may be prompted for SSH password)..."
if ! ssh "${REMOTE_USER}@${REMOTE_HOST}" "test -d '${REMOTE_LOGS}' && ls -la '${REMOTE_LOGS}/g1_base' 2>/dev/null || ls -la '${REMOTE_LOGS}'"; then
    echo ""
    echo "ERROR: Remote logs directory not found:"
    echo "  ${REMOTE_LOGS}"
    echo ""
    echo "The path may be under a different user or project location. On the server run:"
    echo "  find ~ -path '*/logs/g1_base/*' 2>/dev/null"
    echo "  ls -la ~/student_projects/"
    echo ""
    echo "Then sync with the correct path, e.g.:"
    echo "  REMOTE_LOGS=/home/USER/.../logs bash sync_result.sh"
    exit 1
fi

mkdir -p "${LOCAL_LOGS}"

echo ""
echo "Syncing..."
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.ipynb_checkpoints' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_LOGS}/" "${LOCAL_LOGS}/"

echo ""
echo "Sync complete! Local contents:"
ls -la "${LOCAL_LOGS}/g1_base" 2>/dev/null || ls -la "${LOCAL_LOGS}"
