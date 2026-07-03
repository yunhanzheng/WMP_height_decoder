#!/usr/bin/env bash
# Start TensorBoard for g1_base training logs.
# Port 6006 is often taken on shared servers; use 6007 by default.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${PROJECT_ROOT}/logs/g1_base"
PORT="${1:-6007}"

source "${PROJECT_ROOT}/yunhan_env_setup.sh" >/dev/null

if [ ! -d "${LOGDIR}" ]; then
    echo "Log directory not found: ${LOGDIR}"
    echo "Start training first: bash train.sh"
    exit 1
fi

echo "Starting TensorBoard on port ${PORT}"
echo "Logdir: ${LOGDIR}"
echo "Open in browser: http://localhost:${PORT}"
echo "Press Ctrl+C to stop."

exec tensorboard --logdir "${LOGDIR}" --port "${PORT}" --bind_all
