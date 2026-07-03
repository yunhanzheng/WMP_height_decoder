#!/usr/bin/env bash
# One-time wandb setup. Does NOT modify ~/.bashrc.
#
# 1. Open https://wandb.ai/authorize and copy your API key
# 2. Run: bash wandb_setup.sh
#    (or paste key when prompted)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_FILE="${PROJECT_ROOT}/.wandb_key"

source "${PROJECT_ROOT}/yunhan_env_setup.sh" >/dev/null

if [ -f "${KEY_FILE}" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "${KEY_FILE}")"
    echo "Using API key from ${KEY_FILE}"
elif [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Using WANDB_API_KEY from environment"
else
    echo "Get your API key from: https://wandb.ai/authorize"
    read -r -s -p "Paste wandb API key: " api_key
    echo
    printf '%s' "${api_key}" > "${KEY_FILE}"
    chmod 600 "${KEY_FILE}"
    export WANDB_API_KEY="${api_key}"
fi

wandb login --relogin "${WANDB_API_KEY}" >/dev/null
python -c "
import wandb
api = wandb.Api()
user = api.viewer
name = getattr(user, 'username', None) or getattr(user, 'entity', str(user))
print(f'Logged in as: {name}')
print(f'Project: yunhan_zheng_WMP')
print(f'URL: https://wandb.ai/')
"
