#!/usr/bin/env bash
# Activate project environment. Source this file — do NOT add to ~/.bashrc.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="yunhan_zheng_wmp"

if command -v conda >/dev/null 2>&1; then
    if ! conda activate "${CONDA_ENV}" >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
    fi
fi

export LD_LIBRARY_PATH="/home/yulong/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ -f "${PROJECT_ROOT}/.wandb_key" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "${PROJECT_ROOT}/.wandb_key")"
fi

cd "${PROJECT_ROOT}"

echo "Environment ready:"
echo "  CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "  PYTHON=$(command -v python)"
echo "  PWD=${PWD}"
