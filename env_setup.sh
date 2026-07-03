#!/usr/bin/env bash
# One-time environment setup for yunhan_zheng_WMP (README Requirements).
# Does NOT modify ~/.bashrc — source yunhan_env_setup.sh before training instead.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="yunhan_zheng_wmp"
ISAAC_GYM_SRC="/home/yulong/student_projects/yanbo_wang_WMP/IsaacGym_Preview_4_Package"
ISAAC_GYM_LINK="${PROJECT_ROOT}/IsaacGym_Preview_4_Package"
TORCH_WHL="/home/yulong/torch-wheel-for-isaacgym/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl"

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "Creating conda env '${CONDA_ENV}' from yanbo_wmp (Python 3.8 + PyTorch for 5090)..."
    conda create -n "${CONDA_ENV}" --clone yanbo_wmp -y
fi

conda activate "${CONDA_ENV}"

if [ ! -e "${ISAAC_GYM_LINK}" ]; then
    echo "Linking Isaac Gym Preview 4..."
    ln -s "${ISAAC_GYM_SRC}" "${ISAAC_GYM_LINK}"
fi

echo "Ensuring PyTorch 2.3 (5090-compatible wheel)..."
pip install "${TORCH_WHL}" --force-reinstall

echo "Installing rsl_rl (project-local, no dependency override)..."
pip install -e "${PROJECT_ROOT}" --no-deps

echo "Installing README dependencies..."
pip install "ruamel_yaml==0.17.4" opencv-contrib-python wandb
pip install -r "${PROJECT_ROOT}/requirements.txt"
pip install "typing_extensions>=4.8,<5" "numpy==1.23.5" "tensorboard>=2.14.0"

echo ""
echo "Setup complete. Before training, run:"
echo "  source ${PROJECT_ROOT}/yunhan_env_setup.sh"
