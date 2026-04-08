#!/bin/bash

source /root/bayes-tmp/yulong/envs/WMP/bin/activate
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

bash "$(dirname "$0")/train.sh"
