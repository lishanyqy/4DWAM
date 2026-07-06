#!/usr/bin/bash
set -x
set -e

umask 007

NGPU=${NGPU:-1}
MASTER_PORT=${MASTER_PORT:-29501}
LOG_RANK=${LOG_RANK:-0}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_task10"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}"
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

python -m wan_va.train --config-name ${CONFIG_NAME} ${overrides}