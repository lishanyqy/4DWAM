#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"train_4dwam"}
# PYTHON_BIN=${PYTHON_BIN:-"${HOME}/miniconda3/envs/lingbot/bin/python"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi
    export WANDB_API_KEY="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"
    export WANDB_BASE_URL="https://api.wandb.ai"
    export WANDB_TEAM_NAME="infinity4b"
    export WANDB_PROJECT="4DWAM"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "PYTHON_BIN does not exist or is not executable: ${PYTHON_BIN}; falling back to default python." >&2
    PYTHON_BIN=$(command -v python || command -v python3)
    if [ -z "${PYTHON_BIN}" ]; then
        echo "Could not find a default python. Please activate the correct env or set PYTHON_BIN." >&2
        exit 1
    fi
fi

if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
    echo "Torch is not available in ${PYTHON_BIN}. Please activate the correct env or set PYTHON_BIN." >&2
    exit 1
fi

visible_gpu_count=$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")
if [ "${visible_gpu_count}" -lt "${num_gpu}" ]; then
    echo "Requested NGPU=${num_gpu}, but only ${visible_gpu_count} GPUs are visible to ${PYTHON_BIN}." >&2
    echo "Check CUDA_VISIBLE_DEVICES, driver state, and whether this node really has 8 visible GPUs." >&2
    exit 1
fi

echo "Using PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Visible GPU count=${visible_gpu_count}, requested=${num_gpu}"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \


"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
