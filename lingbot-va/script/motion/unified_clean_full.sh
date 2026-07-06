#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"unified_clean_train"}
PYTHON_BIN=${PYTHON_BIN:-"/root/miniconda3/envs/lingbot/bin/python"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi
    export WANDB_API_KEY="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"
    export WANDB_BASE_URL="https://api.wandb.ai"
    export WANDB_TEAM_NAME="gotobcn-the-university-of-adelaide"
    export WANDB_PROJECT="lingbot-ta"

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
    echo "PYTHON_BIN does not exist or is not executable: ${PYTHON_BIN}" >&2
    exit 1
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

# printf "python -m torch.distributed.run --nproc_per_node=%s --local-ranks-filter=%s --master_port %s --tee 3 -m wan_va.train --config-name %s %s" "$num_gpu" "$log_rank" "$master_port" "$config_name" "$overrides"

# python -m wan_va.train --config-name ${config_name} $overrides

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides

