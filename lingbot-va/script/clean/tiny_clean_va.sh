#!/usr/bin/bash

set -x

umask 007

NGPU=${NGPU:-"4"}
MASTER_PORT=${MASTER_PORT:-"29511"}
PORT=${PORT:-"1116"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29520"}
CONFIG_NAME=${CONFIG_NAME:-"clean_va_train"}
PYTHON_BIN=${PYTHON_BIN:-"/root/miniconda3/envs/lingbot/bin/python"}

# 设置要使用的GPU（根据NGPU数量）
export CUDA_VISIBLE_DEVICES=4,5,6,7   # 或者用变量: export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export WANDB_API_KEY="wandb_v1_aJjIp4OZ8Z5nyiluAGhoHlXiFrb"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_TEAM_NAME="gotobcn"
export WANDB_PROJECT="test2"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}  # 修正大小写
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false

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

# 设置环境变量并运行
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHFT_LIGHTHOUSE=${torchft_lighthouse}

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides

# python wan_va/train.py --config-name ${config_name}

# torchrun \
#     --nproc_per_node=${num_gpu} \
#     --master_port ${master_port} \
#     --standalone \
#     -m wan_va.train --config-name ${config_name} $overrides 2>&1 | tee full_error.log