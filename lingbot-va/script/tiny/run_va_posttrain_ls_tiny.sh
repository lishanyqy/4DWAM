#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"4"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"tiny_train"}

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
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \

# printf "python -m torch.distributed.run --nproc_per_node=%s --local-ranks-filter=%s --master_port %s --tee 3 -m wan_va.train --config-name %s %s" "$num_gpu" "$log_rank" "$master_port" "$config_name" "$overrides"

# python -m wan_va.train --config-name ${config_name} $overrides

python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
