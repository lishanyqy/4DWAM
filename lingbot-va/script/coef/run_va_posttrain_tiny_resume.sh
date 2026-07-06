#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"2"}
MASTER_PORT=${MASTER_PORT:-"29510"}
PORT=${PORT:-"1116"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29520"}
CONFIG_NAME=${CONFIG_NAME:-"tiny_train"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi
    export WANDB_API_KEY="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"
    export WANDB_BASE_URL="https://api.wandb.ai"
    export WANDB_TEAM_NAME="gotobcn-the-university-of-adelaide"
    export WANDB_PROJECT="lingbot-va"

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
export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
