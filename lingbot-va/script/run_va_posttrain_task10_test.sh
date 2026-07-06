
#!/usr/bin/bash
set -x
set -e

umask 007

NGPU=${NGPU:-3}
MASTER_PORT=${MASTER_PORT:-29501}
LOG_RANK=${LOG_RANK:-0}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_task10"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

# cmd setting
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHFT_LIGHTHOUSE="${torchft_lighthouse}"

# 3 available
export CUDA_VISIBLE_DEVICES=0,1,2

python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --master_port=${master_port} \
    --tee 3 \
    --local-ranks-filter=${log_rank} \
    -m wan_va.train --config-name ${config_name} ${overrides}