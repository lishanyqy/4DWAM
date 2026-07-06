START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

save_root='visualization/'
mkdir -p $save_root

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
    --nproc_per_node 2 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin_infer_ls \
    --port $START_PORT \
    --save_root $save_root
