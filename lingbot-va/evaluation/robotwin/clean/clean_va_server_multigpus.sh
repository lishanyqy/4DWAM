START_PORT=${START_PORT:-29876}
MASTER_PORT=${MASTER_PORT:-29671}
LOG_DIR='./logs'
mkdir -p $LOG_DIR

save_root='/root/logs/robotwin/visualization'
mkdir -p $save_root

batch_time=$(date +%Y%m%d_%H%M%S)


for i in {4..7}; do  
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    echo "[Task ${j}] GPU: ${i} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=$i  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name clean_va_infer \
        --save_root $save_root \
        --port $CURRENT_PORT  > $LOG_FILE 2>&1 &
    sleep 2;
done

echo "All 8 instances have been launched in the background."
wait
