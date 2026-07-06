START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
LOG_DIR='./logs'
mkdir -p $LOG_DIR

save_root='/cpfs01/projects-HDD/cfff-377aad6b032c_HDD/chenshuai/wenxuan/visualization/mta/visualization'
mkdir -p $save_root

batch_time=$(date +%Y%m%d_%H%M%S)


for i in {0..2}; do  
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    echo "[Task ${j}] GPU: ${i} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=$i  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name robotwin_infer_task10 \
        --save_root $save_root \
        --port $CURRENT_PORT  > $LOG_FILE 2>&1 &
    sleep 2;
done

echo "All 8 instances have been launched in the background."
wait
