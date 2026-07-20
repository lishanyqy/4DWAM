#!/usr/bin/env bash
# Launch LingBot-VA / 4DWAM WebSocket inference server for LIFT2 real robot.
# Default port 7777 matches client/launch_profiles.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINGBOT_VA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT="${PORT:-7777}"
MASTER_PORT="${MASTER_PORT:-29501}"
NGPU="${NGPU:-1}"
CONFIG_NAME="${CONFIG_NAME:-lift2_merged_infer}"
SAVE_ROOT="${SAVE_ROOT:-${LINGBOT_VA_ROOT}/real_robot_server_out}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${SAVE_ROOT}"

echo "=========================================="
echo " real_robot VA server"
echo " root:        ${LINGBOT_VA_ROOT}"
echo " config:      ${CONFIG_NAME}"
echo " port:        ${PORT}"
echo " nproc:       ${NGPU}"
echo " save_root:   ${SAVE_ROOT}"
echo "=========================================="
echo " Ensure checkpoint transformer/config.json has attn_mode=torch|flashattn"
echo " Override weights: LIFT2_VA_CHECKPOINT / LIFT2_4DWAM_CHECKPOINT"
echo "=========================================="

cd "${LINGBOT_VA_ROOT}"
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv --directory "${LINGBOT_VA_ROOT}" run)
else
  RUNNER=()
fi

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"${RUNNER[@]}" python -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --master_port="${MASTER_PORT}" \
  --tee 3 \
  -m wan_va.wan_va_server \
  --config-name "${CONFIG_NAME}" \
  --port "${PORT}" \
  --save_root "${SAVE_ROOT}" \
  "$@"
