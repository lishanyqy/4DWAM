#!/usr/bin/env bash
# Multi-GPU launch wrapper (same process group as single-GPU server).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NGPU="${NGPU:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PORT="${PORT:-7777}"
export CONFIG_NAME="${CONFIG_NAME:-lift2_merged_infer}"

bash "${SCRIPT_DIR}/launch_server.sh" "$@"
