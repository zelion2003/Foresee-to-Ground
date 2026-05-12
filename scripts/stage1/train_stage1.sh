#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

VIDEO_DIR="${VIDEO_DIR:-${ROOT_DIR}/data/stage1/clips}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/stage1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
FPS="${FPS:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

python "${ROOT_DIR}/stage1/train_stage1.py" \
  --video_dir "${VIDEO_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_id "${MODEL_ID}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --fps "${FPS}" \
  --log_interval "${LOG_INTERVAL}"
