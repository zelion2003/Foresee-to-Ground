#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_FRAMES="${NUM_FRAMES:-}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACC="${GRAD_ACC:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-0.05}"
FPS="${FPS:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-48}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/annotations/qwen_training_sft.json}"
ANET_ROOT="${ANET_ROOT:-${ROOT_DIR}/data/videos/anet}"
DIDEMO_ROOT="${DIDEMO_ROOT:-${ROOT_DIR}/data/videos/didemo}"
INTERNVID_ROOT="${INTERNVID_ROOT:-${ROOT_DIR}/data/videos/internvid}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/qwen3vl_sft}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/scripts/zero2.json}"

NUM_FRAMES_ARG=()
if [[ -n "${NUM_FRAMES}" ]]; then
  NUM_FRAMES_ARG=(--num_frames "${NUM_FRAMES}")
fi

torchrun --nproc_per_node="${GPUS_PER_NODE}" "${ROOT_DIR}/qwen3vl_sft/train.py" \
  --data_path "${DATA_PATH}" \
  --anet_root "${ANET_ROOT}" \
  --didemo_root "${DIDEMO_ROOT}" \
  --internvid_root "${INTERNVID_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --fps "${FPS}" \
  --max_frames "${MAX_FRAMES}" \
  "${NUM_FRAMES_ARG[@]}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --warmup_ratio "${WARMUP}" \
  --deepspeed "${DEEPSPEED_CONFIG}"
