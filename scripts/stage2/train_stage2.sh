#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

ANNOTATION_PATH="${ANNOTATION_PATH:-${ROOT_DIR}/data/annotations/stage2_vtg_sft.json}"
STAGE1_CKPT="${STAGE1_CKPT:-${ROOT_DIR}/outputs/stage1/stage1_epoch4.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/stage2}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-2}"
FPS="${FPS:-1.0}"

if [[ -z "${VIDEO_ROOTS_JSON:-}" ]]; then
  VIDEO_ROOTS_JSON="{\"anet\":\"${ROOT_DIR}/data/videos/anet\",\"didemo\":\"${ROOT_DIR}/data/videos/didemo\",\"internvid\":\"${ROOT_DIR}/data/videos/internvid\"}"
fi

python "${ROOT_DIR}/stage2/train_stage2.py" \
  --annotation_path "${ANNOTATION_PATH}" \
  --stage1_ckpt "${STAGE1_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_id "${MODEL_ID}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --fps "${FPS}" \
  --video_roots "${VIDEO_ROOTS_JSON}"
