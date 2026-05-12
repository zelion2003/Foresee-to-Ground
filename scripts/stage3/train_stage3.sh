#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
LR_STAGE1_MULT="${LR_STAGE1_MULT:-0.1}"
LAMBDA_PROP="${LAMBDA_PROP:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
GRAD_ACC="${GRAD_ACC:-8}"
MAX_STEPS="${MAX_STEPS:-}"

DEBUG_GEN_EVERY="${DEBUG_GEN_EVERY:-10}"
DEBUG_GEN_UNTIL="${DEBUG_GEN_UNTIL:-20}"
DEBUG_GEN_MAX_NEW_TOKENS="${DEBUG_GEN_MAX_NEW_TOKENS:-96}"
DEBUG_GEN_NUM_SAMPLES="${DEBUG_GEN_NUM_SAMPLES:-1}"
DEBUG_PRINT_PROMPT="${DEBUG_PRINT_PROMPT:-1}"
DEBUG_PROMPT_CHARS="${DEBUG_PROMPT_CHARS:-800}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/annotations/stage3_ft.json}"
STAGE2_CKPT="${STAGE2_CKPT:-${ROOT_DIR}/outputs/stage2/stage2_vtg_step18000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/stage3}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
FPS="${FPS:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-48}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/scripts/zero2.json}"

if [[ -z "${VIDEO_ROOTS_JSON:-}" ]]; then
  VIDEO_ROOTS_JSON="{\"anet\":\"${ROOT_DIR}/data/videos/anet\",\"didemo\":\"${ROOT_DIR}/data/videos/didemo\",\"internvid\":\"${ROOT_DIR}/data/videos/internvid\"}"
fi

EXTRA_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
  EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ "${DEBUG_GEN_EVERY}" -gt 0 ]]; then
  EXTRA_ARGS+=(--debug_generate_every "${DEBUG_GEN_EVERY}")
  if [[ "${DEBUG_GEN_UNTIL}" -gt 0 ]]; then
    EXTRA_ARGS+=(--debug_generate_until "${DEBUG_GEN_UNTIL}")
  fi
  EXTRA_ARGS+=(--debug_generate_max_new_tokens "${DEBUG_GEN_MAX_NEW_TOKENS}")
  EXTRA_ARGS+=(--debug_generate_num_samples "${DEBUG_GEN_NUM_SAMPLES}")
  EXTRA_ARGS+=(--debug_prompt_chars "${DEBUG_PROMPT_CHARS}")
  if [[ "${DEBUG_PRINT_PROMPT}" -eq 1 ]]; then
    EXTRA_ARGS+=(--debug_print_prompt)
  fi
fi

deepspeed --num_gpus "${GPUS_PER_NODE}" "${ROOT_DIR}/stage3/train_stage3.py" \
  --data_path "${DATA_PATH}" \
  --stage2_ckpt "${STAGE2_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --lr_stage1_mult "${LR_STAGE1_MULT}" \
  --lambda_prop "${LAMBDA_PROP}" \
  --log_interval "${LOG_INTERVAL}" \
  --fps "${FPS}" \
  --max_frames "${MAX_FRAMES}" \
  --model_id "${MODEL_ID}" \
  --video_roots "${VIDEO_ROOTS_JSON}" \
  --grad_acc_steps "${GRAD_ACC}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  "${EXTRA_ARGS[@]}"
