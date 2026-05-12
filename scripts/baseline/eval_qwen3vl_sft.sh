#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/annotations/timelens/charades_timelens_test.json}"
VIDEO_ROOT="${VIDEO_ROOT:-${ROOT_DIR}/data/videos/timelens_bench_336}"
SAVE_PATH="${SAVE_PATH:-${ROOT_DIR}/outputs/eval/qwen3vl_sft_eval.json}"
MODE="${MODE:-seconds}"
SEC_PER_INDEX="${SEC_PER_INDEX:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

python "${ROOT_DIR}/qwen3vl_sft/eval/run_eval.py" \
  --model_id "${MODEL_ID}" \
  --data_path "${DATA_PATH}" \
  --video_root "${VIDEO_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --mode "${MODE}" \
  --sec_per_index "${SEC_PER_INDEX}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --device_map "${DEVICE_MAP}"

python "${ROOT_DIR}/qwen3vl_sft/eval/score_results.py" \
  --pred_json "${SAVE_PATH}" \
  --mode "${MODE}" \
  --sec_per_index "${SEC_PER_INDEX}"
