#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

GPU_NUM="${GPU_NUM:-${HOST_GPU_NUM:-8}}"
MASTER_PORT="${MASTER_PORT:-19103}"
MODEL_PATH="${MODEL_PATH:-data/flux2}"
PROMPT_FILE="${PROMPT_FILE:-assets/prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rl_embeddings_flux2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
SAVE_DTYPE="${SAVE_DTYPE:-bf16}"
TEXT_ENCODER_OUT_LAYERS="${TEXT_ENCODER_OUT_LAYERS:-9 18 27}"
FILTER_CHINESE="${FILTER_CHINESE:-1}"

cmd=(
    torchrun --nproc_per_node="$GPU_NUM" --master_port "$MASTER_PORT"
    flowbp/data_preprocess/preprocess_flux2_embedding.py
    --model_path "$MODEL_PATH"
    --prompt_dir "$PROMPT_FILE"
    --output_dir "$OUTPUT_DIR"
    --train_batch_size "$BATCH_SIZE"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --save_dtype "$SAVE_DTYPE"
    --max_samples "$MAX_SAMPLES"
    --text_encoder_out_layers $TEXT_ENCODER_OUT_LAYERS
)

if [[ "$FILTER_CHINESE" == "1" || "$FILTER_CHINESE" == "true" ]]; then
    cmd+=(--filter_chinese)
fi

"${cmd[@]}"
