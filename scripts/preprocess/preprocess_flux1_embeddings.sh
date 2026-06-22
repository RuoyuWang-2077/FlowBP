#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

GPU_NUM="${GPU_NUM:-${HOST_GPU_NUM:-8}}"
MASTER_PORT="${MASTER_PORT:-19002}"
MODEL_PATH="${MODEL_PATH:-data/flux}"
PROMPT_FILE="${PROMPT_FILE:-assets/prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rl_embeddings}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"

torchrun --nproc_per_node="$GPU_NUM" --master_port "$MASTER_PORT" \
    flowbp/data_preprocess/preprocess_flux_embedding.py \
    --model_path "$MODEL_PATH" \
    --prompt_dir "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --train_batch_size "$BATCH_SIZE" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
