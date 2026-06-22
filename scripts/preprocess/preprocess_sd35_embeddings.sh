#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

GPU_NUM="${GPU_NUM:-${HOST_GPU_NUM:-8}}"
MASTER_PORT="${MASTER_PORT:-19104}"
MODEL_PATH="${MODEL_PATH:-data/sd3.5_medium}"
PROMPT_FILE="${PROMPT_FILE:-assets/prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/sd35_rl_embeddings}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
MAX_ITEMS="${MAX_ITEMS:-}"

cmd=(
    torchrun --nproc_per_node="$GPU_NUM" --master_port "$MASTER_PORT"
    flowbp/data_preprocess/preprocess_sd35_embedding.py
    --model_path "$MODEL_PATH"
    --prompt_dir "$PROMPT_FILE"
    --output_dir "$OUTPUT_DIR"
    --train_batch_size "$BATCH_SIZE"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --max_sequence_length "$MAX_SEQUENCE_LENGTH"
)

if [[ -n "$MAX_ITEMS" ]]; then
    cmd+=(--max_items "$MAX_ITEMS")
fi

"${cmd[@]}"
