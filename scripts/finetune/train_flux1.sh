#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/final/flux1/flux1_flowbp_lagrange.yaml}"
MASTER_ADDR="${MASTER_ADDR:-${CHIEF_IP:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-${HOST_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${HOST_GPU_NUM:-8}}"
TORCHRUN_EXTRA_ARGS="${TORCHRUN_EXTRA_ARGS:-}"
TORCHRUN="${TORCHRUN:-torchrun}"

if [[ "$NNODES" == "1" ]]; then
    distributed_args=(--standalone)
else
    distributed_args=(--master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
fi

"$TORCHRUN" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="$NNODES" \
    $TORCHRUN_EXTRA_ARGS "${distributed_args[@]}" \
    flowbp/train_flowbp_flux.py \
    --config "$CONFIG" \
    "$@"
