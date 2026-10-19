#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MINIMAX_H3_MODEL_PATH:-/models/MiniMax-H3/Ref2VA}}"
TP_SIZE="${MINIMAX_H3_TP_SIZE:-2}"
USP_SIZE="${MINIMAX_H3_USP_SIZE:-4}"
DLO_RESIDENT_LAYERS="${MINIMAX_H3_DLO_RESIDENT_LAYERS:-50}"
TEXT_ENCODER_INIT_ON_CPU="${MINIMAX_H3_TEXT_ENCODER_INIT_ON_CPU:-0}"

for value_name in TP_SIZE USP_SIZE DLO_RESIDENT_LAYERS; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "${value_name} must be a nonnegative integer, got: ${value}" >&2
    exit 2
  }
done
[[ "$TEXT_ENCODER_INIT_ON_CPU" == 0 || "$TEXT_ENCODER_INIT_ON_CPU" == 1 ]] || {
  echo "MINIMAX_H3_TEXT_ENCODER_INIT_ON_CPU must be 0 or 1, got: ${TEXT_ENCODER_INIT_ON_CPU}" >&2
  exit 2
}
(( TP_SIZE > 0 && USP_SIZE > 0 )) || {
  echo "TP_SIZE and USP_SIZE must be positive" >&2
  exit 2
}

EXPECTED_GPUS=$((TP_SIZE * USP_SIZE))
NUM_GPUS="${MINIMAX_H3_NUM_GPUS:-$EXPECTED_GPUS}"
[[ "$NUM_GPUS" =~ ^[0-9]+$ ]] && (( NUM_GPUS == EXPECTED_GPUS )) || {
  echo "NUM_GPUS must equal TP_SIZE × USP_SIZE (${EXPECTED_GPUS}), got: ${NUM_GPUS}" >&2
  exit 2
}

GPU_LIST="${MINIMAX_H3_GPU_LIST:-}"
if [[ -z "$GPU_LIST" ]]; then
  GPU_LIST="$(seq 0 "$((NUM_GPUS - 1))" | paste -sd, -)"
fi
GPU_COUNT="$(awk -F, '{print NF}' <<< "$GPU_LIST")"
[[ "$GPU_COUNT" == "$NUM_GPUS" ]] || {
  echo "GPU_LIST contains ${GPU_COUNT} devices but NUM_GPUS=${NUM_GPUS}" >&2
  exit 2
}
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export MINIMAX_H3_TEXT_ENCODER_INIT_ON_CPU="$TEXT_ENCODER_INIT_ON_CPU"
DRY_RUN="${MINIMAX_H3_DRY_RUN:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || {
  echo "MINIMAX_H3_DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
  exit 2
}

COMMAND=(
  vllm serve "$MODEL_PATH"
  --omni
  --host "${MINIMAX_H3_HOST:-0.0.0.0}"
  --port "${MINIMAX_H3_PORT:-18105}"
  --trust-remote-code
  --task-type ref2va
  --num-gpus "$NUM_GPUS"
  --tensor-parallel-size "$TP_SIZE"
  --usp "$USP_SIZE"
  --ring 1
  --text-encoder-tp-size 1
  --vae-patch-parallel-size "$NUM_GPUS"
  --vae-parallel-mode tile
  --vae-use-tiling
  --disable-multithread-weight-load
  --enable-distributed-layerwise-offload
  --dlo-no-use-allgather
  --dlo-resident-layers "$DLO_RESIDENT_LAYERS"
  --enforce-eager
  --diffusion-attention-backend CUDNN_ATTN
  --ulysses-mode strict
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
