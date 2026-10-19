#!/usr/bin/env bash
set -euo pipefail

QUALITY="${1:-high}"
case "$QUALITY" in
  high|lossless) ;;
  *)
    echo "quality must be high or lossless, got: ${QUALITY}" >&2
    exit 2
    ;;
esac

: "${MINIMAX_H3_PROMPT:?Set MINIMAX_H3_PROMPT}"
: "${MINIMAX_H3_REFERENCE_IMAGE:?Set MINIMAX_H3_REFERENCE_IMAGE}"

exec curl --fail-with-body -sS --max-time "${MINIMAX_H3_MAX_TIME_SECONDS:-14400}" \
  "${MINIMAX_H3_API_URL:-http://127.0.0.1:18105/v1/videos/sync}" \
  -F "prompt=${MINIMAX_H3_PROMPT}" \
  -F "input_reference=@${MINIMAX_H3_REFERENCE_IMAGE}" \
  -F "width=${MINIMAX_H3_WIDTH:-1280}" \
  -F "height=${MINIMAX_H3_HEIGHT:-720}" \
  -F "num_frames=${MINIMAX_H3_NUM_FRAMES:-80}" \
  -F "fps=${MINIMAX_H3_FPS:-16}" \
  -F "num_inference_steps=${MINIMAX_H3_STEPS:-50}" \
  -F "seed=${MINIMAX_H3_SEED:-1101}" \
  -F "quality=${QUALITY}" \
  -F 'extra_params={"task":"ref2va","duration":5.0}'
