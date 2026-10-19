#!/usr/bin/env bash
set -euo pipefail

VARIANT="$1"
SOURCE_DIR="$2"
CACHE_CONFIG="${3:-}"
BASE=/home/niuzifan/h3-5090-bench
EVAL="${MINIMAX_H3_EVAL_ROOT:-$BASE/optimization_eval_offline_fp8_20261016}"
EVAL_CONTAINER_ROOT="/bench/$(basename "$EVAL")"
OUT="$EVAL/results/$VARIANT"
LOG="$EVAL/logs/$VARIANT/server.log"
INSTR="$EVAL/instrument"
MODEL_HOST_ROOT="${MINIMAX_H3_MODEL_HOST_ROOT:-/mnt/data/MiniMax-H3}"
MODEL_DIR="${MINIMAX_H3_MODEL_DIR:-/models/MiniMax-H3/Ref2VA}"
MODEL_CANDIDATE_HOST_ROOT="${MINIMAX_H3_MODEL_CANDIDATE_HOST_ROOT:-}"
MODEL_MOUNT=()
if [[ -n "$MODEL_CANDIDATE_HOST_ROOT" ]]; then
  MODEL_MOUNT=(-v "$MODEL_CANDIDATE_HOST_ROOT:$MODEL_DIR:ro")
fi
GPU_LIST="${MINIMAX_H3_GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${MINIMAX_H3_NUM_GPUS:-8}"
TENSOR_PARALLEL_SIZE="${MINIMAX_H3_TENSOR_PARALLEL_SIZE:-4}"
USP_SIZE="${MINIMAX_H3_USP_SIZE:-2}"
TEXT_ENCODER_TP_SIZE="${MINIMAX_H3_TEXT_ENCODER_TP_SIZE:-1}"
VAE_PATCH_PARALLEL_SIZE="${MINIMAX_H3_VAE_PATCH_PARALLEL_SIZE:-8}"
GPU_COUNT="$(awk -F, '{print NF}' <<< "$GPU_LIST")"
[[ "$GPU_COUNT" == "$NUM_GPUS" ]] || {
  echo "MINIMAX_H3_NUM_GPUS=$NUM_GPUS does not match GPU list count $GPU_COUNT" >&2
  exit 2
}
PORT=18105
CONTAINER="h3-opt-${VARIANT}-$$"
IMAGE=vllm/vllm-omni:nightly
LOGPID=
MEMPID=
GUARDPID=
SAMPLERPID=
CLOCKPID=
HOSTPID=
RESOLUTIONS="${MINIMAX_H3_EVAL_RESOLUTIONS:-480p 768p 2k}"
MIN_AVAILABLE_KIB="${MINIMAX_H3_MIN_MEM_AVAILABLE_KIB:-67108864}"
ULYSSES_MODE="${MINIMAX_H3_ULYSSES_MODE:-strict}"
FORMAL_REPEATS="${MINIMAX_H3_FORMAL_REPEATS:-1}"
WARMUP_REPEATS="${MINIMAX_H3_WARMUP_REPEATS:-2}"
DISABLE_TORCH_PROFILE="${MINIMAX_H3_DISABLE_TORCH_PROFILE:-0}"

[[ "$WARMUP_REPEATS" =~ ^[0-9]+$ ]] || { echo "MINIMAX_H3_WARMUP_REPEATS must be a nonnegative integer" >&2; exit 2; }
[[ "$DISABLE_TORCH_PROFILE" == 0 || "$DISABLE_TORCH_PROFILE" == 1 ]] || {
  echo "MINIMAX_H3_DISABLE_TORCH_PROFILE must be 0 or 1" >&2
  exit 2
}

cleanup() {
  [[ -z "${SAMPLERPID:-}" ]] || kill "$SAMPLERPID" >/dev/null 2>&1 || true
  [[ -z "${CLOCKPID:-}" ]] || kill "$CLOCKPID" >/dev/null 2>&1 || true
  [[ -z "${HOSTPID:-}" ]] || kill "$HOSTPID" >/dev/null 2>&1 || true
  [[ -z "${MEMPID:-}" ]] || kill "$MEMPID" >/dev/null 2>&1 || true
  [[ -z "${GUARDPID:-}" ]] || kill "$GUARDPID" >/dev/null 2>&1 || true
  [[ -z "${LOGPID:-}" ]] || kill "$LOGPID" >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

mkdir -p "$OUT" "$EVAL/logs/$VARIANT" "$BASE/cache"
[[ -d "$SOURCE_DIR/vllm_omni" ]] || { echo "invalid source tree: $SOURCE_DIR" >&2; exit 2; }

cat > "$OUT/config.txt" <<CFG
variant=$VARIANT
source=$SOURCE_DIR
model_host_root=$MODEL_HOST_ROOT (read-only bind at /models/MiniMax-H3)
model_container_dir=$MODEL_DIR
model_candidate_host_root=${MODEL_CANDIDATE_HOST_ROOT:-same as model_host_root}
gpus=$GPU_LIST
num_gpus=$NUM_GPUS
tensor_parallel=$TENSOR_PARALLEL_SIZE
usp=$USP_SIZE
resident_dit_layers=50
cpu_offload=disabled
text_encoder_tp_size=$TEXT_ENCODER_TP_SIZE
vae_patch_parallel_size=$VAE_PATCH_PARALLEL_SIZE
vae_parallel_mode=tile
attention=CUDNN_ATTN
execution=eager
ulysses_mode=$ULYSSES_MODE
frames=124
duration_seconds=5
fps=24
steps=50
seed=1101
warmups=$WARMUP_REPEATS x 2 steps per resolution
resolutions=$RESOLUTIONS
formal_repeats=$FORMAL_REPEATS
host_monitor=cpu_load,process_cpu,runnable_load,mem_available,swap,psi,vmstat_numa_and_reclaim_counters
gpu_monitor=utilization,memory,power,sm_graphics_memory_clocks,temperature,pstate,clock_throttle_reasons
gpu_monitor_interval_seconds=1
direct_dram_bandwidth_counter=unavailable_without_perf_event_access (perf_event_paranoid=4)
min_mem_available_kib=$MIN_AVAILABLE_KIB
reference_image=/home/niuzifan/h3-5090-bench/inputs/reference.png
CFG
docker image inspect --format '{{.Id}}' "$IMAGE" > "$OUT/runtime-image.txt"
date -Is > "$OUT/start.txt"

EXTRA=()
if [[ -n "$CACHE_CONFIG" ]]; then
  # Accept a host-side JSON file for convenience, but pass its contents to
  # vLLM: --cache-config is parsed as a JSON string, not as a path.
  if [[ -f "$CACHE_CONFIG" ]]; then
    CACHE_CONFIG="$(<"$CACHE_CONFIG")"
  fi
  EXTRA+=(--cache-config "$CACHE_CONFIG")
fi

PROFILE_ENV=(-e "MINIMAX_H3_TORCH_PROFILE_DIR=$EVAL_CONTAINER_ROOT/results/$VARIANT/profile")
if [[ "$DISABLE_TORCH_PROFILE" == "1" ]]; then
  PROFILE_ENV=(-e MINIMAX_H3_TORCH_PROFILE_DIR=)
fi
if [[ "${MINIMAX_H3_PROFILE_DLO_LOAD:-0}" == "1" ]]; then
  mkdir -p "$OUT/dlo-load-profile"
  PROFILE_ENV=(
    -e MINIMAX_H3_TORCH_PROFILE_DIR=
    -e "MINIMAX_H3_TORCH_DLO_PROFILE_DIR=$EVAL_CONTAINER_ROOT/results/$VARIANT/dlo-load-profile"
  )
fi
if [[ "${MINIMAX_H3_PROFILE_LOAD_ONLY:-0}" == "1" ]]; then
  mkdir -p "$OUT/load-profile"
  cat > "$OUT/load-profile/config.txt" <<CFG
tool=PyTorch Profiler
capture=MiniMaxH3Transformer.load_weights and pinned DMA staging ranges
profile_memory=false
record_shapes=false
CFG
  PROFILE_ENV=(
    -e MINIMAX_H3_TORCH_PROFILE_DIR=
    -e "MINIMAX_H3_TORCH_LOAD_PROFILE_DIR=$EVAL_CONTAINER_ROOT/results/$VARIANT/load-profile"
    -e MINIMAX_H3_ENABLE_NVTX=1
  )
fi

docker run -d --name "$CONTAINER" --gpus "\"device=$GPU_LIST\"" --network host --ipc=host \
  -e CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
  -e VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=14400 \
  -e VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=1800 \
  -e TZ=UTC \
  -e PYTHONPATH=/instrument:/workspace/vllm-omni \
  -e MINIMAX_H3_DENOISE_MARKER_PATH="$EVAL_CONTAINER_ROOT/results/$VARIANT/denoise-events.jsonl" \
  "${PROFILE_ENV[@]}" \
  -e HF_HOME=/cache/huggingface -e TORCH_HOME=/cache/torch \
  -e XDG_CACHE_HOME=/cache/xdg -e PIP_CACHE_DIR=/cache/pip \
  -v "$SOURCE_DIR:/workspace/vllm-omni:ro" \
  -v "$MODEL_HOST_ROOT:/models/MiniMax-H3:ro" \
  "${MODEL_MOUNT[@]}" \
  -v "$BASE:/bench" -v "$BASE/cache:/cache" -v "$INSTR:/instrument:ro" \
  "$IMAGE" \
  vllm serve "$MODEL_DIR" \
    --omni --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --task-type ref2va --num-gpus "$NUM_GPUS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --usp "$USP_SIZE" --ring 1 \
    --text-encoder-tp-size "$TEXT_ENCODER_TP_SIZE" --vae-patch-parallel-size "$VAE_PATCH_PARALLEL_SIZE" \
    --vae-parallel-mode tile --vae-use-tiling \
    --disable-multithread-weight-load \
    --enable-distributed-layerwise-offload --dlo-no-use-allgather \
    --dlo-resident-layers 50 \
    --enforce-eager --diffusion-attention-backend CUDNN_ATTN \
    --ulysses-mode "$ULYSSES_MODE" \
    --enable-diffusion-pipeline-profiler "${EXTRA[@]}" \
  > "$OUT/container-id.txt"

docker logs -f "$CONTAINER" > "$LOG" 2>&1 & LOGPID=$!
free -b -s 1 > "$OUT/host-memory.txt" 2>&1 & MEMPID=$!
python3 "$INSTR/sample_host_state.py" --output "$OUT/host-state.jsonl" --interval 1 & HOSTPID=$!
(
  while sleep 5; do
    avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if [[ -n "$avail_kb" ]] && (( avail_kb < MIN_AVAILABLE_KIB )); then
      printf 'ABORT_LOW_HOST_MEMORY available_kib=%s time=%s\n' "$avail_kb" "$(date -Is)" >> "$OUT/host-memory-guard.log"
      docker stop --time 10 "$CONTAINER" >/dev/null 2>&1 || true
      exit 0
    fi
  done
) & GUARDPID=$!

READY=0
for _ in $(seq 1 7200); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then READY=1; break; fi
  if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
    tail -200 "$LOG" >&2 || true
    exit 1
  fi
  sleep 5
done
[[ "$READY" == 1 ]] || { tail -200 "$LOG" >&2 || true; exit 1; }
date -Is > "$OUT/ready.txt"

if [[ "${MINIMAX_H3_PROFILE_ONLY:-0}" == "1" ]]; then
  date -Is > "$OUT/end.txt"
  cleanup
  trap - EXIT INT TERM
  exit 0
fi

PROMPT='A quiet cottage in a green meadow at sunset, warm natural light, a slow camera push-in, grass and leaves moving gently in a soft breeze.'
for RES in $RESOLUTIONS; do
  case "$RES" in
    480x320) WIDTH=480; HEIGHT=320 ;;
    480p) WIDTH=832; HEIGHT=480 ;;
    768p) WIDTH=1344; HEIGHT=768 ;;
    2k) WIDTH=2560; HEIGHT=1440 ;;
  esac
  R="$OUT/$RES"
  mkdir -p "$R"
  cat > "$R/config.txt" <<CFG
resolution=$RES
width=$WIDTH
height=$HEIGHT
frames_expected=124
duration_seconds=5
fps=24
steps=50
seed=1101
CFG
  common=(
    -X POST "http://127.0.0.1:$PORT/v1/videos/sync"
    -F "prompt=$PROMPT"
    -F "input_reference=@$BASE/inputs/reference.png;type=image/png"
    -F "width=$WIDTH" -F "height=$HEIGHT" -F 'fps=24'
    -F 'flow_shift=12' -F 'seed=1101'
    -F 'extra_params={"task":"ref2va","duration":5,"audio_flow_shift":3.0}'
  )

  : > "$R/warmup.txt"
  for ((W=1; W<=WARMUP_REPEATS; W++)); do
    if ! curl -sS --max-time 14400 -D "$R/warmup-$W.headers" -o /dev/null \
      "${common[@]}" -F 'num_inference_steps=2' \
      -w 'http_code=%{http_code}\nclient_e2e_s=%{time_total}\n' >> "$R/warmup.txt"; then
      tail -200 "$LOG" >&2 || true
      exit 1
    fi
    STATUS=$(awk '/^HTTP\// {status=$2} END {print status}' "$R/warmup-$W.headers")
    [[ "$STATUS" == 200 ]] || { tail -200 "$LOG" >&2 || true; exit 1; }
  done

  for REP in $(seq 1 "$FORMAL_REPEATS"); do
    SUFFIX=$(printf '%02d' "$REP")
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$R/gpu-samples-$SUFFIX.csv" 2>&1 & SAMPLERPID=$!
    nvidia-smi --query-gpu=timestamp,index,clocks.gr,clocks.mem,pstate,clocks_throttle_reasons.active --format=csv -l 1 > "$R/gpu-state-samples-$SUFFIX.csv" 2>&1 & CLOCKPID=$!
    sleep 2
    if curl -sS --max-time 14400 -D "$R/response-$SUFFIX.headers" -o "$R/output-$SUFFIX.mp4" \
      "${common[@]}" -F 'num_inference_steps=50' \
      -w 'http_code=%{http_code}\nclient_e2e_s=%{time_total}\nsize_bytes=%{size_download}\n' > "$R/timing-$SUFFIX.txt"; then
      RC=0
    else
      RC=$?
    fi
    kill "$SAMPLERPID" >/dev/null 2>&1 || true
    wait "$SAMPLERPID" >/dev/null 2>&1 || true
    SAMPLERPID=
    kill "$CLOCKPID" >/dev/null 2>&1 || true
    wait "$CLOCKPID" >/dev/null 2>&1 || true
    CLOCKPID=
    printf 'curl_exit=%s\n' "$RC" >> "$R/timing-$SUFFIX.txt"
    STATUS=$(awk '/^HTTP\// {status=$2} END {print status}' "$R/response-$SUFFIX.headers")
    printf 'http_status=%s\n' "$STATUS" >> "$R/timing-$SUFFIX.txt"
    [[ "$RC" == 0 && "$STATUS" == 200 ]] || { tail -200 "$LOG" >&2 || true; exit 1; }
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 \
      "$R/output-$SUFFIX.mp4" > "$R/video-$SUFFIX.ffprobe.txt"
    sleep 10
  done
done

date -Is > "$OUT/end.txt"
cleanup
trap - EXIT INT TERM
