# MiniMax H3 5090 ordered optimization stack

This series starts from the clean official source at `c2ba8e7`. Each following
optimization is isolated in its own linear commit. Historical evidence is
recorded per optimization; this reorganization does not rerun the GPU
benchmarks, so the exact combined stack has not been revalidated.

## Parallelism and DLO startup

[`run_server.sh`](run_server.sh) exposes the runtime topology through three
environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `MINIMAX_H3_TP_SIZE` | `2` | Tensor-parallel degree |
| `MINIMAX_H3_USP_SIZE` | `4` | Ulysses sequence-parallel degree |
| `MINIMAX_H3_DLO_RESIDENT_LAYERS` | `50` | Number of DiT layers resident on GPU under DLO |

The script derives `NUM_GPUS = TP_SIZE × USP_SIZE`, checks the visible GPU list,
and passes the matching vLLM startup flags. Use `MINIMAX_H3_GPU_LIST` to select
specific device IDs. `MINIMAX_H3_MODEL_PATH` can override the default model
path. Set `MINIMAX_H3_DRY_RUN=1` to print the resolved command without starting
the server.

Example:

```bash
MINIMAX_H3_TP_SIZE=2 \
MINIMAX_H3_USP_SIZE=4 \
MINIMAX_H3_DLO_RESIDENT_LAYERS=50 \
MINIMAX_H3_GPU_LIST=0,1,2,3,4,5,6,7 \
./experiments/minimax_h3_5090_integration/run_server.sh
```

The following commits add, in order: offline FP8 weights, pinned DMA with DLO,
AdaLN schedule caching and pruning, QKV A2A fusion with Gloo vote deduplication,
the AdaLN-QKV tile64/4-warps kernel, Ulysses INT8 QKV and FP8 attention-output
transport, and high-threshold Cache-DiT on the QKV core. Each result and
validation state is recorded here.

## Offline FP8 weights

The [offline exporter](../minimax_h3_offline_fp8/README.md) creates a separate
ModelOpt-compatible checkpoint. It quantizes the four fused linear matrices in
each of H3's 50 DiT blocks (200 matrices total) to E4M3 with one FP32 scale per
output channel; AdaLN and non-DiT weights retain checkpoint precision. The
custom loader applies the QKV row permutation and FC1 gate/up split to scales
as well as weights. Generated shards stay outside Git; on the 5090 host they
belong under `/home/niuzifan/h3-5090-bench/variants/` and are served by setting
`MINIMAX_H3_MODEL_PATH` to the staged model path. Use
`MINIMAX_H3_TEXT_ENCODER_INIT_ON_CPU=1` for the separately validated startup
memory-peak workaround.

Historical 8-GPU, profiler-off results are in
[`RESULTS-20261018.md`](../minimax_h3_offline_fp8/RESULTS-20261018.md). FP8
alone regressed against BF16 in one matched source/config pair by 8.8% E2E and
11.7% denoise. Profiling showed lower FP8 GEMM time but much higher aggregate
all-to-all event time; this points to communication as a likely bottleneck but
does not prove the cause. With the AdaLN cache/pruning configuration enabled in
both arms, the FP8 arm was 14.3% faster E2E and 16.2% faster in denoising than
the BF16 arm, while saving about 4.5 GiB per card during denoising. That is a
combined result, not an isolated FP8 speedup. The comparison also had output
quality differences (SSIM 0.8761, PSNR 28.568 dB against BF16). None of these
historical runs is a revalidation of this new commit chain.

### Validation record

- `bash -n` and local dry runs passed for the default TP2×USP4/DLO50 configuration
  and TP4×USP2/DLO0. A TP2×USP4 configuration with `NUM_GPUS=4` was rejected.
- The offline FP8 exporter and its targeted loader tests were previously
  validated on the recorded experiment source; this branch has not rerun them.
- The selected CLI flags match prior H3 5090 server launches. Per the current
  instruction, this reorganization does not start a new GPU server; the new
  stack will not be described as end-to-end revalidated until such a run is
  explicitly performed.
