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

The following commits add, in order: offline FP8 weights,
pinned DMA with DLO, AdaLN schedule caching and pruning, QKV A2A fusion with
Gloo vote deduplication, the AdaLN-QKV tile64/4-warps kernel, Ulysses INT8 QKV
and FP8 attention-output transport, and high-threshold Cache-DiT on the QKV
core. Each result and validation state will be recorded here.

### Validation record

- `bash -n` and local dry runs passed for the default TP2×USP4/DLO50 configuration
  and TP4×USP2/DLO0. A TP2×USP4 configuration with `NUM_GPUS=4` was rejected.
- The selected CLI flags match prior H3 5090 server launches. Per the current
  instruction, this reorganization does not start a new GPU server; the new
  stack will not be described as end-to-end revalidated until such a run is
  explicitly performed.
