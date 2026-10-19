# MiniMax H3 selective offline FP8 experiment

This experiment starts from the clean `vllm-omni-baseline` snapshot (`c2ba8e7`)
and keeps the generated model beside, rather than over, the original
`Ref2VA` checkpoint.

## Conversion scope

The NVIDIA `Ref2VA/transformer` checkpoint uses native H3 names. Each of its 50
DiT blocks stores Q/K/V in one `attn.qkv_proj` matrix and gate/up in one
`mlp.fc1` matrix. The six NPU Diffusers projections therefore map to four
serialized matrices per block here (200 matrices total), with the same
quantized parameter count. `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and
`mlp.fc2` are quantized; AdaLN, `condition_proj`, token refiner, final layer,
text encoder, and both VAEs stay at checkpoint precision.

For each output row, conversion computes `scale = max(abs(weight_row)) / 448`,
stores `weight / scale` as E4M3, and writes the FP32 row scale as
`<module>.weight_scale`. This matches ModelOpt `FP8_PER_CHANNEL_PER_TOKEN`:
per-channel static FP8 weights and dynamic per-token FP8 activations. Activation
scales are computed at inference, so no `input_scale` tensor is exported. The
converter runs on CPU and does not construct the full model.

The checkpoint metadata marks the transformer as ModelOpt FP8 and tells vLLM
which modules remain BF16. H3's custom loader applies the exact QKV row
permutation and the FC1 gate/up split to both weights and their scale vectors;
without that, scales would be assigned to the wrong output channels.

This is a ModelOpt-compatible H3 export using the documented serialization
contract; it does not call NVIDIA ModelOpt's `mtq.quantize` exporter because
that exporter does not provide a native MiniMax H3 model conversion path.

## Build a candidate model

Run this inside the CPU-only vLLM-Omni container, with the original model
mounted read-only at `/models/MiniMax-H3`, a writable output directory mounted
at `/artifacts`, and this source tree mounted read-only at
`/workspace/vllm-omni`:

```bash
PYTHONPATH=/workspace/vllm-omni python \
  /workspace/vllm-omni/experiments/minimax_h3_offline_fp8/build_candidate.py \
  /models/MiniMax-H3/Ref2VA \
  /artifacts/Ref2VA-FP8-offline-20261016 \
  --num-threads 32
```

The builder refuses to replace an existing candidate, assembles a staging
directory with relative links for shared non-transformer components, writes
new transformer shards and the manifest, then renames the completed directory
into place. The original checkpoint stays untouched. The relative symlinks in
the candidate resolve to `/models/MiniMax-H3/Ref2VA`, so retain the original
model mount when serving the FP8 transformer.

The shared components are relative symlinks back to the original checkpoint;
the new transformer shards are under
`/artifacts/Ref2VA-FP8-offline-20261016/transformer`.

## CPU audit

Run the focused unit tests from this checkout:

```bash
pytest -q tests/diffusion/models/minimax_h3/test_offline_fp8.py
```

Audit every converted matrix and scale on CPU, and compare representative
scales to their source BF16 tensors:

```bash
PYTHONPATH=/workspace/vllm-omni python \
  /workspace/vllm-omni/experiments/minimax_h3_offline_fp8/verify_checkpoint.py \
  /artifacts/Ref2VA-FP8-offline-20261016/transformer \
  /models/MiniMax-H3/Ref2VA/transformer
```

After conversion, check `transformer/offline_fp8_manifest.json` for the 200
matrix count, quantized parameter count, shard sizes, and SHA-256 values.

## Serving and evaluation

The H3 pipeline resolves the model partition from the path basename. When
serving the candidate, bind the host candidate directory to the container path
`/artifacts/Ref2VA`, and serve `/artifacts/Ref2VA`; keep the original model
mounted read-only at `/models/MiniMax-H3` for the shared-component symlinks.
The benchmark runner accepts:

```bash
MINIMAX_H3_MODEL_CANDIDATE_HOST_ROOT=/home/niuzifan/h3-5090-bench/variants/offline-fp8-20261016/artifacts/Ref2VA-FP8-offline-20261016 \
MINIMAX_H3_MODEL_DIR=/artifacts/Ref2VA \
  experiments/minimax_h3_offline_fp8/run_variant.sh fp8-480x320 /path/to/source-copy
```

The actual 2026-10-16 run successfully loaded the candidate and generated a
fixed-seed 480×320 H.264 video on four RTX 5090 cards. A single matched 4-card
BF16/FP8 comparison, profiler summaries, output videos, and the remaining
8-card TP4×USP2 evaluation status are recorded in
[`RESULTS-20261016.md`](RESULTS-20261016.md). The 4-card run confirms format
and generation-path support; it is not a substitute for the target 8-card
performance comparison.
