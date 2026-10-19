# MiniMax H3 FP8 Communication Optimization on 8x RTX 5090

This report records controlled measurements for Q/K/V all-to-all fusion,
cancellation-related Gloo synchronization, and the advanced Ulysses size cache
on the full-FP8 MiniMax H3 Ref2VA workload.

## Workload and measurement protocol

- Run date: 2026-10-19 UTC.
- Hardware: 8x RTX 5090.
- Model: full-FP8 Ref2VA checkpoint, staged at
  `/home/niuzifan/h3-5090-bench/optimization_eval_h3_commit_series_20261018/models/staged-fp8-allqkv-20261018`.
- Input: 480x320, 124 frames, 24 fps, 50 denoising steps, seed 1101.
- Parallelism: TP=4, USP=2, VAE tile parallelism=8; 50 resident DiT layers,
  CPU offload disabled.
- Attention: CUDNN_ATTN; eager execution; full DLO enabled.
- Each variant used two 2-step warmups and three unprofiled formal requests.
  Profiler traces were collected separately for diagnostics.
- All eight GPUs were checked idle before each run. Formal requests returned
  HTTP 200 and produced 480x320, 124-frame H.264 videos.

The remote harness was
`/home/niuzifan/h3-5090-bench/optimization_eval_h3_commit_series_20261018/run_variant_qkv_eval.sh`.
It starts `vllm serve /models/MiniMax-H3-fp8/Ref2VA` with TP=4, USP=2,
`--enable-distributed-layerwise-offload --dlo-no-use-allgather
--dlo-resident-layers 50`, VAE tile parallelism=8, eager execution, and
CUDNN_ATTN. The harness records the resolved settings in each result's
`config.txt` and the expanded server startup in `logs/<variant>/server.log`.

Each formal request was a multipart `POST /v1/videos/sync` for the prompt
“A quiet cottage in a green meadow at sunset, warm natural light, a slow
camera push-in, grass and leaves moving gently in a soft breeze.” with the
same reference image, `task=ref2va`, `duration=5`, `fps=24`, `flow_shift=12`,
`audio_flow_shift=3`, seed 1101, and 50 steps. The harness's two warmups used
the same request with 2 steps; formal requests were serialized and profiler
disabled.

## Results

| Change | Comparison | Median denoise (ms) | Median E2E (ms) | Output | Decision |
|---|---|---:|---:|---|---|
| Gloo boundary vote deduplication, strict UAA | Parent vs patched, same source/config | 47,839 -> 43,036 (-10.0%) | 55,188 -> 50,301 (-8.9%) | Exact output SHA-256 match | Keep |
| Q/K/V A2A fusion alone | Unfused vs packed fusion | 40,690 -> 44,425 (+9.2%) | 47,831 -> 51,564 (+7.8%) | Exact output match | Keep off without Gloo dedup |
| Gloo dedup + Q/K/V fusion | Strict, fusion off recheck vs on | 42,256 -> 36,940 (-12.6%) | 49,940 -> 44,038 (-11.8%) | Exact output hash match | Keep enabled for this stack |
| UAA size cache | Advanced UAA, cache off vs on | 49,591 -> 47,019 (-5.2%) | 56,923 -> 54,584 (-4.1%) | Exact output SHA-256 match | Keep opt-in; strict remains faster overall |

The Gloo parent/patched runs used the same FP8 checkpoint, image digest
`sha256:3f1af149f06af0b323db4fc4c612adc55e18129b9f25dbc49547264e0dcf4c7e`,
input, random seed, stage layout, and no-profiler measurement path. Parent
denoise samples were 48,387, 47,839, and 47,720 ms; patched samples were
43,036, 42,631, and 43,600 ms. Their output hash was
`93d305b2527b4623b2e1d2f0ca109feb019211d51193f5f80c177f9623086986` in every
request.

### Why the Gloo change helps

The denoise loop checked cancellation both at the start and end of every step.
The end-of-step check and the next step's start check vote at the same model
boundary. Each vote performs a CPU Gloo all-reduce when cancellation scopes are
active. The implementation now keeps one vote before every step and one final
synchronized vote after the last step. For 50 steps this reduces votes from
100 to 51 while preserving cross-rank agreement at every step boundary and
draining device work before returning from the final step.

In the 2-step diagnostic trace, the Gloo collective count fell from 4 to 3.
The measured full-request improvement was consistent in all three paired
samples, and outputs were byte-identical. The Gloo collective belongs to
request-cancellation coordination; it is separate from the NCCL Q/K/V
all-to-all path.

### Q/K/V A2A fusion depends on the Gloo configuration

Without the Gloo change, the packed path issued fewer A2A calls but ran slower:
denoise regressed 9.2%, and the trace showed a slower packed collective path
with increased aggregate NCCL send/receive work. With Gloo boundary votes
deduplicated, a strict-mode 8-GPU run changed the outcome: fusion reduced the
denoise median from 42,256 to 36,940 ms, with exact output hashes. The 2-step
profile reports 201 rather than 401 A2A calls. Its aggregate A2A CUDA time
rises from 212.816 to 417.057 ms, showing that each packed call has more
device work. The profiler trace is diagnostic; unprofiled full-request
measurements decide the net result. In the same strict Gloo-deduplicated
stack, the fusion-on median was 36,940 ms denoise / 44,038 ms E2E; a no-fusion
recheck after it measured 42,256 / 49,940 ms. The two warmed requests in each
variant averaged 35,829 / 43,181 ms with fusion and 41,048 / 48,509 ms without
it, preserving an approximately 11-13% advantage. All three fusion outputs
were byte-identical to the no-fusion output. Thus fusion is a win in the
Gloo-deduplicated strict stack even though it regressed in the earlier
Gloo-duplicated stack.

The likely reason is the interaction between halving per-layer A2A dispatches
and removing the duplicate CPU Gloo votes; this is an inference, since the
profile shows more summed CUDA work in the packed A2A calls. The low-overhead
full-request results confirm the combined win but do not assign all of its
12.6% to either change individually.

The three formal denoise samples were 40,975, 36,940, and 34,718 ms with
fusion, versus 48,843, 39,839, and 42,256 ms in the no-fusion recheck. This
shows that two 2-step warmups do not fully warm the first 50-step request; the
comparison therefore includes both the three-sample medians and the mean of
requests 2-3. The warm-request means are 35,829 vs 41,048 ms for denoise and
43,181 vs 48,509 ms E2E. This is the strongest evidence that the fusion win
persists after the first full request, although a longer-run benchmark would
further tighten the latency estimate.

### UAA size-cache evidence

The size cache applies only to `advanced_uaa`; strict mode bypasses the cached
size exchange. In a two-step diagnostic, scalar NCCL `all_gather` calls fell
from 101 to 2, and measured CUDA time attributed to those gathers fell from
67.828 to 28.145 ms. Scaling only this CUDA time suggests roughly 1 second
saved over 50 steps; the matched no-profiler full-request A/B measured a
2,571 ms median denoise improvement, so host-side collective overhead also
appears to matter. Earlier cache-off timings varied between batches and are
excluded from this matched comparison.

Advanced UAA cache-on outputs matched each other exactly:
`7f1501665e75cd72f54a2e157442f1a640adb266a50568b3a3640606e57bd75e`.
In the matched Gloo-deduplicated comparison, the cache-on samples were 47,731,
47,019, and 39,415 ms for denoise; cache-off samples were 49,147, 49,980,
and 49,591 ms. The medians give a 2,571 ms denoise improvement. E2E medians
improved by 2,339 ms. This is a measurable win within advanced UAA, with
bit-identical output across all six requests. The advanced cache-on median is
10,079 ms slower than the best strict + Gloo-dedup + fusion configuration
(36,940 ms), so the advanced mode should not replace strict mode for this
workload.

Advanced UAA output differs from strict UAA; a strict-vs-advanced decoded
video comparison measured SSIM 0.960621 and mean PSNR 36.924 dB. Therefore the
advanced mode's output-quality delta must be evaluated separately from the
size-cache toggle.

## Code change and validation

The implementation is in `vllm_omni/diffusion/models/minimax_h3/denoise_loop.py`.
The step-execution test asserts one cancellation vote per step boundary plus
the final synchronized vote. UAA size-cache and Q/K/V fusion tests set the
size-cache opt-in explicitly, so they remain deterministic if the default is
disabled.

Validation completed so far:

- CPU/Gloo cancellation, denoise-boundary, and UAA/A2A size-cache tests:
  19 passed.
- Distributed Q/K/V fusion equivalence and size-cache tests: 5 passed.
- `python3 -m compileall -q vllm_omni/diffusion/models/minimax_h3/denoise_loop.py`:
  passed.
- `git diff --check`: passed.
- Matched full-request cache comparison: complete; all six videos have the
  same output hash.
- Matched strict no-fusion recheck after the Q/K/V fusion run: complete.
- Distributed cancellation agreement test: passed in the 19-test run.

The best measured configuration is strict UAA + Gloo boundary deduplication +
`MINIMAX_H3_QKV_A2A_FUSION=1`; UAA size caching is not applicable in strict
mode. Its three-request median denoise is 36,940 ms and E2E is 44,038 ms, with
exact output equality against strict UAA without fusion. The under-10-second
denoise target has not been reached.

## Raw benchmark artifacts

Raw logs, per-request headers, output videos, profiler traces, and GPU samples
are retained under the following root. Each raw variant directory also
contains the exact `config.txt`, request headers, output videos, timings, GPU
telemetry, and diagnostic profile where applicable:

`/home/niuzifan/h3-5090-bench/optimization_eval_h3_commit_series_20261018/results/`

Relevant variant directories:

- `fp8-gloo-dedup-parent-strict-recheck-14c1974`
- `fp8-gloo-dedup-patched-strict-14c1974`
- `fp8-a2a-no-cache-control-ad5890c`
- `fp8-a2a-no-cache-fused-ad5890c`
- `fp8-gloo-dedup-advanced-cache-on-14c1974`
- `fp8-gloo-dedup-advanced-cache-off-14c1974`
- `fp8-gloo-dedup-strict-qkv-fusion-on-14c1974`
- `fp8-gloo-dedup-strict-qkv-fusion-off-recheck-14c1974`
