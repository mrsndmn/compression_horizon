# End-to-end native Standard and SPS cramming with fixed-order attention

## Hypothesis

With invariant tokenwise operations and masked attention, a converged vector
should reproduce its prefix during ordinary growing-prefix greedy decoding.
Changing numerical kernels can also change the achieved optimization horizon.

## Setup

The registry and training function are in `scripts.run_native_fixed_cramming`.
The Standard arm learns an ordinary memory and the SPS arm learns its shared
prediction embedding. Both replay the historical initial vectors and native
target IDs. The fixed-order forward consumes the adapter's visibility mask.
The analytical backward uses FP32 softmax-attention gradients. Frozen model
weights are shared by training and replay.

Use the module's `--phase dry`, `submit` and `summary` entry points. Workers
write stage artifacts, reload the selected vector, compare a full pass to
growing true prefixes, and separately decode blindly. The pinned dependencies
on the shared volume are the Thinking Machines source checkout and GPT-2
tokenizer snapshot identified in the registry.

## Results

Both real-checkpoint A100 smokes trained a vector and recovered its selected
prefix after serialization. The CUDA regression checks native visibility,
gradient parity and growing-prefix forward parity, including padding.
MLSpace accepted all eight shards from commit
`1bb5d346ed1ec05dff6c7a4af50cce3516a5f308`. Full job IDs are in
`artifacts/sps_cramming/fixed_attention_cramming_v1/submission_manifest.json`.
Arkhip wait group `6eb89864f71e4a6fab6d91a008bded79` resolved with all eight shards completed.

Both models converged and replayed the selected prefix exactly under teacher
forcing on all 48 texts. The paired results below use each run's own selected
horizon. Exact logits count texts for which every vocabulary logit agreed at
every scored position between full and growing true-prefix forwards.

| Model | Mean TF horizon, old → new | Exact greedy, old → new | Mean correct greedy prefix, old → new | Exact logits, new |
|---|---:|---:|---:|---:|
| Standard | 54.83 → 55.29 | 44/48 → 48/48 | 52.50 → 55.29 | 13/48 |
| SPS | 132.10 → 130.73 | 38/48 → 45/48 | 110.23 → 128.02 | 0/48 |

On SPS, nine formerly inexact samples became exact and two formerly exact
samples became inexact. A fresh replay of the three failures (sample IDs 7,
18 and 46) reproduced the archived greedy token at the first mismatch after
95, 91 and 26 correct tokens respectively. In every case the target beat the
observed competitor by 0.0625 in the full pass and tied it in the short pass.
Comparing the shared rows layer by layer first found a difference at the
attention output of zero-based layers 3, 0 and 6 respectively, with equal
shared Q/K/V rows entering that attention call.
The per-text comparison and failure probes are in `paired_results.csv` and
`sps_first_errors.json` in the run directory.

The softmax denominator diverged in the row-wise FP32 sum of the unnormalized
weights, `tl.sum(p, 1)`. The shared visibility entries were identical. Triton
specialized the reduction layout on alignment, so the full and short kernels
grouped the FP32 additions differently. Changing only the physical row stride
of an otherwise identical mask could switch the BF16 output. Zeroing future
values had no effect on the shared outputs.

To measure the difference without changing the reduction layout, the probe
read the operands of the final division from each compiled PTX variant.
Reassembling the original PTX reproduced the captured attention outputs.
The numerators matched across every shared element of the first affected
attention call. Denominators differed on 653, 553 and 164 head/query pairs
for samples 7, 18 and 46, respectively. After division and BF16 rounding,
one, two and three shared output elements differed. Per-element operands and
quotients are in `sps_softmax_reduction_probe.json` in the run directory.

Widening only the per-block row sum to FP64 and casting its result back to
FP32 recovered all three saved prefixes blindly, with 128/128, 135/135
and 79/79 correct tokens. All vocabulary logits matched bitwise at every
one of the 342 scored positions in full versus growing true-prefix replay.
This intervention used the original saved vectors without retraining.

MLSpace running-to-completed durations and TensorBoard training timestamps
give the following paired A100 costs for the original fixed-order backend
with its FP32 row sum. Each historical job processed the whole cohort.
The new jobs processed it in four parallel single-GPU shards per family.

| Model | Total GPU hours, old → new | New elapsed, first start to last finish | Logged training steps, old → new | Training ms/step, old → new |
|---|---:|---:|---:|---:|
| Standard | 2.18 → 3.66 | 1 h 06 min | 259,545 → 269,826 | 29.76 → 47.74 |
| SPS | 3.97 → 6.55 | 1 h 44 min | 474,784 → 465,070 | 29.79 → 49.75 |

GPU hours include worker startup and validation. Training ms/step divides
the summed time between the first and last logged training losses by the
summed global-step differences. The logged step counts also match the
final per-sample counters. Per-job measurements and provider timestamps
are in `historical_job_timing.csv` in the run directory.

The later FP64 sum has a separate paired A100 measurement using the frozen
models, margin loss, gradient to the compression vector and an optimizer
update. The median change in step time was −0.28% and −0.09% for Standard
at 32 and 64 target tokens, and +0.08% and +0.04% for SPS at 64 and 128.
Those differences were within run-to-run variation. The benchmark source
is in the run directory under `softmax_step_benchmark/`. Individual block
timings are in `softmax_sum_training_step_timing.json`.

The cached replay entry point is `python -m scripts.probe_sps_kv_replay`.
With the FP64 sum, it compared full, growing uncached, and incremental cached
forwards on the same target histories. All 315,657,600 vocabulary logits
matched across the three paths at 6,275 positions. Blind decoding with and
without cache produced identical token sequences on all 48 saved SPS vectors.
Both recovered 38 target texts exactly, including the three prior failures.
The other ten had 12 incorrect teacher-forced top-1 decisions under the wider
sum. The original FP32 training backend still reproduced every target on
full teacher-forced replay. Per-text results are in `sps_kv_replay.json`
and `sps_kv_full_kernel_comparison.json` in the run directory.

## Conclusions

On this paired cohort fixed-order cramming improved exact blind recovery in
both models. Standard recovered every selected prefix; SPS retained three
counterexamples to a general reconstruction guarantee. The wider softmax
sum resolved the three original SPS failures on A100 and changed the full-pass
predictions for ten other archived vectors. Every vector that still passed
full teacher-forced replay under the wider sum recovered its complete target
with KV caching. Fresh training results for both families are reported in [the matched wide-sum experiment](sps_clean_wide_cramming.md).

## Changelog

- 2026-11-23. Implement and smoke-test matched Standard and SPS cramming.
- 2026-11-24. Audit all 96 selected prefixes and reproduce the three SPS greedy failures.
- 2026-11-24. Isolate the softmax row reduction and verify blind recovery on the three saved failures.
- 2026-11-24. Compare historical job and training-step durations and time the FP64 sum on matched steps.
- 2026-11-25. Compare blind and teacher-history cached replay on the entire saved SPS cohort.
