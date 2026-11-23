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

## Conclusions

On this paired cohort fixed-order cramming improved exact blind recovery in
both models. Standard recovered every selected prefix; SPS retained three
counterexamples to a general reconstruction guarantee. The wider softmax
sum resolved all three observed SPS failures on A100. Its replay accuracy
on the rest of the cohort and its throughput remain unmeasured.

## Changelog

- 2026-11-23. Implement and smoke-test matched Standard and SPS cramming.
- 2026-11-24. Audit all 96 selected prefixes and reproduce the three SPS greedy failures.
- 2026-11-24. Isolate the softmax row reduction and verify blind recovery on the three saved failures.
