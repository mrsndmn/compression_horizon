# Standard cramming with a shared additive input vector

## Hypothesis

Repeated injection of the same learned vector may increase reconstruction
length without adding trainable scalars. Unlike SPS shared prediction, adding
the vector to ordinary inputs also changes context representations.

## Setup

`AdditiveCrammingTrainer` in `scripts/analyze_standard_additive.py` implements
the requested input sequence: one learned prefix vector followed by ordinary
token embeddings with that same vector added. Training, teacher replay and
greedy generation use the same conditioning function and ordinary causal mask.
The Python `experiment_configs` registry and `resolved_training_args` are the
design authority. They reuse the completed Standard strict-hinge control,
including exact cached texts and the saved initial vector for each text.

Use `--phase dry` to inspect the immutable MLS payloads, `--phase submit` to
submit registered shards, and `--phase merge` after successful completion.
The merge checks source snapshots, settings, raw stages, sample IDs, target IDs,
paired initialization and replay metrics before computing paired comparisons.
Artifacts live under the shared SPS artifact root in `standard_additive_20261025`.

## Results

All 308 repository tests pass; the independent portable adversarial review
returned APPROVE. A real native Standard checkpoint passes BF16 forward/backward
with a finite nonzero vector gradient and frozen model weights. All paired
source-control initializations and exact cached targets load successfully.
Compatibility evidence is saved as `standard_additive_20261025/preflight.json`.

All eight registered shards were submitted from immutable training source
`e200a381abb6f4f7cd7e64bdbed3cd2515f40ec3`:

- `lm-mpi-job-302e982d-7b7d-4ee1-8cbf-932bc818cd3b`
- `lm-mpi-job-fd5adfe2-fda5-4b2d-bf43-33b1cefdfa3b`
- `lm-mpi-job-a5456fb6-0155-4162-93bf-04aff0506a68`
- `lm-mpi-job-1efc8715-3e2c-44cc-afa9-356c1ab3a346`
- `lm-mpi-job-d3657da0-9ca8-4c20-8bdd-885a89893108`
- `lm-mpi-job-9a4f2c0f-6843-4c68-85ab-732684a270b1`
- `lm-mpi-job-dd70d473-e93e-41c2-afcb-9ee603951dd9`
- `lm-mpi-job-c1fe52f1-0103-4b1e-987f-5b40e6c97237`

The submission manifest is saved beside the shards. Arkhip durable wait group
`d0949dd7ebac4fb584146457e71d7e31`, key
`standard-additive-hinge-batch1-8x6-e200a38-20261025`, resolved with
`all_finished`: every job completed without provider polling errors. The merge
validated all eight shards, all 48 unique text IDs, all 158 raw stages, exact
cached targets, paired initializations, source snapshots and replay metrics.

Mean exact teacher-forced horizon falls from 54.8333 to 2.2917 tokens. Mean
initial correct greedy prefix falls from 52.5000 to 2.1250 tokens. All 48 texts
lose teacher-forced length; the paired mean difference is -52.5417 tokens
(text-bootstrap 95% interval -55.4583 to -49.7708). Horizons are two tokens on
35 texts, three on 12, and four on one. Teacher replay is exact on all 48
selected prefixes. Exact greedy counts are 45/48 versus 44/48 in the control,
but concern substantially different selected lengths and do not indicate
improved generation robustness.

The first optimized vector, first loss and first-stage update count match the
ordinary-memory control exactly on every text. Every text terminates at its
per-stage update cap, without reaching the total budget. Actual cumulative
updates average 1308.8542. Thus this is already a very early optimization
failure under the matched settings, not exhaustion of the total allowance.

Reproduce the aggregate and per-text audit with
`python scripts/analyze_standard_additive_results.py`. It writes
`merged/paired_analysis.json`, `per_sample.csv` and `paired_comparison.png`.
The original merge's `summary.json` independently agrees with the recomputed
paired mean and interval.

An optional `--probe` loads saved failed-stage vectors and the original
BF16-quantized model promoted to CPU FP32, without any optimization or new job.
Its conditions are defined in that analysis script's `experiment_configs`.
At the failed vectors, the mean norm ratio of the learned vector to the mean
target-token embedding is 38.4714. Under the trained additive interface, mean
cosine of history inputs to the prefix vector is 0.999653, and mean relative
variation of normalized final hidden states across prediction positions is
0.010895. Removing the history addition at the same vectors increases hidden
variation to 0.610963, but lowers mean argmax accuracy from 0.271528 to 0.147222.
It therefore changes the representation collapse without rescuing the failed
prefixes. Replacing all inputs by the same vector produces relative variation
of about 1.8e-7 and one distinct top prediction per text.

Probe measurements are in `merged/mechanism_probe.json`; they concern failed
vectors under different arithmetic, not the measured GPU greedy horizons.

## Conclusions

The requested additive intervention strongly hurts achieved reconstruction
in this fixed-checkpoint, paired-initialization comparison. Repeated access
to a shared vector alone is insufficient for the tested protocol.

The terminal probes support a concrete failure mode: a large shared bias makes
ordinary input vectors nearly parallel and prediction-position states similar.
In the constant-input limit, RoPE can change attention weights, but equal value
vectors still produce equal weighted averages; residual and positionwise MLP
updates preserve equality across positions in exact arithmetic. This differs
from SPS's shared-predict interface, where ordinary context states do not
receive the learned vector for a fixed token history.

This does not establish a capacity limit or show that large norm caused the
optimizer to fail. Weakening the history path after fitting did not rescue it.
A next controlled training test could weaken or normalize only that injection,
or separately change learning rate; both require new optimization, keeping
other factors fixed. The current result applies to the matched tested settings.

## Changelog

- 2026-10-25: Implement the user-requested shared additive intervention with
  paired initialization and matched original-budget control.
- 2026-10-25: Pass the full suite, real-checkpoint compatibility preflight and
  independent review; submit immutable shards and register the durable wait.
- 2026-10-25: Merge the completed cohort, audit first-stage equivalence and
  stopping points, save paired plots and run read-only terminal-vector probes.
