# Standard versus SPS: single-sample remeasurement

## Hypothesis
Removing cross-sample batch-shape changes may alter the measured cramming gap
and improve consistency between teacher-forcing and greedy reconstruction.

## Setup
Training function: `scripts.activation_distillation`, `ProgressiveCrammingTrainer`.
The current `experiment_configs` and settings are in `scripts/jobs/run_jobs_sps.py`.
The post-training audit is `scripts.validation.validate_sps_batch1_run`.
It uses the same effective text subset as the original comparison and scores
teacher-forcing and greedy reconstruction separately from exact cached IDs.

## Results
Both jobs completed, including the post-training audit. Training code snapshot:
`e95b204e0bddf65a0d82953e8140c69d64a3e083`.

- Standard: `lm-mpi-job-1891c368-372c-4e84-a617-db1a87406843`;
  artifacts: `artifacts/sps_cramming/standard_pilot_1b_batch1_a5db9c4c6583`.
- SPS: `lm-mpi-job-d7fe44a6-3db5-47b6-a35d-511b2f7f5a43`;
  artifacts: `artifacts/sps_cramming/sps_w0_pilot_1b_batch1_48d1890a7c90`.

Independent adversarial review approved the implementation; focused checks passed.
The durable Arkhip watch resolved with both jobs completed.

| Model | Earlier mean horizon | New mean horizon | New median horizon | Exact teacher replay | Exact greedy reconstruction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 32.0625 | 32.2292 | 32.5 | 48/48 | 47/48 |
| SPS w=0 | 17.1667 | 17.0000 | 16.0 | 48/48 | 43/48 |

Standard reaches a longer horizon on every matched text. Compared with its own
earlier run, Standard improves on 23 texts, ties on 11, and declines on 14;
SPS improves on 5, ties on 33, and declines on 10. Greedy success is measured
at each sample's selected teacher-forcing horizon; it is not a new search for
the longest greedily reconstructible prefix.

All selected memories replay exactly under teacher forcing. The raw Standard
validation JSON counts only 47 exact replays because its FP32 mean reduction
reports one all-correct sequence as 0.9999999403953552. Recovering the integer
correct-token count from saved accuracy and target length gives 48; rounding
residuals are below 0.001 for every sample. Original run JSON files are preserved;
`artifacts/sps_cramming/batch1_comparison_20261021.json` records the corrected
counts and the derivation. Future audits count integer correct predictions directly.

Standard's greedy failure is sample 7; SPS failures are samples 0, 7, 16, 25, 35.
For SPS sample 0 the final target token is 16938 but greedy decoding chooses
1268, reproducing the previously audited BF16 shape-sensitive mismatch.

## Conclusions
Single-sample training leaves the original large Standard/SPS horizon gap
essentially unchanged. Cross-sample batch size therefore does not explain that
gap in this experiment. Exact teacher forcing still does not guarantee exact
free generation, even with one sample per forward; growing generation context
changes the computation shape. The prior arithmetic audit isolated this effect
for one SPS prefix, but the other greedy failures have not been individually
attributed. These results do not establish an architectural capacity limit.

## Changelog
- 2026-10-21: Added paired remeasurement and explicit post-training validation.

- 2026-10-21: Submitted both checkpoints and registered their completion watch.

- 2026-10-21: Recorded completed matched runs, greedy checks, and integer-corrected teacher replay counts.
