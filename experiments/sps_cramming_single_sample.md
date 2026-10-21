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
Submitted; both jobs initially Pending. Training code snapshot:
`e95b204e0bddf65a0d82953e8140c69d64a3e083`.

- Standard: `lm-mpi-job-1891c368-372c-4e84-a617-db1a87406843`;
  artifacts: `artifacts/sps_cramming/standard_pilot_1b_batch1_a5db9c4c6583`.
- SPS: `lm-mpi-job-d7fe44a6-3db5-47b6-a35d-511b2f7f5a43`;
  artifacts: `artifacts/sps_cramming/sps_w0_pilot_1b_batch1_48d1890a7c90`.

Independent adversarial review approved the implementation; focused checks passed.
A separate durable Arkhip watch observes both jobs. Results pending completion.

## Conclusions
Compare both models with the earlier runs on matched samples. Do not assume
that changing the batch shape eliminates every finite-precision boundary flip.

## Changelog
- 2026-10-21: Added paired remeasurement and explicit post-training validation.

- 2026-10-21: Submitted both checkpoints and registered their completion watch.
