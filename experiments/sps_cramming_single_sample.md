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
Pending submission.

## Conclusions
Compare both models with the earlier runs on matched samples. Do not assume
that changing the batch shape eliminates every finite-precision boundary flip.

## Changelog
- 2026-10-21: Added paired remeasurement and explicit post-training validation.
