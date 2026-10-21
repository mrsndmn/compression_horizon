# Native Standard/SPS learning-rate sweep

## Hypothesis
The two checkpoints may require different memory-optimization step sizes.
Changing the learning rate alone may improve their progressive horizons.

## Setup
Training function: `scripts.activation_distillation`, `ProgressiveCrammingTrainer`.
The registry `lr_sweep_experiment_configs` in `scripts/jobs/run_jobs_sps.py`
defines the paired sweep and references completed baseline runs.
Launch with `--suite lr_sweep`; use `--dry` to inspect commands.
The existing post-training audit reports teacher replay and greedy reconstruction.

## Results
Pending submission.

## Conclusions
Compare every variant against its checkpoint's completed baseline on matched
samples. Separate teacher-forcing horizons from free-generation success.

## Changelog
- 2026-10-21: Added paired learning-rate-only sweep reusing completed baselines.
