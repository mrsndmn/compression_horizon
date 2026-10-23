# SPS shared-predict cramming with an independently fixed LR trajectory

## Hypothesis

The shorter horizons in the preceding sharded run may result from slower LR
annealing rather than the larger available update budget. Preserve the original
LR trajectory and hold its endpoint while allowing additional updates.

## Setup

Training uses `PredictEmbeddingTrainer` via
`scripts/analyze_sps_predict_shards.py --suite fixed_schedule`. The sole design
authority is its `fixed_schedule_experiment_configs` registry. It reuses the
matched source cohort, native predict initialization, objective, stage limits,
model, attention mask and existing single-GPU shard/validation/merge flow.

The selected approach pins the scheduler horizon independently of the stopping
budget and clamps the scheduler input thereafter. Stretching the cosine repeats
the prior confound; allowing the old cosine to run beyond its horizon would make
LR rise again. This is a fresh deterministic replay, not optimizer-state resume.

Implementation/verification: add scheduler boundary and CLI regressions first;
add the opt-in scheduler argument and a separate shard suite; verify the original
suite and training paths, commit and independently review; submit immutable
snapshots and register a new durable wait. After all jobs finish, run the same
entrypoint with `--suite fixed_schedule --phase merge`, then compare with both
completed cohorts and plot the results.

## Results

Pending submission and completion. No model result is claimed yet.

## Conclusions

This control isolates the LR trajectory from the larger budget. An unchanged
horizon can still reflect the unchanged per-stage stopping limit; it would not
establish a global capacity bound for SPS.

## Changelog

- 2026-10-23: Implement the user-approved fixed-trajectory control, preserving the
  previous experiment and its artifacts.
