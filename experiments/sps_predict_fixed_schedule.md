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

All eight registered shards were submitted successfully and returned `Pending`.
Training source: `cce451fd6625609f1d089de25bb637f479ec827c`.
`pytest -q tests` passed all 270 tests; the independent adversarial review returned
APPROVE after focused executable checks and validation of the actual historical
baseline and shard reports. The committed launcher's dry payloads and the actual
scheduler values passed the saved preflight audit.

Full job identifiers, in shard order:

- `lm-mpi-job-f7fb2d76-4640-43c5-8cae-59ecd9b3583c`
- `lm-mpi-job-0371a82b-788f-4eb8-b981-6ccfcbe84171`
- `lm-mpi-job-33169f33-a59d-4a2c-94cb-0efebf662b89`
- `lm-mpi-job-ca1fff5a-ba93-4d87-89b7-cecff6622827`
- `lm-mpi-job-8e4ad9fb-76a4-4272-906e-a5870e9342e9`
- `lm-mpi-job-ea20c30e-e642-4892-a989-19fa08205628`
- `lm-mpi-job-b0cc259e-7f03-4465-83fa-3a20be288eff`
- `lm-mpi-job-e535f027-ef62-4be5-8931-ed5e1a6fae8d`

Artifacts and submission/preflight manifests live under
`artifacts/sps_cramming/predict_native_hinge_fixed_schedule_20261023/`.
Arkhip durable group `efcd96f731e547c5a742a29595ef7b05`, key
`sps-native-hinge-fixed-schedule30000-8x6-cce451f-20261023`, watches all submitted
jobs. No model result is claimed before successful training and validation.

On its completion outcome, merge with
`python scripts/analyze_sps_predict_shards.py --suite fixed_schedule --phase merge`.
Audit the resulting cohort and stopping points, compare with both the original
predict baseline and the completed stretched-schedule merge, and show the plots.
Do not submit the completed previous experiments again.

## Conclusions

This control isolates the LR trajectory from the larger budget. An unchanged
horizon can still reflect the unchanged per-stage stopping limit; it would not
establish a global capacity bound for SPS.

## Changelog

- 2026-10-23: Implement the user-approved fixed-trajectory control, preserving the
  previous experiment and its artifacts.
- 2026-10-23: Pass tests and independent review, submit the committed shards and
  register the durable wait for their combined analysis.
