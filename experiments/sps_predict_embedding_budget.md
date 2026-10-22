# Larger optimization budget for shared-predict cramming

## Hypothesis
Most completed margin runs stopped at the total optimization budget. Increasing
that budget may extend the prefixes reconstructed by the same frozen checkpoint
and one shared predict-input vector.

## Setup
Training function: `scripts.analyze_sps_predict_embedding.probe`.
The `budget_experiment_configs` registry selected by `--suite budget` defines
all settings and same-initialization completed predict controls. Runs start
fresh because the earlier artifacts do not include Adam/scheduler state. The
existing LR schedule spans the new total budget; this comparison measures the
whole budget/schedule protocol, not extra updates at an unchanged LR trajectory.
Per-stage stopping, fixed BOS, native attention/generation semantics and the
matched text cohort use the existing implementation.

## Results
Both new jobs were submitted from reviewed snapshot
`db75f0f915890dc92d0b050836142455c2ea98df` and initially reported Pending:

- Native initialization: `lm-mpi-job-742ad753-3f98-4ba1-9a0d-5b647d3540b7`.
- Random initialization: `lm-mpi-job-04a21972-ffc6-47a7-a479-8b9fb8978a9d`.

Artifacts are under `artifacts/sps_cramming/predict_embedding_20261022/`,
with one new registry-named directory per run. Durable Arkhip group
`fefb740d3daa4fd0ba1c98a932411f01` observes both jobs, with idempotency key
`sps-predict-budget30000-db75f0f-20261022`. Results remain pending; after both
complete, run `scripts/analyze_sps_predict_embedding.py --phase summary --suite budget`.

Implementation passed 22 focused tests and independent adversarial review
(APPROVE). Real executable dry CLIs emitted the expected baseline and new job
sets; invalid suite/experiment combinations rejected before training. Review
also verified actual resolved probe settings and paths, completed controls,
default-summary compatibility and isolation of new analysis outputs.

Earlier provider-reported execution times, including
startup/training and final validation but excluding queue time, were 3 h 58 m
17 s for native-initialized margin and 3 h 49 m 26 s for random-initialized margin.
Their queue waits were 41 m 46 s and 50 m 32 s. Each used one A100.
The corresponding CE job took 1 h 34 m 53 s, excluding its queue wait.
Provider timestamps and full job IDs are recorded in
`artifacts/sps_cramming/predict_embedding_20261022/provider_runtimes.json`.

## Conclusions
Compare achieved teacher-forced prefixes and original-precision greedy
generation with the saved same-initialization controls. A remaining per-stage
failure does not establish infeasibility; longer contexts also make each
optimizer step more expensive, so runtime need not scale linearly with budget.

## Changelog
- 2026-10-22: Added fresh larger-budget margin comparisons and recorded actual previous-job runtimes from MLSpace status timestamps.

- 2026-10-22: Verified both suite CLI paths and report compatibility, obtained independent APPROVE, submitted both jobs and registered a new durable wait group.
