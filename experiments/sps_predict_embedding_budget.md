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
Preparation in progress. Earlier provider-reported execution times, including
startup/training and final validation but excluding queue time, were 3 h 58 m
17 s for native-initialized margin and 3 h 49 m 26 s for random-initialized margin.
Their queue waits were 41 m 46 s and 50 m 32 s. Each used one A100.
The corresponding CE job took 1 h 34 m 53 s, excluding its queue wait.

## Conclusions
Compare achieved teacher-forced prefixes and original-precision greedy
generation with the saved same-initialization controls. A remaining per-stage
failure does not establish infeasibility; longer contexts also make each
optimizer step more expensive, so runtime need not scale linearly with budget.

## Changelog
- 2026-10-22: Added fresh larger-budget margin comparisons and recorded actual previous-job runtimes from MLSpace status timestamps.
