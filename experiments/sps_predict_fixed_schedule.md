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

All eight registered shards completed successfully and were merged. The audit
verified the complete matched cohort without missing or duplicate sample IDs,
all raw saved stages, exact target token IDs, resolved training settings and the
shared training snapshot.
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
`sps-native-hinge-fixed-schedule30000-8x6-cce451f-20261023`, resolved with
`all_finished`; every job was `completed` without provider polling errors.

The completed merge used
`python scripts/analyze_sps_predict_shards.py --suite fixed_schedule --phase merge`.
It contains 6473 saved stages. The additional read-only comparison saves
`merged/three_way_summary.json`, `three_way_per_sample.csv`, `stopping_points.json`,
`three_way_comparison.png` and `schedule_and_stopping.png`. Its exact analysis
source is preserved in `merged/analysis_source.py`; it performs no inference,
optimization or job submission.

| Completed protocol | Mean teacher horizon | Mean initial correct greedy prefix | Exact final greedy prefix |
|---|---:|---:|---:|
| Original shared-predict control | 132.10 | 110.23 | 38/48 |
| Extended budget, stretched LR | 79.42 | 76.13 | 44/48 |
| Extended budget, original LR then endpoint | 133.85 | 121.19 | 41/48 |

Every protocol has exact teacher replay on all 48 selected prefixes. The exact
greedy counts refer to different selected lengths and are not matched-length
robustness measurements. Initial correct greedy prefix is the final vector's
free-generation prefix to its first error, not a maximal greedy horizon.

Against the original control, 33 texts improve in teacher horizon, 15 tie and none
worsen. Per-text gains range from zero to ten tokens; the mean gain is 1.75
(paired text-bootstrap 95% interval 1.19 to 2.40), about 1.3%. The mean initial
greedy-prefix difference is 10.96, but its interval includes zero (-3.42 to 25.35).
Against the stretched-LR protocol, all 48 teacher horizons improve, by 54.44
tokens on average (50.65 to 58.06). These intervals measure text variability at
one checkpoint and seed, not seed uncertainty.

Initial vectors, first-stage optimized vectors and first-stage step counts match
the original control on all 48 texts. All 6342 paired stages ending before the
original budget boundary match exactly in optimized vector, cumulative updates
and convergence. The 15 tied teacher horizons also retain identical selected
vectors and generated token IDs.

The stopping audit changes the interpretation of the available budget. The
original control exhausted its total budget on 47 texts and stopped on a stage
failure on one. In the fixed-schedule extension, all 48 texts stop at a failed
stage after 1000 additional updates; none exhausts the total budget. Final
updates average 11391.65 (range 4784–15082), versus 9891.33 originally. Thus the
available budget grew substantially, but actual used updates increased only
about 15.2%, for a small teacher-horizon gain.

## Conclusions

At the same extended budget and shard design, restoring the original LR
trajectory recovers the loss seen with the stretched schedule. The previous
regression is therefore attributable to that schedule intervention in this
matched comparison, rather than evidence that more available optimization
reduces representational capacity.

The added updates yield only a small teacher-horizon gain; this protocol now
terminates on the per-stage limit at the small endpoint LR. That stopping event
does not prove that a larger per-stage limit would help, or establish a global
SPS capacity bound. A next controlled test could change the per-stage allowance
while retaining the LR trajectory and objective. No new training jobs were
submitted during this result analysis.

## Changelog

- 2026-10-23: Implement the user-approved fixed-trajectory control, preserving the
  previous experiment and its artifacts.
- 2026-10-23: Pass tests and independent review, submit the committed shards and
  register the durable wait for their combined analysis.
- 2026-10-23: Merge the completed cohort, verify saved pre-boundary trajectories
  exactly, compare all three protocols and plot outcomes, LR curves and stopping
  reasons. The fixed schedule recovers the regression with a small extra gain;
  every extended run now terminates on its per-stage cap.
