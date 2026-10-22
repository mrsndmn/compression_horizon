# Reconstruction objective and numerical stopping margin

## Hypothesis
Mean cross entropy keeps rewarding already correct tokens, while exact cramming
requires every target to beat its rivals. A margin hinge may better match this
constraint, and a positive stopping gap may improve BF16 generation reliability.
The effects on the Standard/SPS horizon gap must be measured in both models.

## Setup
Training entry point: `scripts.jobs.run_jobs_sps.main`, objective suite.
Its `objective_experiment_configs` registry is authoritative for models, settings,
completed baseline sources and the additional factorial cells. The shared
progressive trainer and validation pipeline remain in use. The complementary
read-only function `scripts.audit_sps_objective_gradients.probe` compares saved
rescue solutions and correct/wrong-token gradient contributions.

## Results
Submitted from snapshot `4e5726aa444e07b49512cb2af6416a7136b78f3c`.
The gradient audit completed successfully; the full progressive comparison is
still in progress under its independent durable Arkhip wait. Source and full job metadata live
beside each output in `job.json`. Independent adversarial review passed and
36 focused tests passed. Executable dry launchers and synthetic audit rendering
also passed.

The preceding rescue is documented in `sps_mechanism_research.md`; it is not a
replacement for this full comparison.

The completed read-only audit is in
`artifacts/sps_cramming/mechanism_objective_gradients_20261022/`.
Its saved source commit matches the submitted snapshot. All twenty points were
cross-checked against the rescue's recorded minimum margins and exact FP32
teacher-replay token counts. CE correct/wrong contributions sum to total CE;
the maximum relative gradient additivity residual is 0.00000153.

Across the four selected failed prefixes, mean CE at the CE-restart selected
vectors is 1.209 with 1.75 erroneous teacher-forced tokens. Hinge solutions have
zero erroneous tokens and mean CE 3.181; each individual hinge solution has
higher CE than its corresponding erroneous CE solution. The constrained
controller's exact solutions have mean CE 1.379, showing that hinge's large CE
is not necessary for feasibility. These are sampled solutions, not global CE
minima or complete Pareto fronts.

At the selected CE vectors, the cosine between correct- and wrong-token CE
gradients ranges from -0.655 to -0.560. Total gradient norm divided by the sum
of group norms averages 0.522. The negative CE gradient has negative cosine
with the worst-token margin gradient in every case (-0.228 to -0.118): an
infinitesimal plain CE-descent step locally worsens that margin. This is not a
claim about the direction of an actual momentum/preconditioned Adam update.
Hinge-descent cosines at these same vectors are positive (0.868 to 1.000).
The plot `objective_gradient_audit.png` was rendered and visually inspected.

Submitted progressive jobs:
- `standard_pilot_1b_batch1_ce_strict_bb6b7cec8e81`: `lm-mpi-job-253b8328-31b4-4860-89ff-b624ed4849c0`.
- `standard_pilot_1b_batch1_hinge_raw_4fd031e4db53`: `lm-mpi-job-a3515822-881f-45b3-b2eb-74efb9f69fb9`.
- `standard_pilot_1b_batch1_hinge_strict_feb2c4c51ffa`: `lm-mpi-job-147e0ba3-24ec-47e6-ba73-218ba91726d4`.
- `sps_w0_pilot_1b_batch1_ce_strict_d34a35494916`: `lm-mpi-job-fcabbe61-59ea-456b-b5d2-850d0593ef49`.
- `sps_w0_pilot_1b_batch1_hinge_raw_bed545d16418`: `lm-mpi-job-8252518d-0cb4-4f22-8130-84ac307079d5`.
- `sps_w0_pilot_1b_batch1_hinge_strict_893c490bf406`: `lm-mpi-job-4f0d0369-cb5e-45af-afff-db2ac4c14bba`.

## Conclusions
A single failed progressive stage is not a capacity certificate. The comparison
must report full horizons and original-precision free generation, and distinguish
loss effects from stopping-criterion effects. The audit supports an objective mismatch at these selected SPS boundaries:
reducing mean CE and making every token rank first can favor different memory
vectors, and CE gradients from already correct tokens can oppose error repair.
It does not yet establish that SPS has more such conflict than Standard or
quantify how much of the complete horizon gap this explains. Neither comparison
isolates learned weights from architectural differences between checkpoints.

## Changelog
- 2026-10-22: Added the objective/convergence factorial and gradient audit.

- 2026-10-22: Submitted six progressive jobs and read-only audit `lm-mpi-job-02e4ebd3-839f-4943-aff8-55d865b91988`; registered separate durable watches.

- 2026-10-22: Validated completed objective audit against all saved rescue points; documented per-sample objective mismatch and local gradient conflict. Full progressive comparison remains live.
