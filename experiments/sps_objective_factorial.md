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
Initial provider status is Pending. The full comparison and the short gradient
audit have independent durable Arkhip waits. Source and full job metadata live
beside each output in `job.json`. Independent adversarial review passed and
36 focused tests passed. Executable dry launchers and synthetic audit rendering
also passed.

The preceding rescue is documented in `sps_mechanism_research.md`; it is not a
replacement for this full comparison.

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
loss effects from stopping-criterion effects. Neither comparison isolates learned
weights from architectural differences between the checkpoints.

## Changelog
- 2026-10-22: Added the objective/convergence factorial and gradient audit.

- 2026-10-22: Submitted six progressive jobs and read-only audit `lm-mpi-job-02e4ebd3-839f-4943-aff8-55d865b91988`; registered separate durable watches.
