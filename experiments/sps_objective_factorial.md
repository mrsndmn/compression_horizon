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
Pending submission and completion. The preceding rescue is documented in
`sps_mechanism_research.md`; it is not a replacement for this full comparison.

## Conclusions
A single failed progressive stage is not a capacity certificate. The comparison
must report full horizons and original-precision free generation, and distinguish
loss effects from stopping-criterion effects. Neither comparison isolates learned
weights from architectural differences between the checkpoints.

## Changelog
- 2026-10-22: Added the objective/convergence factorial and gradient audit.
