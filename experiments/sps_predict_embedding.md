# Cramming through a shared SPS prediction embedding

## Hypothesis
Replacing the learned ordinary memory token with an existing fixed BOS and
optimizing one shared prediction-input vector may make SPS reconstruction
constraints easier to satisfy. At zero prediction window, ordinary context
states remain independent of this vector, so this removes the context-mediated
control route and isolates conditional readout adaptation.

## Setup
Training function: `scripts.analyze_sps_predict_embedding.probe`, specialized
`PredictEmbeddingTrainer`. Its `experiment_configs` registry and imported
training settings define all hyperparameters, checkpoint, initializations,
cohort, budget and matched saved memory controls. The experiment changes input
predict embeddings; the tied output-head row is frozen. No vocabulary is added.
The BOS is resolved from the tokenizer and verified against the checkpoint.

The native initialization starts from the model's predict row; the random
control retains the earlier memory initialization. Both layouts have one learned
hidden-size vector per text. BOS is not counted as a reconstructed target token.
Training and generation use the same shared vector, ordinary history and native
zero-window attention. Specialized artifacts label the learned site explicitly.

## Results
Submitted from snapshot `9d20b38a8ab842e2dcb97df21cff373c8b99f54a`.
All three jobs initially report Pending. A durable Arkhip wait observes the group.
Implementation passed 48 focused tests and independent adversarial review.
The real BF16 checkpoint passed a short train/save/replay/generation smoke;
this smoke is not a capacity measurement. Executable dry payloads and isolated
staged-checkout path resolution were verified. Baseline settings and availability
were independently checked.

Full reconstruction lengths and original-precision greedy generation remain
pending. After completion run `scripts/analyze_sps_predict_embedding.py --phase summary`.

Submitted jobs:

- `predict_ce_native`: `lm-mpi-job-f933dc8a-58dd-49e3-a511-48b1c7875b07`.
- `predict_hinge_native`: `lm-mpi-job-425f508b-7c9b-451d-93a0-79d491821122`.
- `predict_hinge_random`: `lm-mpi-job-caff9877-72a5-4bee-9224-3e1e7cb53103`.

## Conclusions
A gain would support prediction-readout adaptation as an alternative control
site. A loss would show the usefulness of the ordinary-context control route,
subject to optimization and initialization dependence. Neither outcome alone
establishes a global capacity limit.

## Changelog
- 2026-10-22: Designed the fixed-BOS/shared-predict comparison with matched per-text parameter count.

- 2026-10-22: Submitted all three comparisons from a reviewed snapshot; registered durable observation.
