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
Implementation and job submission pending. Full reconstruction lengths and
original-precision greedy generation must both be reported.

## Conclusions
A gain would support prediction-readout adaptation as an alternative control
site. A loss would show the usefulness of the ordinary-context control route,
subject to optimization and initialization dependence. Neither outcome alone
establishes a global capacity limit.

## Changelog
- 2026-10-22: Designed the fixed-BOS/shared-predict comparison with matched per-text parameter count.
