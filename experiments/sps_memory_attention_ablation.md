# Causal layerwise memory-attention ablation

## Hypothesis
The Standard/SPS cramming gap may involve different layers reading the memory
key or different transfer of memory information into ordinary token states.

## Setup
Diagnostic function: `scripts.analyze_sps_memory_ablation.probe`.
Experiment sources and settings are defined in that script's registry.
Saved optimized memories are frozen; no retraining is performed.
The intervention removes memory-key attention edges and renormalizes remaining
attention. It preserves residual paths, positions, causal structure and SPS
prediction-token visibility. Common per-text target lengths make models'
position comparisons meaningful, but each memory was trained at its own horizon.
Primary curves exclude the first token because Standard reads that token from
the memory residual directly. Position heatmaps retain it for transparency.

## Results
Submitted as `lm-mpi-job-633c13df-b713-4841-9306-76a6bd94175d`
(initial status Pending), code snapshot
`07a74820de6b4a33a3828165bfb097de7d16e040`.
Independent adversarial review approved; ten focused tests passed and real saved
inputs were validated. Durable Arkhip observation is registered.
Artifacts will include layer NLL and accuracy curves and single-layer position
heatmaps under the script-defined output directory. Results pending completion.

## Conclusions
Interpret results as effects of removing direct memory-key access at selected
layers; already-transferred memory information may survive in other states.
Teacher-forced effects do not establish free-generation effects or explain the
entire difference in optimized horizon.

## Changelog
- 2026-10-22: Added matched-prefix causal memory-attention probe and plots.

- 2026-10-22: Submitted causal evaluation and registered its completion watch.
