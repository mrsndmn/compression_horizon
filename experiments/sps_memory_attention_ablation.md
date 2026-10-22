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
Completed successfully as `lm-mpi-job-633c13df-b713-4841-9306-76a6bd94175d`,
code snapshot
`07a74820de6b4a33a3828165bfb097de7d16e040`.
Independent adversarial review approved; ten focused tests passed and real saved
inputs were validated. Durable Arkhip observation resolved with completion.
Artifacts: `layer_nll.png`, `layer_accuracy.png`, `position_heatmaps.png`,
`probe.json`, and `summary.json` under the script-defined output directory.

FP32 results below are equal-text mean changes relative to each model's own
unmodified baseline, excluding target token one. Layers are numbered from one.

| Model | Only layer 1: delta NLL | Only layer 2: delta NLL | Only layer 12: delta NLL | Layers 8–12: delta NLL | All layers: delta NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 4.172 | 3.843 | 0.058 | 4.041 | 4.480 |
| SPS w=0 | 3.020 | 0.817 | 0.136 | 3.348 | 4.009 |

Only-layer-one removal reduces accuracy by 79.77 percentage points for Standard
and 68.85 for SPS. Layer two alone reduces it by 73.01 versus 21.73 points.
The SPS response is much smaller after the first layer in the single-layer
curve, with a secondary sensitivity peak at layer eight. Standard remains
strongly affected through the early/middle layers.

Early-layer ablation is not repaired by intact later layers: removing layer one
alone already causes most of the NLL increase seen when all memory-key access
is removed. Late access nevertheless matters jointly, as shown by suffix
ablations. Effects are non-additive; for example, Standard's removal of layers
2–12 hurts more than removal of all layers. Renormalized attention and altered
hidden states make additive layer-contribution interpretations inappropriate.

All per-token NLLs are finite; all mean NLL curves were independently recomputed
from raw records and match the saved summary. The two instances of the full
ablation (last prefix / first suffix) match exactly. Maximum BF16/FP32 difference
across NLL curves is 0.00845 nats/token for Standard and 0.00596 for SPS.
Accuracy differences reach 1.38 and 3.03 percentage points, respectively.
Unmodified mean accuracy on evaluated tokens is 99.85%/99.88% for Standard
(BF16/FP32) and 100%/97.87% for SPS. The evaluation uses common cropped lengths
and FP32 arithmetic variants, so these baselines need not reproduce the original
full-horizon BF16 exact-match counts.

Position heatmaps include the first token, showing Standard's retained direct
memory-readout path. Coverage falls at later target positions: every text
contributes through position eight, while position 26 has only one text.
Colors share a common scale across models; the sparse tail is descriptive.

## Conclusions
Interpret results as effects of removing direct memory-key access at selected
layers; already-transferred memory information may survive in other states.
The first layer's memory access is especially consequential in both models;
Standard has a broader early/middle-layer single-ablation sensitivity than SPS.
Joint late-layer effects demonstrate that small individual effects do not imply
those pathways are dispensable. Similar NLL curves under both arithmetic modes
support the robustness of these intervention effects.

These are sensitivity measurements on already-optimized memories, not evidence
that stronger dependence produces greater capacity. Each memory was optimized
at its model-specific horizon even though evaluation lengths are matched.
Teacher-forced effects do not establish free-generation effects, isolate an
architecture-only cause, or explain the entire difference in optimized horizon.

## Changelog
- 2026-10-22: Added matched-prefix causal memory-attention probe and plots.

- 2026-10-22: Submitted causal evaluation and registered its completion watch.

- 2026-10-22: Verified completed raw metrics, inspected all three figures, and recorded causal findings and interpretation limits.
