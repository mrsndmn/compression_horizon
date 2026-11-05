# Progressive cramming of SPS-adapted pretrained models

## Hypothesis

The separated SPS prediction stream may provide a useful conditioning interface
after adaptation from pretrained autoregressive models. CE- and KD-adapted
checkpoints may differ in reconstruction at the same optimization budget.

## Setup

Training function: `scripts.pretrained_sps_worker.train_shard`, using
`PretrainedPredictTrainer` for shared prediction cramming and `TokenOnlyTrainer`
for original autoregressive controls. The sole experiment-design registry is
`scripts/pretrained_sps_experiments.py`; the launcher is
`scripts.run_pretrained_sps_cramming`. Models, revisions, objectives, budgets,
data selection, shard boundaries, initialization and compute resources live there.

## Results

Checkpoint metadata confirms two Qwen3-Base variants and two SmolLM2 variants,
rather than four Qwen models. Focused CPU tests pass for both backbones,
gradient/reference parity, context isolation, native AR equivalence, strict
loading, progressive trainer artifacts and submission guards. Both preflight
and main dry payloads pass validation; family-native datasets are prepared.
Real-checkpoint GPU preflight and the main cohorts remain pending. No new
reconstruction horizon is measured yet.

## Conclusions

Compare variants within each tokenizer family. Original autoregressive controls
use a different conditioning site and initialization from SPS; achieved lengths
alone cannot isolate architecture, adaptation or a universal capacity limit.
Historical SmolLM CE results are separate from the new matched protocol.

## Changelog

- 2026-11-05: Complete the interactive requirements interview; verify checkpoint
  sources and start-token metadata; implement pretrained adapters and staged MLS
  launch/preflight routes.
