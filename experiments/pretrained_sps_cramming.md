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
The complete repository suite passes (325 tests), and the fresh independent
adversarial review approves the implementation after fixes to prepared-data
identity/mask validation, source-snapshot guards and global embedding filenames.
Real-checkpoint preflights were submitted from
`4da4f0f1af82698888ba06ecda7fc5a8a282dacd`:

- Qwen: `lm-mpi-job-86fb4d31-952b-4652-8055-5d2289c593ca`.
- SmolLM: `lm-mpi-job-505d934f-4e54-4c5b-84bb-6106265bda6b`.

Arkhip durable group `67041115a5ce4e1c8aca12e6e9d09f39`, key
`pretrained-sps-preflight-9603ad849360-20261105`, observes these jobs. Artifacts
are under the source registry's `ROOT`; each family preflight checks its original
AR and both adapted variants. The main cohorts wait for successful preflight
completion. No new reconstruction horizon is measured yet.

## Conclusions

Compare variants within each tokenizer family. Original autoregressive controls
use a different conditioning site and initialization from SPS; achieved lengths
alone cannot isolate architecture, adaptation or a universal capacity limit.
Historical SmolLM CE results are separate from the new matched protocol.

## Changelog

- 2026-11-05: Complete the interactive requirements interview; verify checkpoint
  sources and start-token metadata; implement pretrained adapters and staged MLS
  launch/preflight routes.
- 2026-11-05: Address independent-review findings, pass focused/full verification,
  preserve the raw cohort while upgrading integrity checks, submit GPU preflights
  and register durable observation.
