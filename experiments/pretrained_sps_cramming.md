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
`pretrained-sps-preflight-9603ad849360-20261105`, resolved with both jobs completed. Artifacts
are under the source registry's `ROOT`; each family preflight checked its original
AR and both adapted variants. These preflights qualify the execution interface;
they do not measure reconstruction horizons.

Both real-checkpoint preflights completed successfully. For all six variants,
logits match the pinned upstream reference exactly (maximum error zero); gradient
and independent-current-prediction checks pass their registered tolerances.
Bounded train/save/replay/generation calls completed with frozen model weights.
None of the deliberately short training smokes reached reconstruction convergence;
these calls are interface checks rather than compression-horizon measurements.

The first main wave, original Qwen AR, was submitted from
`bde700240cac62d5b5e3d6fe1691da79930d136d`. The immutable worker archive has the
same code fingerprint as the successful preflights; payload audits verify unique
jobs, disjoint registered offsets and single-GPU resources. Main results remain
pending. Full job identifiers in shard order:

- `lm-mpi-job-499954e3-2c15-4774-8a08-29814c8e5371`.
- `lm-mpi-job-2379e738-ca1e-494c-a3e5-9ec951d1f339`.
- `lm-mpi-job-a313e020-2c20-47a2-81cd-a70696dc91a5`.
- `lm-mpi-job-5c03b484-21b3-457b-a1dc-6d229c01eb44`.
- `lm-mpi-job-ddb4b70b-a0b4-407f-a62b-e6a526714e12`.
- `lm-mpi-job-9980eb60-b1c8-40c6-a765-a540c7a21b31`.
- `lm-mpi-job-01f0bf4b-84ea-4336-a6a8-b25740e1262f`.
- `lm-mpi-job-6e3cc7af-1066-4c1d-9c1c-049620e71ee7`.

Arkhip group `17b4d1437baf49d2b994784f35f02198`, key
`pretrained-sps-qwen-ar-9603ad849360-20261105`, observes this main wave.

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

- 2026-11-05: Audit successful real-checkpoint preflights, submit the original
  Qwen AR main wave and register a new durable group for the complete shard set.
