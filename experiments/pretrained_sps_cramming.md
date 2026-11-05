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
Completed model cohorts are audited by `scripts.analyze_pretrained_sps.audit_experiment`.

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
jobs, disjoint registered offsets and single-GPU resources. The wave subsequently completed. Full job identifiers in shard order:

- `lm-mpi-job-499954e3-2c15-4774-8a08-29814c8e5371`.
- `lm-mpi-job-2379e738-ca1e-494c-a3e5-9ec951d1f339`.
- `lm-mpi-job-a313e020-2c20-47a2-81cd-a70696dc91a5`.
- `lm-mpi-job-5c03b484-21b3-457b-a1dc-6d229c01eb44`.
- `lm-mpi-job-ddb4b70b-a0b4-407f-a62b-e6a526714e12`.
- `lm-mpi-job-9980eb60-b1c8-40c6-a765-a540c7a21b31`.
- `lm-mpi-job-01f0bf4b-84ea-4336-a6a8-b25740e1262f`.
- `lm-mpi-job-6e3cc7af-1066-4c1d-9c1c-049620e71ee7`.

Arkhip group `17b4d1437baf49d2b994784f35f02198`, key
`pretrained-sps-qwen-ar-9603ad849360-20261105`, resolved with all jobs completed.


### Completed original Qwen AR cohort

The reproducible raw-stage audit is
`python -m scripts.analyze_pretrained_sps --experiment qwen_ar`.
Its JSON/CSV outputs are under `ROOT/analysis/9603ad849360/qwen_ar/` and are
regenerated from the complete original shards, native target cache and worker
provenance. It reads no model weights and performs no optimization or inference.

All 48 texts are present once. Mean teacher-forced horizon is 295.0625 tokens
(median 302.5). Every selected prefix replays exactly under teacher forcing.
Mean initial correct greedy prefix is 168.6667 tokens; exact generation of the
selected prefix succeeds on 15/48 texts. These generation values refer to one
selected vector per text and are not a search for a maximal greedy horizon.
All 48 texts terminate at the total update budget; no capacity ceiling is
established. Worker times range from 5524.35 to 6035.49 seconds per shard;
they exclude scheduler waiting and container startup.

The completed Qwen SPS adapted with CE wave was submitted from
`d30039f3c7ff19f1fd2b829b3f3c91c5e4732980`. Its reconstruction objective is still
the registered hinge; CE names the upstream adaptation recipe. Job identifiers:

- `lm-mpi-job-e6fd1595-3e75-460f-a773-f753023108df`.
- `lm-mpi-job-39b2deb4-192a-4bcf-a1fb-2b82460cfb77`.
- `lm-mpi-job-87ecdfd0-7c83-4805-93aa-265896ea6cd9`.
- `lm-mpi-job-2a410d40-0a0b-441c-8175-983340371d02`.
- `lm-mpi-job-fa472ff8-e74f-4fbc-83fd-26b21887acfd`.
- `lm-mpi-job-b1744259-0113-45ad-a020-1fa9db8c5d38`.
- `lm-mpi-job-14acc8b7-94e8-4c5f-89b2-cbb1c5ffc0da`.
- `lm-mpi-job-efdb9585-ad24-4f6b-9ffb-8dc2b4453cf6`.

Arkhip group `a5f9a4cc7568462c8450dea4bbc7f477`, key
`pretrained-sps-qwen-ce-9603ad849360-20261105`, resolved with all jobs completed.

### Completed Qwen SPS-CE cohort

`python -m scripts.analyze_pretrained_sps --experiment qwen_ce` audits all 48
texts and writes the external JSON/CSV alongside the AR audit. Their ordered
raw-document hashes match exactly. Both cohorts use the same native tokenizer.

Mean teacher-forced horizon is 280.0208 tokens (median 279). All 48 selected
prefixes replay exactly under teacher forcing. Mean initial correct greedy
prefix is 106.9583 tokens; exact generation of the selected prefix succeeds on
5/48 texts. Every text exhausts the total update budget. Worker times range from
6085.51 to 6508.34 seconds per shard, excluding scheduler waiting and startup.

At the registered budget, CE-adapted SPS therefore achieves 15.0417 fewer
teacher-forced tokens on average than original Qwen AR (5.10% lower), and its
initial correct greedy prefix is 61.7083 tokens shorter (36.59% lower). The latter
scores generation from the selected teacher-forced vector; it is not a search
over all saved vectors for a maximal greedy horizon. No capacity ceiling or
cause of the gap is established by these budget-limited measurements. The
complete SmolLM comparison is pending.

### Completed Qwen SPS-KD cohort

The Qwen SPS-KD wave was submitted from immutable source snapshot
`545de7f9d3a399eb1b9c864197d3aa373b890642`. The training fingerprint is unchanged;
archived code, shard payloads and unique complete job identifiers were verified.
As with CE, KD names the upstream adaptation recipe; cramming uses the registered
reconstruction objective. Full identifiers in shard order:

- `lm-mpi-job-1ac096ce-c31d-46eb-b115-1292896d9a35`.
- `lm-mpi-job-3b7052a5-4f73-4334-bf0d-baa279b81406`.
- `lm-mpi-job-aae1a841-5915-4136-9a6c-fd718f1e7735`.
- `lm-mpi-job-eed0aa92-2a7c-4546-84a9-06077f4058d6`.
- `lm-mpi-job-a4eea671-b642-4144-bc3e-7192ec1f813c`.
- `lm-mpi-job-afa48859-262a-4836-948c-10a4fb7188d0`.
- `lm-mpi-job-96d750ec-4260-4310-a3da-e03651a9cea0`.
- `lm-mpi-job-b9086c1c-2fac-4752-8226-d34701196517`.

Arkhip group `4f8b9d015b874a51bc106086f8f5a4c6`, key
`pretrained-sps-qwen-kd-9603ad849360-20261105`, resolved with all jobs completed.
Its automatic continuation failed during event delivery with
`workspace routing discovery timed out`; this was a delivery error rather than
a failed training job. Processing resumed following the user's group-status check.

`python -m scripts.analyze_pretrained_sps --experiment qwen_kd` passes for the
complete cohort. All 48 sample identities and ordered document hashes match
both other Qwen variants. Mean teacher-forced horizon is 275.1042 tokens (median
273), with exact teacher-forced replay for all selected prefixes. Mean initial
correct greedy prefix is 171.75 tokens, and exact greedy generation succeeds on
16/48 selected prefixes. Every text exhausts the total update budget. Worker
times range from 6299.28 to 6644.08 seconds per shard, excluding scheduling and
container startup.

KD has a lower mean teacher-forced horizon than both original AR and SPS-CE,
while its mean initial greedy prefix is close to original AR and above SPS-CE.
The different rankings reinforce that teacher-forced horizon and free generation
from the selected vector are distinct measurements. These are descriptive
sample results and do not establish a statistically reliable advantage for KD
over AR, a capacity ceiling, or the causal source of differences.

### Active original SmolLM AR wave

The source-registry launcher submitted the next wave from immutable snapshot
`5cf15914950fa9bb349f7bba34080724f720c408`. Archived training fingerprints and all
shard payloads were verified. Full unique job identifiers in shard order:

- `lm-mpi-job-3cc121d3-68e3-4767-a880-66fc3f0cf4ca`.
- `lm-mpi-job-fde88bb5-a046-4300-ad0c-351cccdfc415`.
- `lm-mpi-job-e1d60b8e-8a7d-4da7-8d4a-b69bf53fb526`.
- `lm-mpi-job-17ea846b-657f-4624-b2d6-41d0d87270cf`.
- `lm-mpi-job-f2b4ab1c-d175-4d0e-9930-cd15d7a9c0f8`.
- `lm-mpi-job-b42f44cd-be5a-461c-b31e-7a9b03cddf91`.
- `lm-mpi-job-be59b587-6c9c-4fb4-9f67-07003217c979`.
- `lm-mpi-job-245742c7-30d9-4efc-aa61-b77f2d6adb1e`.

New durable group `c6400159d2f24d47aada04e34758eb31`, key
`pretrained-sps-smol-ar-9603ad849360-20261105`, observes this wave. The resolved
Qwen group was not reused, and no completed Qwen shard was resubmitted.

## Conclusions

Compare variants within each tokenizer family. Original autoregressive controls
use a different conditioning site and initialization from SPS; achieved lengths
alone cannot isolate architecture, adaptation or a universal capacity limit.
Historical SmolLM CE results are separate from the new matched protocol.
The completed Qwen CE/AR comparison currently favors original AR under the
registered protocol, including its distinct conditioning site and initialization.
KD narrows the observed greedy-generation gap while retaining a lower
teacher-forced horizon than original AR. Interpretation must preserve both metrics.

## Changelog

- 2026-11-05: Complete the interactive requirements interview; verify checkpoint
  sources and start-token metadata; implement pretrained adapters and staged MLS
  launch/preflight routes.
- 2026-11-05: Address independent-review findings, pass focused/full verification,
  preserve the raw cohort while upgrading integrity checks, submit GPU preflights
  and register durable observation.

- 2026-11-05: Audit successful real-checkpoint preflights, submit the original
  Qwen AR main wave and register a new durable group for the complete shard set.

- 2026-11-05: Complete and audit the original Qwen AR cohort; submit the Qwen
  SPS-CE wave, register durable observation and add a reusable cohort auditor.
- 2026-11-05: Address independent-review findings in the cohort auditor: recognize
  successful final measurements at a stage limit and reject continuation after
  either stopping budget. Training code and completed measurements are unchanged.
- 2026-11-05: Audit the complete Qwen SPS-CE cohort and verify paired raw texts
  against original Qwen AR; record teacher-forced and greedy measurements separately.
- 2026-11-05: Submit the Qwen SPS-KD wave after resolving a local proxy timeout;
  verify all archived payloads and register a new durable group.
- 2026-11-05: Recover continuation after the Qwen KD event-delivery error;
  audit its complete cohort and confirm raw-document pairing across all Qwen variants.
- 2026-11-05: Submit original SmolLM AR and register a new durable group after
  verifying its complete job set and immutable source payloads.
