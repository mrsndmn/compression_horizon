# Pretrained SPS cramming

The accepted interview covers four adapted checkpoints, matched original AR
controls, and reproducible integration into `sps_paper/`. The sole executable
experiment design is `scripts/pretrained_sps_experiments.py`; its settings must
not be duplicated here. The interview record is local at
`run/deep-interview/deep-interview-pretrained-sps-cramming.md`.

## Latest user schedule and storage instructions

The user now requests immediate parallel launch of the remaining cohorts,
without waiting for the active AR jobs. This supersedes the earlier requirement
to return storage before launching later waves. Use
`scripts.run_pretrained_sps_parallel.py` for the selected remaining variants;
scientific model/data/training settings still come from the original registry.

New CE/KD outputs and immutable source archives go directly to physical NFS2 at
`ROOT.parent / (ROOT.name + "_nfs2")`; model and source-commit links in logical
ROOT provide compatibility with the existing worker and auditor. AR continues
writing its existing NFS1 paths. This launcher does not relocate existing data.
The operational entrypoint records its own source SHA and Git commit separately
from the unchanged scientific worker fingerprint.

Keep all active output paths stable until every cohort's writers are terminal.
Then consolidate temporary NFS1 artifacts into the already populated NFS2 tree,
preserving CE/KD directories already there. Skip routing links whose targets are
those directories; do not copy them onto their own targets. Verify all files and
cohort audits before replacing logical ROOT with the consolidated physical NFS2
directory and removing verified NFS1 duplicates. On any space-related failure,
wait for cleanup; do not autonomously relocate artifacts or requeue while full.

## Tasks and current status

- [x] Verify actual checkpoint families/revisions and existing BOS identities.
- [x] Implement differentiable native-backbone SPS interfaces and reference tests.
- [x] Define immutable, deduplicated preflight/main launch routes and wave limits.
- [x] Pass focused/full checks and independent review after fixes.
- [x] Complete both real-checkpoint GPU preflights; verify all six variants.
- [x] Complete and audit original Qwen AR.
- [x] Complete and audit Qwen SPS-CE.
- [x] Complete and audit Qwen SPS-KD.
- [x] Recover the artifact tree from full NFS2 and re-audit all Qwen cohorts.
- [ ] Retry original SmolLM AR after the storage failure; audit its full cohort.
- [ ] After current jobs finish, return the artifact tree to physical NFS2.
- [ ] Submit/observe and audit SmolLM SPS-CE on NFS2.
- [ ] Submit/observe and audit SmolLM SPS-KD.
- [ ] Merge complete cohorts, compare within families, and plot results.
- [ ] Extend versioned article evidence/replay, regenerate assets, compile PDF.
- [ ] Commit/push the final evidence and manuscript; verify applicable PR checks.

Implementation and verification records, complete job UUIDs and measurements are
in `experiments/pretrained_sps_cramming.md` and the external manifests. Source
training fingerprint remains
`9603ad8493600a9e5577a3dad2d7abf28c2b2a41cfd241e3bfeb7f87bdcd5a4f`.
The implementation passed 325 repository tests and independent review; the later
cohort-auditor change passed 61 focused compatibility/replay tests and a fresh
`APPROVE`. Storage recovery changes no model/trainer code or scientific settings.

## Storage incident and completed recovery

The initial SmolLM AR wave came from
`5cf15914950fa9bb349f7bba34080724f720c408`. Durable group
`c6400159d2f24d47aada04e34758eb31`, key
`pretrained-sps-smol-ar-9603ad849360-20261105`, resolved on a failed job.
Its other statuses were cached; a subsequent provider check confirmed all eight
failed with no shard completion marker. Every log confirms `OSError: [Errno 28]
No space left on device` during TensorBoard writes. NFS2 filled completely;
inodes were available. Partial attempts are not valid cohort measurements.

The series' artifact tree was copied to available shared NFS1, and every file
was checked by SHA-256. The registry's logical `ROOT` now links through
`/workspace-SR004.nfs1/d.tarasov/cramming` to
`/workspace-SR004.nfs1/d.tarasov/sps-artifacts/cramming_pretrained_20261105`.
The short intermediate link allowed switching when NFS2 had no free data blocks.
The migration ledger is `ROOT/operations/storage-migration-20261105.json`.
Original derived summaries whose provenance was refreshed by the round-trip
audit are retained under `ROOT/operations/pre-migration-originals/`.

All three Qwen audits passed again with unchanged metrics and fingerprints.
Only verified duplicate copies of this series were removed from NFS2. The eight
failed shard directories, original job records, provider statuses and failure
logs are retained under
`ROOT/failed_attempts/smol_ar/initial-20261105-enospc/`. Canonical SmolLM AR shard
paths are ready for the normal-launcher retry. Input checkpoints remain at their
original read paths. Unrelated datasets and checkpoints were not changed.

## Execution and continuation

Use the existing project Python and `PYTHONPATH=src:.`. The inherited HTTP proxy
hangs on MLspace authentication; use `NO_PROXY=api.ai.cloud.ru` and
`no_proxy=api.ai.cloud.ru` for launcher commands in this environment.

```sh
python -m scripts.run_pretrained_sps_cramming --phase advance
# Register returned complete job UUIDs with a NEW Arkhip job_wait group.
# After a wave successfully completes, audit its full cohort, for example:
python -m scripts.analyze_pretrained_sps --experiment smol_ar
```

`advance` is the historical sequential route. The new concurrent route is:

```sh
python -m scripts.run_pretrained_sps_parallel --dry
python -m scripts.run_pretrained_sps_parallel
```

The new route reuses registered workers/resources, immutable snapshots,
preflight gates and shard deduplication. It accounts for active suite jobs,
checks NFS2 space, and never moves current AR data. Use independent durable groups
for each submitted model. A failure resolves an entire one-shot group, so observe
remaining/replacement jobs with a NEW group. Do not repeat completed Qwen cohorts
or preflights, and do not keep a local dispatcher/polling shell alive. Once all
remaining variants are submitted, do not call advance to submit another wave.

## Final analysis and article

Wait for all requested cohorts before final paired plots and article integration.
Preserve raw-document pairing and native-token identity within each family.
Keep teacher-forced horizon separate from initial correct greedy generation of
the selected vector; the latter is not a search for a maximal greedy horizon.
Report stopping budgets and AR/SPS conditioning-site and initialization differences.
Budget-limited lengths do not establish capacity ceilings or a causal explanation.

Bundle audited evidence with source/job/data provenance for offline replay, add
paired figures/tables, extend article reproduction/checks, compile `sps_paper`,
and commit/push the result. Do not report job submission as measurement.

The branch is `feat/pretrained-sps-cramming`. Push works, but GitHub rejected PR
creation with HTTP 403, `Resource not accessible by personal access token`.
No PR was created and server CI is unverified. This does not block the MLS jobs.

## Active SmolLM AR retry

All replacement shards were submitted from
`41e9014e3c1a62f83b87453175b7d568ee073850`; complete job UUIDs are in the experiment
file and external manifests. Every payload, offset, source archive and physical
output path was verified. Replacement IDs are disjoint from the failed attempt.

Active durable group: `4ac4f9c8421a45309c365b1ecb2903a1`;
key: `pretrained-sps-smol-ar-retry1-9603ad849360-20261105`.
On this group's outcome, audit `smol_ar` when complete and record its result.
Other independently observed cohorts keep running. Do not change global ROOT
or consolidate storage while any of the three model cohorts still has writers.
The remaining CE/KD cohorts are submitted by the concurrent entrypoint above;
record their complete job IDs and independent groups after successful submission.
When all cohorts finish, consolidate storage as above, then finish the complete
paired analysis and article. On new ENOSPC failures, wait for cleanup.
