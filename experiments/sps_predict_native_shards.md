# Native-hinge predict cramming with disjoint cohort shards

## Hypothesis
Independent per-text cramming can run in parallel without changing the learned
conditioning site or the comparison cohort. The increased-budget native-hinge
protocol may extend reconstruction beyond the completed original-budget control.

## Setup
Training function: `scripts.analyze_sps_predict_embedding.probe`, selected by
the `scripts.analyze_sps_predict_shards.experiment_configs` registry. The source
registry defines all partitioning, checkpoint, initialization, objective and
training settings. Existing caches, frozen model, fixed BOS, predict-only input
optimization and batch-one generation checks remain in use. Global sample IDs
are preserved across disjoint data slices.

Merge function: `scripts.analyze_sps_predict_shards.merge`. It requires the
complete registered cohort, validates every shard's configuration and provenance,
matches target IDs to the pinned cache and selected horizons to raw saved stages,
and recomputes pooled metrics. It atomically combines all stages, validation
reports, per-text metrics and provenance, then plots the completed comparison.

## Results
All registered shards were submitted from reviewed snapshot
`f96ca50fe8e834c9fe6f9e0251a15042aaf8a3aa`; each initially reported Pending.
Artifacts and the authoritative submission manifest are under
`artifacts/sps_cramming/predict_native_hinge_shards_20261022/`.

- Shard 00: `lm-mpi-job-b91b8d6e-c4d4-448c-84d3-352bc1b73bb5`.
- Shard 01: `lm-mpi-job-68760179-00b5-40d4-b50c-d2e509d82cb0`.
- Shard 02: `lm-mpi-job-6fb7d346-84a4-4d6b-a78b-8cf689c1e8a9`.
- Shard 03: `lm-mpi-job-2f4a193a-cabb-4e0b-9a9a-7549c7478612`.
- Shard 04: `lm-mpi-job-f8a2de97-9a02-421f-9e9c-8a805d4833e5`.
- Shard 05: `lm-mpi-job-de37d8c1-3c25-4967-9956-3954c2b3e3b7`.
- Shard 06: `lm-mpi-job-c5941a33-dcab-496c-95b7-52ec0d4534c1`.
- Shard 07: `lm-mpi-job-a1ee2242-198a-4c03-b977-142e3f92f5dc`.

Durable Arkhip group `1a1d1754301744058fefd9b82af71173` watches the complete
set, with idempotency key `sps-native-hinge-budget30000-8x6-f96ca50-20261023`.
After an all-success outcome, run the merge command in Conclusions and inspect
`merged/`. If a failure resolves observation early, inspect it and register a
new group for remaining/restarted work; do not merge an incomplete cohort.

Implementation passed 52 focused tests and fresh independent adversarial review
(APPROVE). Actual final dry payloads use the same full immutable SHA. Real pinned
cache slices cover every original text exactly once; all cached texts are fully
valid at the registered length. A real tiny CPU native-hinge train/save/replay/
generation smoke verified offset mapping and global IDs. Synthetic merge checks
verified complete stage preservation, pooled metrics, atomic/idempotent outputs
and rejection of incomplete or inconsistent inputs. The snapshot regression also
executes a real Git archive of an older commit after creating a newer HEAD.

The two superseded monolithic budget jobs were
cancelled while still pending. Their original submission records remain intact.
The previous completed native-hinge predict run is the comparison control.
Initial independent review identified missing-stage detection and archived
partial-retry snapshot propagation as edge cases. Regression tests reproduced
both; fixes enforce the original SHA through submission and complete registered
stage progression before merging.

## Conclusions
Use a new durable Arkhip group for the complete shard set. On successful
completion run `scripts/analyze_sps_predict_shards.py --phase merge`, then
inspect combined teacher-forced and free-generation results. Shard failures
must be resolved before merging; cancelled jobs are not successful results.

## Changelog
- 2026-10-22: Replaced the pending native/random monolithic pair with a native-hinge shard registry and audited complete-cohort merge.

- 2026-10-23: Applied independent review findings: immutable archived-snapshot retries and strict raw-stage progression checks, with regression coverage.

- 2026-10-23: Final independent APPROVE and 52 passing tests; submitted every registered shard from the reviewed snapshot and registered the complete durable wait group.
