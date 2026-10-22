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
Preparation in progress. The two superseded monolithic budget jobs were
cancelled while still pending. Their original submission records remain intact.
The previous completed native-hinge predict run is the comparison control.

## Conclusions
Use a new durable Arkhip group for the complete shard set. On successful
completion run `scripts/analyze_sps_predict_shards.py --phase merge`, then
inspect combined teacher-forced and free-generation results. Shard failures
must be resolved before merging; cancelled jobs are not successful results.

## Changelog
- 2026-10-22: Replaced the pending native/random monolithic pair with a native-hinge shard registry and audited complete-cohort merge.
