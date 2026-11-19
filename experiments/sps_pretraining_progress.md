# SPS cramming across pretraining checkpoints

## Hypothesis

The teacher-forced gain from shared prediction over ordinary memory may grow
as the same SPS model sees more pretraining tokens. Measure the paired gain at
each snapshot on the original pilot cohort. The comparison follows one training
trajectory, including the learned prediction embedding at each snapshot.

## Setup

The registry and launcher are in `scripts.analyze_sps_pretraining`. Ordinary
memory uses `PairedMemoryTrainer`, a specialization of
`ProgressiveCrammingTrainer` that restores each text's initial vector from the
historical memory run. Shared prediction uses `PredictEmbeddingTrainer` through
`scripts.analyze_sps_predict_embedding.probe`.

1. Add an `experiment_configs` sweep to the native SPS launcher. Pair the two
   conditioning interfaces at each intermediate snapshot and reuse the audited
   final-checkpoint pilot as an anchor. Derive settings and token IDs from the
   historical strict-margin pilot. Verify the anchor's provenance and resolved
   settings before combining it with the new runs.
2. Pass the registered checkpoint through shared-prediction training and the
   shard audit. Make ordinary-memory validation read the registered shard offset
   so it replays the correct target IDs and checks global sample IDs. Merge by
   checkpoint and conditioning interface. Reuse the existing trainers and
   snapshot submission flow.
3. Verify checkpoint selection and nonzero offsets in the dry payloads and a
   short real-checkpoint run for each interface. Before analysis, check the
   complete paired cohort, exact token IDs, stage counters, stopping reasons,
   teacher replay, and checkpoint provenance for each result.
4. Report both absolute teacher-forced horizons and their paired difference at
   each snapshot. Compare the change in that difference against the earliest
   snapshot with paired bootstrap intervals over texts. Plot achieved horizons
   at common optimizer-step cutoffs from saved converged stages. Report the
   greedy prefix to first error and greedy success when truncating the selected
   vectors' decoded outputs to a common length.

## Results

All 24 intermediate shards completed and passed the merge audit. Every selected
prefix replays exactly under teacher forcing. Shared prediction reaches a longer
teacher-forced prefix on every paired text at every snapshot. From the earliest
snapshot to the final anchor, the mean paired gap grows from 71.15 to 100.58
tokens, an increase of 29.44 tokens (paired 95% text-bootstrap interval
24.98–33.90). The ratio of mean horizons falls from 8.41 to 4.19. The absolute
gap also grows at the common cutoffs from 2,500 updates onward. At 1,000 updates
it falls from 17.44 to 13.29 tokens between those endpoints.

The mean shared-prediction greedy prefix rises from 80.75 tokens to 109.40 at the
last intermediate snapshot and 110.23 at the final anchor. The complete selected
shared-prediction prefix decodes exactly on 48/48 texts at the earliest snapshot
and 38/48 at the final anchor, where selected prefixes are longer.

`artifacts/sps_cramming/pretraining_progress_20261120/` contains the merged
`summary.json`, `comparison.png`, and submission manifest with the full job IDs.
The immutable training source is `d6f7e3da944b2944a7ef4ffd7607514cfeb95917`.
Durable wait group `61ebf51355e74a9684de9a2b58fa267e` resolved successfully.

## Conclusions

The conditioning-site effect is already large at the earliest measured snapshot.
Both modes improve over pretraining. Shared prediction gains more tokens in
absolute terms, and ordinary memory improves faster relative to its early
horizon. The comparison follows one training trajectory, including changes in
native initialization, and the intervals describe variability over texts.

## Changelog

- 2026-11-19 — Record the hypothesis, implementation plan, and comparisons.
- 2026-11-20 — Add the checkpoint sweep and offset-aware replay.
- 2026-11-20 — Submit the sweep and register durable observation. Verify the
  archived memory initialization hashes and the historical prediction design.
- 2026-11-20 — Audit all completed shards and record the paired outcomes.
