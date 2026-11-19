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

Pending.

## Conclusions

A growing gap supports dependence on pretraining progress along this run.
Parallel gains in both interfaces indicate a shared improvement in reconstruction.
The native initialization also evolves with the checkpoint. This comparison
estimates the effect of pretraining progress on the full cramming protocol.

## Changelog

- 2026-11-19 — Record the hypothesis, implementation plan, and comparisons.
- 2026-11-20 — Add the checkpoint sweep and offset-aware replay.
