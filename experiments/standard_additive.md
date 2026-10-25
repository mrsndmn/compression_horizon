# Standard cramming with a shared additive input vector

## Hypothesis

Repeated injection of the same learned vector may increase reconstruction
length without adding trainable scalars. Unlike SPS shared prediction, adding
the vector to ordinary inputs also changes context representations.

## Setup

`AdditiveCrammingTrainer` in `scripts/analyze_standard_additive.py` implements
the requested input sequence: one learned prefix vector followed by ordinary
token embeddings with that same vector added. Training, teacher replay and
greedy generation use the same conditioning function and ordinary causal mask.
The Python `experiment_configs` registry and `resolved_training_args` are the
design authority. They reuse the completed Standard strict-hinge control,
including exact cached texts and the saved initial vector for each text.

Use `--phase dry` to inspect the immutable MLS payloads, `--phase submit` to
submit registered shards, and `--phase merge` after successful completion.
The merge checks source snapshots, settings, raw stages, sample IDs, target IDs,
paired initialization and replay metrics before computing paired comparisons.
Artifacts live under the shared SPS artifact root in `standard_additive_20261025`.

## Results

The focused implementation suite passes. A real native Standard checkpoint
passes BF16 forward/backward with a finite nonzero vector gradient and frozen
model weights. All source-control initializations load successfully. Training
jobs have not yet been submitted; no new reconstruction result is claimed.

## Conclusions

The primary comparison changes the conditioning interface within the same
Standard checkpoint. A gain would test repeated access to the shared vector;
it would not establish SPS-style independence of ordinary context states.

## Changelog

- 2026-10-25: Implement the user-requested shared additive intervention with
  paired initialization and matched original-budget control.
