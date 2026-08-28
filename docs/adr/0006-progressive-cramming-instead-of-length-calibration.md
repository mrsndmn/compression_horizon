# 6. Reach the lossy regime with progressive cramming rather than a calibrated fixed length

Date: 2026-08-29

## Status
Accepted

## Context
A "does it preserve more semantics" comparison is only meaningful in a regime where
information is actually being lost. Cramming a short enough span into a memory embedding
reaches token-perfect reconstruction on both architectures, and a comparison run there
returns a null by construction: nothing is lost, so nothing can be preserved *better*.

The straightforward fix is to calibrate a fixed crammed length per model — sweep the span
until the baseline's convergence lands in some lossy band, then run the real comparison at
that length. That introduces a tuning axis that must itself be justified, and a length
tuned on one architecture is not obviously the right length for the other.

## Decision
Run **Progressive Cramming** and let each arm grow its own span until it stops converging.
The horizon each arm reaches is the measured quantity, not a hyperparameter, so no length
calibration step exists.

The `tab:semantic_evaluation` protocol (fixed short benchmark contexts from HellaSwag and
ARC-E) is kept as the semantic axis unchanged, because the paper already shows a ~20 point
[[Retention Gap]] there — that protocol is empirically far from ceiling and needs no
calibration either.

## Consequences
- Arms reach **different** final compressed lengths, so their memory embeddings do not hold
  the same amount of text. Any comparison of pg19 metrics across arms is a comparison at
  each arm's own horizon and must be read that way; the benchmark protocol, where context
  length is fixed by the dataset, carries the matched-comparison weight instead.
- "Equal budget" is therefore defined as equal **optimization steps** (10k per sample /
  1k per token, matching every other progressive run in the project), not equal compressed
  length and not equal final convergence.
- The compression horizon, information gain and trajectory length come out of the same runs
  for free, so the new arms drop straight into the existing 4-metric progressive tables.
