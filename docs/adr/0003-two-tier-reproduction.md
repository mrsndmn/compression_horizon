# 3. Two-tier reproduction: toy quickstart + documented full configs

Date: 2026-06-23

## Status
Accepted

## Context
The paper's experiments run at a scale no public visitor can casually reproduce: 8×A100,
the pg19 benchmark, ~50 samples, 4096-token sequences, up to 10k optimization steps per
sample. A README that only documents those runs would be technically faithful but
practically un-runnable for almost everyone, undermining the goal of supporting
reproducibility. Conversely, a toy-only demo would leave no path to the paper numbers.

## Decision
Ship **two tiers**:
1. A **toy quickstart** (primary path) — each of the three methods on a small model over
   a few samples, printing its headline metric (full: exact reconstruction ≈ 0.99;
   progressive: stages converging; low-dim: recovered accuracy) in minutes on a single
   GPU, saving a tiny artifact.
2. **Documented full configs** — the exact CLI flags / commands to reproduce paper-scale
   numbers on a cluster, presented as documentation (not promised to be cheap).

## Consequences
- Anyone can verify the mechanics quickly; serious reproducers have the real recipe.
- The README is structured around these two sections; the quickstart carries a concrete
  metric bar (acceptance-tested), the full tier carries config fidelity, not a runtime
  promise.
- The toy quickstart must stay fast and dependency-light, constraining model/sample
  choices (reinforces [[0001-lean-core-extraction-scope]]).
- Trade-off: the public repo does not itself certify the paper tables — bit-faithful
  reproduction is an explicit non-goal.
