# 1. Lean-core extraction scope for the public repo

Date: 2026-06-23

## Status
Accepted

## Context
The public `progressive_cramming` repo must support reproducibility of three methods:
[[full-cramming]], [[progressive-cramming]], and [[low-dim-projection]]. The internal
`compression_horizon` source tree is much larger — it also contains `compression_head`,
`prefix_tuning`, ~15 ablation launchers (layer/width/init/prefix/added-tokens/basin/
attention-knockout), and a heavy topology-analysis stack (`ripser`/`persim`/`umap`).
Copying all of it would bloat the public surface, drag in unrelated dependencies, and
create maintenance burden for code no visitor needs.

## Decision
Copy only the **lean core**: the three trainers (`FullCrammingTrainer`,
`ProgressiveCrammingTrainer`, `LowDimTrainer`), their transitive `src` dependencies
(parametrization, embedding_init, loss, inputs, optimization, base trainer, convergence,
perplexity, generation, information_gain, model/data loading, utils), and one adapted
entry point. Exclude `compression_head`, `prefix_tuning`, all ablation launchers, and the
topology stack. Trim `pyproject.toml` to the deps the lean core actually imports.

## Consequences
- Small, auditable public surface; faster install; fewer dependencies to break.
- The import closure must be verified during extraction — anything not reached by the
  three trainers + their eval is dropped, and no dead imports remain.
- Future releases of additional methods/ablations require a deliberate follow-up, not an
  accidental copy.
- Trade-off: the public repo is not a full mirror of the research code; some analyses
  shown in the paper/presentation cannot be rerun from it (acceptable — see
  [[0003-two-tier-reproduction]]).
