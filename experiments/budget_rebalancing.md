# Information-Gain Budget Rebalancing

## Hypothesis

A model's information-gain budget — the total surprisal it can absorb into one crammed embedding —
is roughly fixed, but plain cross-entropy cramming spends it unevenly: over-margined easy tokens hog
budget while others sit near zero margin. A **budget-rebalancing loss** that reclaims budget from
over-margined tokens should let **more tokens clear a fixed reconstruction margin ε** (a longer
crammed span) than the fixed-ε convergence-margin baseline. Two loss arms are compared — Arm C
(surprisal-cap) and Arm D (dual water-filling) — with gains measured on **true bits-saved** using a
cached `H_base`. Pre-registered null: if the budget is not fungible across tokens (through the single
shared embedding), the crammed-token count stays flat while the margin distribution still flattens,
which would itself be evidence that the budget is not redistributable.

## Setup

- **Training function**: `ProgressiveCrammingTrainer`
- **num_gpus**: 1
- **Instance type**: `a100.1gpu`
- **Model**: Pythia-1.4B deep dive
- **Artifact path**: `artifacts/experiments_progressive/<experiment_name>/`

All experiment configurations — arms C and D plus the fixed-ε convergence-margin baseline — are
defined in `scripts/jobs/run_jobs_progressive.py`. Source is the single source of truth.

## Results

_To be filled after running the experiment._

## Conclusions

_To be filled after analysis._

## Changelog

- 2026-07-03: Plan created; `cap` (C) / `dual` (D) losses, cached `H_base`, and the Pythia-1.4B arms
  (plus fixed-ε baseline) implemented and tested. Design and rationale in the deep-interview spec
  (`run/deep-interview/deep-interview-budget-rebalancing-loss.md`) and ADRs 0004/0005.
- 2026-07-05: Reformatted plan to the standard high-level template.
- 2026-07-05: Launched the 12 Pythia-1.4B budget-rebalance jobs (2 base × 3 ε × {cap, dual}); the
  fixed-ε CE / +LM baselines already exist on disk. Added `scripts/aggregate_budget_rebalancing.py`
  (arms in rows, metrics in columns) and `scripts/watch_budget_rebalancing.py`. Results/Conclusions
  to be filled once the jobs finish and aggregate.
