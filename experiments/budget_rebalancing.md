# Information-Gain Budget Rebalancing

## Hypothesis

A model's information-gain budget (`Σ_token Δsurprisal`) is roughly constant, but plain
cross-entropy cramming spends it unevenly — over-margined easy tokens hog budget while others sit
at near-zero margin. A **budget-rebalancing loss** that reclaims budget from over-margined tokens
should let **more tokens clear a fixed convergence margin ε** (a longer crammed span) than plain-CE
or the existing margin-deficit reweighting (+LM), which only raises the floor and never reclaims.
Pre-registered null: if the budget is **not** fungible across tokens (through the single shared
embedding), the token count stays flat while the margin distribution still flattens — evidence that
the budget is not redistributable, which is itself a finding.

## Setup

- **Training function**: `ProgressiveCrammingTrainer` (`src/compression_horizon/train/trainers/progressive_cramming.py`)
- **Loss**: `budget_rebalance_loss_with_prefix` (`src/compression_horizon/train/loss.py`), two arms
  - `cap` (C): margin-deficit CE floor + a reclaim term pulling over-budget tokens down to an
    adaptive water level `c = B/L` (B = per-sample IG at stage start, cached `H_base`).
  - `dual` (D): maximize a soft count of tokens past ε subject to `Σ Δbits ≤ B` via a per-sample
    dual variable (dual ascent). KKT optimum is water-filling.
- **Baselines (already run)**: plain-CE (`convergence_margin=ε`) and +LM (`loss_margin=ε`).
- **Model / dataset / grid**: Pythia-1.4B deep dive (baseline + low-dim 256), ε ∈ {0.5, 1.0, 2.0},
  PG19, 50 samples. Full matrix lives in `scripts/jobs/run_jobs_progressive.py`
  (`BUDGET_REBALANCE_EXPERIMENTS`, 12 new runs). **Source is the single source of truth.**
- **Metric**: crammed-token count at fixed ε (C/D vs CE/+LM). Diagnostics (fungibility probe):
  per-token margin variance, min-margin, Δbits inequality — logged under `budget/*` in TensorBoard.

## Training

```bash
PYTHONPATH=./src python scripts/jobs/run_jobs_progressive.py --dry
```

## Expected outcome

At each ε, `cap`/`dual` converge a **longer span** (more crammed tokens) than plain-CE and +LM at
matched ε, with the IG budget conserved (`Σ Δbits` ≈ baseline) and a flatter per-token margin
distribution. A flat token count with flattened margins would support the non-fungible-budget null.

## Results

_To be filled after running the experiment._

## Conclusions

_To be filled after analysis._

## Changelog

- 2026-07-03: Plan created; `cap`/`dual` losses + cached `H_base` + 12 Pythia arms implemented and
  unit/integration-tested. Runs not yet launched. Design + rationale: deep-interview spec
  (`run/deep-interview/deep-interview-budget-rebalancing-loss.md`) and ADRs 0004/0005.
