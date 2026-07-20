# 4. Cross-entropy temperature as a training knob (raw + T² arms; convergence stays argmax)

Date: 2026-07-10

## Status
Accepted

## Context
We want to ablate how a **temperature** applied to the reconstruction cross-entropy
affects progressive cramming on pythia-1.4b. The reconstruction loss is standard
next-token CE (`next_token_cross_entropy_loss_with_prefix`, `src/compression_horizon/train/loss.py`),
while convergence — the pass/fail bar that ends a stage — is judged separately on the
**argmax** match rate (`token_argmax_match_rate_with_prefix`), which is invariant to any
positive scaling of the logits.

Two things had to be decided before coding:
1. **Where temperature acts.** Scaling `logits / T` inside CE changes the loss surface and
   gradient, but leaves the argmax convergence criterion untouched. So temperature can only
   change the *optimization path* (steps-to-converge, and whether a run hits the per-token
   step cap), not the converged solution a run reaches given unlimited steps.
2. **Gradient magnitude vs. distribution shape.** Raw `CE(logits/T)` has a gradient that
   scales ~`1/T`; at the sweep's fixed `learning_rate=0.5` that entangles temperature with
   effective step size. The Hinton distillation convention multiplies the loss by `T²` to
   hold gradient magnitude ~constant, isolating the pure distribution-shape effect. It is
   not obvious a priori which effect drives any observed change.

## Decision
- Add a training-time **[[Cross-Entropy Temperature]]** `T` that divides the logits by `T`
  before the reconstruction cross-entropy. `T = 1.0` is the default and is byte-identical to
  the current code path (no flag emitted, no exp_suffix change — same `.get()`-guarded
  extension pattern as `convergence_margin`/`loss_margin`).
- Add a **[[Gradient-Compensated Temperature]]** mode: sweep both the **raw** form and the
  `T²`-compensated form as separate experiment arms, rather than pre-committing to one.
- **Leave convergence argmax-based** (temperature-invariant). Temperature is a loss-only
  knob; the convergence bar is unchanged.
- Ablate on **pythia-1.4b baseline CE only** (temperature is the entire loss there), over
  `T ∈ {0.1, 0.5, 1.0, 1.5, 2.0}` × {raw, T²}. The two `T=1.0` arms are identical and to
  each other and to the existing baseline, so they collapse to a single control run.

## Consequences
- The ablation's expected signal lives mainly in **Trajectory Length** (steps) and in
  **Compressed Tokens** via runs that time out at the per-token step cap under an unfavorable
  temperature — not in a categorically different converged solution. The results table
  (`tab:progressive_temperature`) reuses the `tab:progressive_modifications` 4-column metric
  set so the temperature rows are directly comparable to the existing baseline row.
- Running both raw and T² arms doubles the run count (~9 distinct runs after `T=1.0` dedup)
  but lets the data, not an upfront guess, decide whether any effect is distribution-shape or
  effective step-size.
- Because `T=1.0` is byte-identical, every pre-existing experiment and cached artifact is
  unaffected; only the new temperature-tagged experiments carry the flag and a `_temp_*`
  (and compensation-mode) suffix.
- Temperature is defined for the CE term only; it has no meaning for the activation-alignment
  (l1/l2/cosine) loss, which is one reason the sweep is confined to the pure-CE baseline.
