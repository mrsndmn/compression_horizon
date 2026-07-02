# 4. Budget-rebalancing losses (C: surprisal-cap, D: dual water-filling)

Date: 2026-07-03

## Status
Accepted

## Context
Progressive cramming converges each stage to a per-token logit margin (the
[[Convergence Margin (ε)]] criterion). A model's total [[Information-Gain Budget]]
(`Σ_token Δsurprisal`) is roughly constant, yet plain cross-entropy spends it unevenly:
easy tokens acquire huge margins (over-served) while others sit near zero. The branch already
ships `loss_margin` (the "+LM" margin-deficit reweighting), whose docstring claims "more
crammed tokens" — but +LM only **raises the floor** (up-weights deficient tokens, zeroes
easy ones) and **never reclaims** the budget hogged by over-margined tokens. So a naive "new
loss" risks re-describing +LM.

## Decision
Introduce a distinct family of [[Budget-Rebalancing Loss]] objectives that actively
**reclaim** budget from over-margined tokens so more tokens clear ε at a **fixed** ε:

- **C — surprisal-budget cap.** Cap each token's bits-saved at a shared, **adaptive**
  water-level `c = B / L` (B = per-sample IG budget, L = current span length), penalizing
  `(Δbits_i − c)₊`. This is the +LM floor-raiser **plus** a reclaim term +LM lacks.
- **D — dual water-filling.** Maximize a soft count of tokens ≥ ε subject to `Σ Δbits_i ≤ B`,
  enforced by a dual variable λ (dual ascent); KKT optimum ⇒ water-filling; C is its
  fixed-level special case.

The water-level/budget is **adaptive from the measured IG** (no hyperparameter sweep). The
convergence criterion stays margin-ε; the reclaim term is floored so it never pulls a token
below ε.

## Consequences
- The loss is genuinely distinct from +LM: reclaim, not just floor-raising. The comparison
  arms are CE, +LM, C, D at matched ε — success = C/D cram **more tokens at fixed ε**.
- Adaptive water-level adds **zero** sweep runs and directly operationalizes the fixed-budget
  premise; if the premise is wrong (budget not fungible), the pre-registered null still holds
  (a finding, not a failure).
- Two objectives (fixed-level C and learned-level D) probe the same water-filling idea at
  different complexity; D's dual update lives in the inner cramming loop.
- Hard to reverse once runs, tables, and appendix text are built around C/D as the canonical
  rebalancers; the alternatives (two-sided margin, raise-the-floor soft-min) are deferred, not
  deleted.
