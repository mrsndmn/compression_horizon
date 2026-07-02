# 5. Rebalance on true bits-saved, with a cached per-stage H_base

Date: 2026-07-03

## Status
Accepted

## Context
The [[Budget-Rebalancing Loss]] must cap "how much budget a token consumes." Two candidate
quantities:

- **Per-token margin** `m_i = logit[true] − runnerup` — falls out of the logits already
  computed, so it is **free**, and is monotone in bits-saved near the decision boundary.
- **Per-token bits-saved** `Δbits_i = (H_base,i − CE_i)/ln2` — faithful to the
  [[Information-Gain Budget]] (`H_base` = frozen-LM surprisal without the memory embedding),
  but naively needs an extra `H_base` forward **every** optimization step (≈2× forward cost
  over up to 10k steps/sample).

The margin proxy is cheaper; bits-saved is the actual quantity the "IG budget" story is about.

## Decision
Rebalance on **true per-token bits-saved**, not the margin proxy. Exploit that **`H_base` is
independent of the memory embedding being optimized** (it is the frozen LM on the plain
prefix+continuation): compute per-token `H_base` **once per stage and cache it**. Per step,
only `H_comp,i = CE_i` changes — and that is already computed by the loss — so
`Δbits_i = (H_base,i − CE_i)/ln2` is available with **no extra per-step forward**. Reuse the
base-forward logic in `analysis/information_gain.py` (Eq. 9) for both the cache and the
offline IG-conservation check.

## Consequences
- Faithfulness without the feared 2× cost: the extra cost collapses to **one** frozen-LM
  forward per stage, not per step.
- Correctness dependency: the cache must be **invalidated when the span grows** (each new
  progressive stage recomputes `H_base` for the extended continuation) and be prefix-aware
  (`H_lm(cont | prefix)`); a stale cache silently corrupts every `Δbits_i`.
- Ties the loss directly to the paper's IG definition, so the IG-conservation acceptance
  criterion (`Σ Δbits` under C/D ≈ baseline) is a first-class check.
- Trade-off: rejects the essentially-free margin proxy; if bits-saved and margin turn out to
  rank tokens near-identically, the extra machinery buys little — an escalation/​fallback to
  the margin proxy remains open but is not the chosen path.
