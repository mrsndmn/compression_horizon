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

**The first launch was broken and was fixed before the reported runs.** The initial 12 cap/dual jobs
stalled at stage 0 — every sample crammed 0 tokens (`final_convergence = 0`, `final_loss = 0`)
because the loss gated its whole objective, *including the reconstruction CE floor*, on the
base-surprisal mask, which is empty at the single-token / no-prefix first stage (token 0 has no
defined base surprisal). Fixed in commit `0c6d206` — the padding mask used by the floor / soft-count
was split from the base-surprisal mask used only by reclaim / the bits budget (+2 regression tests) —
then the 12 arms were re-launched and completed.

**Crammed tokens (headline metric), Pythia-1.4B, 50 samples of `pg19_1k`:**

| base / ε | CE | +LM | cap (C) | dual (D) |
|---|--:|--:|--:|--:|
| full · 0.5      | 407.1 | 353.2 | 293.1 | 2.8 |
| full · 1.0      | 390.9 | 316.1 | 270.6 | 2.6 |
| full · 2.0      | 352.8 | 259.1 | 226.7 | 2.4 |
| lowdim256 · 0.5 | 440.2 | 253.2 | 163.5 | 2.7 |
| lowdim256 · 1.0 | 400.6 | 223.1 | 149.4 | 2.6 |
| lowdim256 · 2.0 | 337.7 | 172.7 | 132.9 | 2.4 |

Ordering is **CE > +LM > cap ≫ dual** at every (base, ε). Arm C now trains correctly (crammed 1 → 133–293)
but never beats the baselines: it is "+LM floor + reclaim", and the reclaim term — pulling budget off
over-margined tokens — *shortens* the horizon versus plain +LM. Arm D is degenerate, cramming only
2–3 tokens with **negative** true bits-saved (−8 to −21 bits, `tf_conv` ≈ 0.5), because its soft-count
objective has no reconstruction term to satisfy the 100%-argmax convergence gate as the prefix grows.
Full metrics (cleared_ε, crammed, true bits-saved, bits/tok, tf_conv) per arm are in
`artifacts/results/budget_rebalancing.csv`.

**Fungibility diagnostic** (`scripts/diagnose_ig_budget.py`, GPU-free, on the plain-CE runs): the
marginal information gain of the newly-crammed token, `IG(L) − IG(L−1)`, stays at **~3.6–4.1 bits
right at the horizon** — it does not decay toward 0 — so the IG budget is *not exhausted* where
cramming stops. Meanwhile per-token IG erodes (~5.3 → ~3.3 bits) and **17–30 % of stage steps have a
negative marginal** (adding a token costs earlier tokens some bits). Figure:
`artifacts/results/ig_budget_diagnostic.png`.

## Conclusions

**The hypothesis is refuted — more strongly than the pre-registered null.** The null said a
non-fungible budget would leave the crammed-token count *flat* while the margin distribution
flattened. In fact the count *drops*: reclaiming surplus bits actively costs horizon (cap < +LM at
every ε). Budget-rebalancing by suppression does not work.

**Why (from the diagnostic): there is no exhausted budget to redistribute.** The frontier token still
brings ~4 fresh bits when cramming halts, so the horizon is not limited by a depleted budget — it is
limited by the embedding's *representational capacity* to encode the next token. Reclaiming budget
from over-margined earlier tokens therefore hands the frontier nothing it needs, and instead spends
capacity suppressing non-bottleneck tokens — exactly why cap shortens the horizon. The resource is
genuinely shared and contended (negative marginals, per-token IG erosion, and the low-dim runs being
relatively worst all point to capacity as the binding constraint), but the contention is a minority
effect that cannot be arbitraged by suppression. Arm D fails for a second, simpler reason: with no
reconstruction floor its objective cannot meet the argmax convergence gate.

**Takeaway / next steps.** Do not pursue further reclaim-surplus variants — the premise is wrong at
the source; the constraint is capacity, not a fixed budget. The directions the data leaves open are
(a) an *additive* max-min / soft-min-margin objective that concentrates optimization effort on the
weakest token without ever suppressing well-served ones (the "no suppression" property is what makes
+LM beat cap), and (b) increasing usable capacity via multiple / specialized memory slots. Both are
untried here.

## Reproduce

All steps run from the repository root with the fixed loss (commit `0c6d206` or later).

1. **Launch the arms** (idempotent — skips existing output dirs; the `_cm_{ε}` CE and `_cm_{ε}_lm_{ε}`
   +LM baselines must already be on disk). Add `--dry` to preview:
   ```
   PYTHONPATH=./src python scripts/jobs/run_jobs_progressive.py --model pythia
   ```
   12 `a100.1gpu` jobs = 2 base × 3 ε × {cap, dual} (`BUDGET_REBALANCE_EXPERIMENTS`).
2. **Aggregate** (arms → rows, metrics → columns; prints the per-(base, ε) table and writes the CSV):
   ```
   PYTHONPATH=./src python scripts/aggregate_budget_rebalancing.py --csv artifacts/results/budget_rebalancing.csv
   ```
3. **Fungibility diagnostic** (GPU-free; reads the saved plain-CE `information_gain_bits`):
   ```
   PYTHONPATH=./src python scripts/diagnose_ig_budget.py
   ```
   Prints the per-run table and writes `artifacts/results/ig_budget_diagnostic.png`.

## Changelog

- 2026-07-03: Plan created; `cap` (C) / `dual` (D) losses, cached `H_base`, and the Pythia-1.4B arms
  (plus fixed-ε baseline) implemented and tested. Design and rationale in the deep-interview spec
  (`run/deep-interview/deep-interview-budget-rebalancing-loss.md`) and ADRs 0004/0005.
- 2026-07-05: Reformatted plan to the standard high-level template.
- 2026-07-05: Launched the 12 Pythia-1.4B budget-rebalance jobs (2 base × 3 ε × {cap, dual}); the
  fixed-ε CE / +LM baselines already exist on disk. Added `scripts/aggregate_budget_rebalancing.py`
  (arms in rows, metrics in columns) and `scripts/watch_budget_rebalancing.py`. Results/Conclusions
  to be filled once the jobs finish and aggregate.
- 2026-07-06: The 2026-07-05 launch was degenerate — all arms stalled at stage 0 (crammed 0). Root
  cause: the loss gated the reconstruction CE floor on the empty base-surprisal mask of the
  single-token first stage. Fixed in `0c6d206` (split padding vs base-surprisal masks; +2 regression
  tests). Deleted the broken run dirs + stale aggregation and re-launched the 12 arms.
- 2026-07-07: All 12 arms finished and aggregated. Added `scripts/diagnose_ig_budget.py` (GPU-free
  fungibility diagnostic on plain-CE `information_gain_bits`). Filled Results/Conclusions: hypothesis
  **refuted** (CE > +LM > cap ≫ dual; reclaim shortens the horizon; the marginal IG never exhausts at
  the horizon, so the constraint is representational capacity, not a fixed redistributable budget).
  Added a Reproduce section.
