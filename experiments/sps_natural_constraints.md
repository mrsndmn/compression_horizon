## Hypothesis
At the natural text boundary, protecting additional old targets can prevent a shared prediction embedding from satisfying the new target. Two natural contexts ending in the same tokens may also be harder to control independently.

## Setup
Training and measurement function: `scripts.run_sps_natural_constraints.probe`.

For the boundary comparison, report per-text success, old-position violations and the first positions to lose margin. Average the two subset draws within text. Distinguish texts originally stopped by the overall update budget from per-stage failures. Compare the original short pass to the expanded pass before updating.

For the context comparison, report text coverage, shared suffix lengths, singleton success, and paired outcomes on texts where all three singleton fits succeed. Compare joint step counts within pairs that both converge and count censored fits separately. Matching fixes the target IDs and anchor, with a tolerance on starting margin for the replacement. Position and earlier context can still differ. Report how many matches involve a boundary conflict as a descriptive follow-up.

## Results
All 48 PG19 boundaries were verified. No new target met the margin at initialization. Expanding the old input before updating the vector cost at least one old margin in 12 texts (one to six positions). Fitting the new target alone succeeded on 48/48, with a median of seven updates. Every text lost at least one additional old margin on the first update. At convergence a median of 98 old positions missed the margin, and the median within-text fraction was 74%. The old targets selected for protection plus the new target passed in 96/96 fits at each intermediate protection level. Median updates increased to 20.5, 28.5 and 42 across those levels. Across all arms with unprotected positions, zero final vectors preserved top-1 on the entire text. Full protection recovered the extended prefix on 14/48 by margin and 25/48 by teacher-forced argmax. Strict recoveries comprised 11/45 originally stopped at the total budget and 3/3 stopped at the per-stage limit.

All 48 texts supplied a matched triple. Changing selection to use only token IDs and initial margins replaced six triples. Their fits were recomputed. The median gap between the matched starting margins was 0.0625. The shared suffix spanned one to five tokens. On 47 texts all singleton fits succeeded. Both joint fits passed on all 47, with a median of six updates in each condition and zero median paired difference (shared slower in 20, faster in 15, tied in 12). Both joint fits also passed on the remaining text despite one singleton timeout. None of the primary matched pairs contained the new boundary target together with an old target observed to lose margin in its new-only fit. Corrected raw reports and the per-text CSV are under `artifacts/sps_cramming/natural_constraints_v2/`.

## Conclusions
The new target is individually controllable throughout this cohort. Updating it from a converged vector produces widespread interference, and protecting all old targets is considerably harder than protecting sampled subsets. Same-dimension full-prefix rescues establish feasibility at those rescued boundaries. The other timeouts do not identify the representational capacity of the embedding. At native initialization the paired natural-context probe showed no joint success deficit for shared suffixes. It probes two constraints at a time and does not test the same pairs in a saturated vector. All scores condition on true earlier tokens.

## Changelog
- 2026-11-24. Added natural boundary restarts and matched within-text prefix pairs.
- 2026-11-24. Verified full boundary and corrected matched-pair results and exported per-text outcomes.
