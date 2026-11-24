## Hypothesis
At the natural text boundary, protecting additional old targets can prevent a shared prediction embedding from satisfying the new target. Two natural contexts ending in the same tokens may also be harder to control independently.

## Setup
Training and measurement function: `scripts.run_sps_natural_constraints.probe`.

For the boundary comparison, report per-text success, old-position violations and the first positions to lose margin. Average the two subset draws within text. Distinguish texts originally stopped by the overall update budget from per-stage failures. Compare the original short pass to the expanded pass before updating.

For the context comparison, report text coverage, shared suffix lengths, singleton success, and paired outcomes on texts where all three singleton fits succeed. Compare joint step counts within pairs that both converge and count censored fits separately. Matching fixes the target IDs and anchor, with a tolerance on starting margin for the replacement. Position and earlier context can still differ. Report how many matches involve a boundary conflict as a descriptive follow-up.

## Results
Pending.

## Conclusions
Pending paired measurements.

## Changelog
- 2026-11-24. Added natural boundary restarts and matched within-text prefix pairs.
