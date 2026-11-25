# Standard and SPS cramming and cached replay with a wide softmax row sum

## Hypothesis

A vector learned with the wider softmax sum should reproduce its converged target during blind cached decoding with the same arithmetic.

## Setup

Use the wide (SPS) and wide_standard suites in `scripts.run_native_fixed_cramming.train`. SPS starts from the native prediction embedding. Standard optimizes an ordinary memory from the original per-text initialization. Each worker checks initialization and target identity against the historical cohort, reloads its saved selected stages, and evaluates them using `scripts.probe_sps_kv_replay.replay`.

## Results

Pending cohort validation.

## Conclusions

Compare the selected horizon, full teacher-forced exact recovery, blind cached and uncached exact recovery, and logits on identical histories. Include every text in the denominator.

## Changelog

- 2026-11-25. Register fresh training and matched cached replay.
- 2026-11-25. Add the matched ordinary-memory Standard arm for the appendix comparison.
