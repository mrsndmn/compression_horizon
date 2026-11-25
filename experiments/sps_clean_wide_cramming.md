# Standard and SPS cramming and cached replay with a wide softmax row sum

## Hypothesis

A vector learned with the wider softmax sum should reproduce its converged target during blind cached decoding with the same arithmetic.

## Setup

Use the wide (SPS) and wide_standard suites in `scripts.run_native_fixed_cramming.train`. SPS starts from the native prediction embedding. Standard optimizes an ordinary memory from the original per-text initialization. Each worker checks initialization and target identity against the historical cohort, reloads its saved selected stages, and evaluates them using `scripts.probe_sps_kv_replay.replay`.

## Results

All 48 Standard vectors met the convergence criterion and survived full teacher-forced replay after serialization. Both blind decoders recovered every selected prefix, spanning 2,578 target tokens in total (53.71 per text). Full, growing and cached forwards agreed on all 129,683,712 vocabulary logits scored on identical histories. The per-text counts are versioned in `sps_paper/data/standard_wide_cache.csv`. The matched SPS cohort is pending.

## Conclusions

Compare the selected horizon, full teacher-forced exact recovery, blind cached and uncached exact recovery, and logits on identical histories. Include every text in the denominator.

## Changelog

- 2026-11-25. Register fresh training and matched cached replay.
- 2026-11-25. Add the matched ordinary-memory Standard arm for the appendix comparison.
- 2026-11-25. Audit all four completed Standard shards against serialized targets and blind outputs.
