# Standard and SPS cramming and cached replay with a wide softmax row sum

## Hypothesis

A vector learned with the wider softmax sum should reproduce its converged target during blind cached decoding with the same arithmetic.

## Setup

Use the wide (SPS) and wide_standard suites in `scripts.run_native_fixed_cramming.train`. SPS starts from the native prediction embedding. Standard optimizes an ordinary memory from the original per-text initialization. Each worker checks initialization and target identity against the historical cohort, reloads its saved selected stages, and evaluates them using `scripts.probe_sps_kv_replay.replay`.

## Results

All 48 Standard vectors met the convergence criterion and survived full teacher-forced replay after serialization. Both blind decoders recovered every selected prefix, spanning 2,578 target tokens in total (53.71 per text). Full, growing and cached forwards agreed on all 129,683,712 vocabulary logits scored on identical histories. The per-text counts are versioned in `sps_paper/data/standard_wide_cache.csv`.

All 48 SPS vectors converged and recovered their selected prefix in full teacher-forced and both blind decoders. Across 6,313 target tokens (131.52 per text), full, growing and cached replay agreed on all 317,569,152 logits. Per-text counts are in `sps_paper/data/sps_wide_cache.csv`.

## Conclusions

Both models recovered every selected prefix on this cohort when compression and decoding used the same wider row sum. SPS recovered longer prefixes on average.

## Changelog

- 2026-11-25. Register fresh training and matched cached replay.
- 2026-11-25. Add the matched ordinary-memory Standard arm for the appendix comparison.
- 2026-11-25. Audit all four completed Standard shards against serialized targets and blind outputs.
- 2026-11-25. Audit all four SPS shards and report the paired clean cohorts in the manuscript appendix.
