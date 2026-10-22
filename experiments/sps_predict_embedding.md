# Cramming through a shared SPS prediction embedding

## Hypothesis
Replacing the learned ordinary memory token with an existing fixed BOS and
optimizing one shared prediction-input vector may make SPS reconstruction
constraints easier to satisfy. At zero prediction window, ordinary context
states remain independent of this vector, so this removes the context-mediated
control route and isolates conditional readout adaptation.

## Setup
Training function: `scripts.analyze_sps_predict_embedding.probe`, specialized
`PredictEmbeddingTrainer`. Its `experiment_configs` registry and imported
training settings define all hyperparameters, checkpoint, initializations,
cohort, budget and matched saved memory controls. The experiment changes input
predict embeddings; the tied output-head row is frozen. No vocabulary is added.
The BOS is resolved from the tokenizer and verified against the checkpoint.

The native initialization starts from the model's predict row; the random
control retains the earlier memory initialization. Both layouts have one learned
hidden-size vector per text. BOS is not counted as a reconstructed target token.
Training and generation use the same shared vector, ordinary history and native
zero-window attention. Specialized artifacts label the learned site explicitly.

## Results
Submitted from snapshot `9d20b38a8ab842e2dcb97df21cff373c8b99f54a`.
All three jobs completed successfully; their durable Arkhip wait resolved.
Implementation passed 48 focused tests and independent adversarial review.
The real BF16 checkpoint passed a short train/save/replay/generation smoke;
this smoke is not a capacity measurement. Executable dry payloads and isolated
staged-checkout path resolution were verified. Baseline settings and availability
were independently checked.

The completed analysis is produced by
`scripts/analyze_sps_predict_embedding.py --phase summary` and saved under
`artifacts/sps_cramming/predict_embedding_20261022/`. The plot was rendered and
visually inspected. A separate audit matched every reported horizon to the
deepest converged raw stage, all target IDs to the original pinned cache, and
the native initialization to the BF16 checkpoint predict row. It also checked
artifact roles, vector dimensions, saved objective/budget fields and source
commits, and recomputed generation metrics directly from saved token IDs.
Additional audit outputs are `independent_results_audit.json` and
`per_sample_comparison.csv` in that directory.

| SPS control site / objective | Mean teacher-forced length | Mean initial correct greedy prefix | Exact greedy reconstruction |
|---|---:|---:|---:|
| Ordinary memory / CE control | 19.42 | 19.06 | 46/48 |
| Shared predict / CE, native initialization | 64.17 | 56.52 | 37/48 |
| Ordinary memory / strict hinge control | 31.52 | 30.56 | 45/48 |
| Shared predict / strict hinge, native initialization | 132.10 | 110.23 | 38/48 |
| Shared predict / strict hinge, random initialization | 130.23 | 118.65 | 43/48 |

Every selected prefix replays exactly under teacher forcing. Each of the three
predict arms exceeds its ordinary-memory control's teacher-forced horizon on
all 48 texts. Paired-text gains are 44.75 tokens for CE (95% bootstrap interval
41.50–47.98), 100.58 for native hinge (96.46–104.60), and 98.71 for random hinge
(93.00–103.96). The random hinge arm gains 88.08 initial correct greedy tokens
(77.27–97.75). These intervals describe text variation at fixed checkpoints and
training seed, not seed variance.

Native-minus-random hinge horizon is 1.88 tokens, with interval -2.23–6.04;
the gain therefore does not depend on starting from the pretrained predict row
in these runs. No initialization winner is established. The total optimization
budget was reached on 47/48 native-hinge and 45/48 random-hinge texts, versus
none of the CE texts. These are achieved lengths under the registered protocol,
not estimates of a global capacity ceiling. Greedy prefix lengths refer to the
single selected final vector per text, not a search over earlier saved stages.

The follow-up saved-stage cost analysis is in `optimization_steps/` under the
same output directory: executable read-only `analysis.py`, `per_stage.csv`,
`summary.json` and `steps_by_token_position.png`. Run selection is derived from
the existing objective and predict registries. It reads cumulative optimizer
update counters and differences adjacent stages, including zero-update stages.
Successful-stage costs and unsuccessful attempts are kept distinct; failures
remain in the CSV. The first stage starts at a zero counter. An independent
pandas calculation verified all 22,636 stages across the seven compared runs,
their convergence flags, cumulative telescoping and every per-position median
and sample count. The rendered comparison plot was visually inspected.

For native-initialized predict/hinge, median incremental costs at token positions
10, 25, 50, 75, 100 and 125 are 17, 33.5, 40, 70, 52 and 71 optimizer updates.
Median cumulative costs at those positions are 179, 529.5, 1269, 3316.5, 6344
and 8476 updates. Cohort sizes are 48, 48, 48, 48, 47 and 38 respectively.
The increase is noisy rather than monotonic: at position 75, mean incremental
cost is 162.5 and the 90th percentile is 425.6, versus median 70. All 48 texts
have positive within-text position/cost Spearman correlation (median 0.557);
random initialization gives a similar pattern (median 0.552).

Holding the full native-predict cohort through position 75 avoids survivor
selection over that range. Pooled successful-stage median costs grow from 18
updates in positions 1–10, to 24 in 11–25, 32 in 26–50 and 46 in 51–75.
Zero-cost stages mean the extended prefix already satisfies the convergence
criterion; they are not dropped. Late cost curves condition on surviving texts,
and the plot suppresses cost estimates with fewer than eight texts while
showing the full survival counts separately. These are costs of reoptimizing
the entire prefix with inherited optimizer state and the running LR schedule,
not isolated-token difficulty or wall-clock costs, and they do not identify
length as the sole causal driver.

Submitted jobs:

- `predict_ce_native`: `lm-mpi-job-f933dc8a-58dd-49e3-a511-48b1c7875b07`.
- `predict_hinge_native`: `lm-mpi-job-425f508b-7c9b-451d-93a0-79d491821122`.
- `predict_hinge_random`: `lm-mpi-job-caff9877-72a5-4bee-9224-3e1e7cb53103`.

## Conclusions
The user's proposed site change gives a large gain on this same frozen SPS
checkpoint: approximately 3.3 times the CE teacher-forced length and 4.2 times
the strict-hinge length, with the same per-text parameter count. The ordinary
memory site's earlier disadvantage is therefore not a ceiling on what this
checkpoint can reconstruct from one learned vector. This is a different
conditioning protocol: the shared vector is injected at every prediction
position, giving direct control of the prediction residual stream and its
queries while ordinary context states stay independent of it. Equal parameter
count does not mean equal intervention locations or computational influence.
The result supports the usefulness of SPS's separated prediction stream as a
place to condition reconstruction; it does not isolate a single causal route
within that stream or establish an architecture-only comparison with Standard.

Exact free generation is still less reliable than teacher replay at the longer
selected lengths. The completed larger-budget native-hinge follow-up is in
`sps_predict_native_shards.md`: its coupled budget/schedule change gives shorter
horizons and stops at per-stage failures before using the extra total budget.
Retaining the original LR trajectory while extending available updates is a
cleaner next control. Neither a capacity ceiling nor a guaranteed lossless
greedy horizon has been measured here.

## Changelog
- 2026-10-22: Designed the fixed-BOS/shared-predict comparison with matched per-text parameter count.

- 2026-10-22: Submitted all three comparisons from a reviewed snapshot; registered durable observation.

- 2026-10-22: All three jobs completed; audited raw stages and exact token IDs, measured paired gains and greedy prefixes, inspected the plot and documented budget censoring.

- 2026-10-22: Measured optimizer cost versus token position from saved stages; independently verified all counter differences and aggregates, retained zero-cost and failed stages, and plotted marginal/cumulative cost with surviving-cohort counts.

- 2026-10-23: Linked completed native-hinge larger-budget shard results and retained the LR-schedule confound explicitly.
