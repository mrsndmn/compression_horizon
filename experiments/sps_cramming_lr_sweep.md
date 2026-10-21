# Native Standard/SPS learning-rate sweep

## Hypothesis
The two checkpoints may require different memory-optimization step sizes.
Changing the learning rate alone may improve their progressive horizons.

## Setup
Training function: `scripts.activation_distillation`, `ProgressiveCrammingTrainer`.
The registry `lr_sweep_experiment_configs` in `scripts/jobs/run_jobs_sps.py`
defines the paired sweep and references completed baseline runs.
Launch with `--suite lr_sweep`; use `--dry` to inspect commands.
The existing post-training audit reports teacher replay and greedy reconstruction.

## Results
All eight jobs completed successfully from code snapshot
`01e077006ad2913699a7556a41992dd46ebfdd46`. Each writes `single_sample_validation.json`
before marking completion.

- `standard_pilot_1b_batch1_lr0.01_92f4945947e8`: `lm-mpi-job-787b43fd-1e0c-4712-bde4-7becb30fd34a`.
- `standard_pilot_1b_batch1_lr0.03_48c68c553278`: `lm-mpi-job-d08a431e-30ac-4def-841e-849097ca5647`.
- `standard_pilot_1b_batch1_lr0.3_0f4b7f53eb25`: `lm-mpi-job-3e0d25e9-f3a0-4a45-9c4a-9866d3f1ee82`.
- `standard_pilot_1b_batch1_lr1_f1b059bcd15d`: `lm-mpi-job-dc171702-5005-4dc8-89d7-32402fde0044`.
- `sps_w0_pilot_1b_batch1_lr0.01_c6217b05e789`: `lm-mpi-job-7d5a8fdc-c56e-4cfb-8ff0-014ba8510a56`.
- `sps_w0_pilot_1b_batch1_lr0.03_3163ddb89ee6`: `lm-mpi-job-fa496697-b32a-45bc-9664-4a47fb029b7f`.
- `sps_w0_pilot_1b_batch1_lr0.3_f59ca78c7aa7`: `lm-mpi-job-47bf235f-dd30-49dc-be3f-df589d08e45a`.
- `sps_w0_pilot_1b_batch1_lr1_3866fc9af9f5`: `lm-mpi-job-aa1d4c07-cd46-4623-8629-518ff24d6a52`.

The durable Arkhip wait resolved with all eight jobs completed. The
implementation passed independent adversarial review and focused tests.
Completed baselines were reused. All runs have matched sample IDs, source-token
prefixes and training settings apart from the swept learning rate.

Mean teacher-forcing horizon (tokens):

| Model | LR 0.01 | LR 0.03 | LR 0.1 baseline | LR 0.3 | LR 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 28.8542 | 29.7500 | 32.2292 | 32.3125 | 32.7708 |
| SPS w=0 | 11.7708 | 14.1042 | 17.0000 | 19.4167 | 19.2917 |

Exact greedy reconstruction at each run's selected teacher-forcing horizon:

| Model | LR 0.01 | LR 0.03 | LR 0.1 baseline | LR 0.3 | LR 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 48/48 | 48/48 | 47/48 | 48/48 | 47/48 |
| SPS w=0 | 42/48 | 45/48 | 43/48 | 46/48 | 48/48 |

All 480 selected memories replay exactly under teacher forcing. The old baseline
float-mean reporting issue is corrected by recovering integer token counts,
without changing original run files. Greedy success counts refer to different
prefix lengths across runs and are not fixed-length generation comparisons.

For SPS, the best observed mean horizon improves by 2.4167 tokens (14.22%)
relative to its baseline. That variant improves 34 samples, ties 9, and declines
on 5. The two largest tested rates differ by just 0.125 tokens: paired wins,
ties and losses are 15/18/15, and a paired percentile bootstrap interval for their
mean difference is [-0.75, 1.02] tokens. This estimates sample variability only;
initialization-seed variability and selection effects were not measured.

The best observed Standard mean improves by 0.5417 tokens (1.68%).
Derived JSON and CSV reports are in
`artifacts/sps_cramming/lr_sweep_comparison_20261022.{json,csv}`;
original per-run audits and design files are preserved.

## Conclusions
The original rate was suboptimal for SPS in this tested grid, but tuning it
does not close the gap to Standard. Lower rates worsen both models' mean
horizons. The two largest tested rates are similar for SPS; do not claim a
resolved optimum from the small difference. The higher one achieves exact
greedy decoding for every selected SPS prefix in this run. Follow-up candidates
are repeated memory initialization seeds and scheduler comparisons; none were
launched as part of this completed sweep.

## Changelog
- 2026-10-21: Added paired learning-rate-only sweep reusing completed baselines.

- 2026-10-21: Submitted the complete sweep and registered durable observation.

- 2026-10-22: Verified all completed runs, recorded horizon/greedy results and paired comparisons.
