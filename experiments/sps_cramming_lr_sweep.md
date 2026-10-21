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
All eight jobs were submitted (initial status Pending) from code snapshot
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

The durable Arkhip wait covers all eight jobs. The implementation passed
independent adversarial review and focused tests; results await completion.

## Conclusions
Compare every variant against its checkpoint's completed baseline on matched
samples. Separate teacher-forcing horizons from free-generation success.

## Changelog
- 2026-10-21: Added paired learning-rate-only sweep reusing completed baselines.

- 2026-10-21: Submitted the complete sweep and registered durable observation.
