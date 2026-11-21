# Progressive cramming of the merged Qwen DoRA checkpoint

## Hypothesis

The adapter-trained SPS checkpoint may change the teacher-forced horizon and
the length of the initial correct greedy prefix on the audited Qwen documents.

## Setup

Training function: `scripts.pretrained_sps_worker.train_shard`. The registered
experiment is `qwen_dora_standard_1e_4` in `scripts/pretrained_sps_experiments.py`.

## Results

MLS accepted the checkpoint-specific GPU preflight from `fbffd06` as
`lm-mpi-job-46f835e8-6fc0-4e74-b905-f882b4d6e5b7`. Cohort results are pending.

## Conclusions

Compare the audited results by document with the earlier Qwen SPS-CE runs.

## Changelog

- 2026-11-21: Register the merged checkpoint for shared-prediction cramming.
