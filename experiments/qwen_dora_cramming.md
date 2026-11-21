# Progressive cramming of the merged Qwen DoRA checkpoint

## Hypothesis

The adapter-trained SPS checkpoint may change the teacher-forced horizon and
the length of the initial correct greedy prefix on the audited Qwen documents.

## Setup

Training function: `scripts.pretrained_sps_worker.train_shard`. The registered
experiment is `qwen_dora_standard_1e_4` in `scripts/pretrained_sps_experiments.py`.

## Results

Pending.

## Conclusions

Compare the audited results by document with the earlier Qwen SPS-CE runs.

## Changelog

- 2026-11-21: Register the merged checkpoint for shared-prediction cramming.
