# Progressive cramming of the merged Qwen DoRA checkpoint

## Hypothesis

The adapter-trained SPS checkpoint may change the teacher-forced horizon and
the length of the initial correct greedy prefix on the audited Qwen documents.

## Setup

Training function: `scripts.pretrained_sps_worker.train_shard`. The registered
experiment is `qwen_dora_standard_1e_4` in `scripts/pretrained_sps_experiments.py`.

## Results

MLS accepted the checkpoint-specific GPU preflight from `fbffd06` as
`lm-mpi-job-46f835e8-6fc0-4e74-b905-f882b4d6e5b7`. The preflight passed with zero maximum logit error.
MLS accepted the full cohort from `eab916b`. The jobs in shard order are

- `lm-mpi-job-e34cee5e-af9e-4b3c-9edd-37980a379995`
- `lm-mpi-job-57093fb2-939d-4e06-954a-e343ea9e6218`
- `lm-mpi-job-841e375c-16c6-43a3-8f8d-45ff08344c43`
- `lm-mpi-job-893650a6-c6c6-4e96-9173-e88f8b89bde1`
- `lm-mpi-job-a12df99c-b810-43e2-b2f5-cd3f09b809a4`
- `lm-mpi-job-2ecbc9c0-6db3-4687-986e-bd5882b7d0de`
- `lm-mpi-job-23f110bd-0326-46db-bae4-2a9a61cd44e4`
- `lm-mpi-job-156e8e53-38a4-4604-9fea-9fd5c6bb4b86`

All eight shards completed. The audit confirmed exact teacher-forced replay of
the selected prefix on all 48 texts. All 48 exhausted the total update budget.
The mean horizon was 265.98 tokens (median 265.5). The mean initial correct
greedy prefix was 155.125 tokens, with exact greedy recovery on 12/48 texts.

Pairing by document hash, DoRA gained 55.58 horizon tokens on average over
SPS-CE 50k, with gains on every text (median difference +43). Its mean greedy
prefix was 4.04 tokens shorter, with gains on 25/48 texts. Against the earlier
SPS-CE checkpoint, DoRA lost 14.04 horizon tokens on average and gained 48.17
greedy tokens, with greedy gains on 34/48 texts. The per-document comparison
is saved alongside the audit in `analysis/7559610b05d5/qwen_dora_standard_1e_4/`.

## Conclusions

The DoRA checkpoint recovered longer teacher-forced prefixes than SPS-CE 50k
at the tested budget. Their mean free-running prefix lengths were close on
this cohort. Exact greedy counts score each checkpoint at its own selected
prefix length.

## Changelog

- 2026-11-21: Register the merged checkpoint for shared-prediction cramming.
- 2026-11-21: Audit the completed cohort and compare both SPS-CE checkpoints by document.
