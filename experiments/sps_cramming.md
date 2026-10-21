# Standard versus SPS cramming

## Hypothesis
The persistent input stream and temporary prediction stream of SPS may change
how much text can be recovered from an optimized memory embedding.

## Setup
Training function: `scripts.activation_distillation`, `ProgressiveCrammingTrainer`.
The experiment registry and launch settings live in `scripts/jobs/run_jobs_sps.py`.
Native checkpoint loading and SPS attention semantics live in
`compression_horizon.models.sps_causal_lm`.

## Results
Port validation passed: logits matched upstream argmax, input-gradient cosine
similarity exceeded 0.99996 for both checkpoints, and full-context backward
checks fit on one A100. Native trainer, existing cramming, information-gain and
launcher tests passed. Independent adversarial review approved the final fixes.

Training submitted from commit `2683216db54e0ae8f51a69fa467c2dc07f5509f9`:
- Standard: `lm-mpi-job-e1447677-1cba-4b9f-b022-93fa6b7c1dce`.
- SPS: `lm-mpi-job-818d7b80-9a5f-480c-9acb-db17b07941dd`.

Both runs completed successfully on 2026-10-16. On 48 matched texts, mean
exact-reconstruction horizons were 32.0625 tokens for Standard and 17.1667 for
SPS. Standard won on every pair. Submission records, saved embeddings and
validation logs are under `artifacts/sps_cramming/`.

## Conclusions
Standard reaches longer exactly reconstructed prefixes under the current
optimization protocol. Mechanism diagnostics are described in
`sps_cramming_diagnostics.md`; the horizon should not be read as a proven
representational capacity limit.

## Changelog
- 2026-10-16: Added the native Standard/SPS comparison and checkpoint adapter.
- 2026-10-16: Validated upstream parity, fixed review findings, submitted both jobs.
- 2026-10-21: Recorded completed results and linked mechanism diagnostics.
