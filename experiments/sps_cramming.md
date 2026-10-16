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
Pending.

## Conclusions
Compare reconstruction convergence after both jobs complete.

## Changelog
- 2026-10-16: Added the native Standard/SPS comparison and checkpoint adapter.
