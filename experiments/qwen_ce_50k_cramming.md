# Progressive cramming after extended Qwen SPS-CE adaptation

## Hypothesis

The requested final checkpoint may recover longer prefixes than the previously
audited Qwen SPS-CE checkpoint under the same cramming protocol. The primary
comparison is the paired change in teacher-forced horizon. Greedy decoding
tests how much of that prefix can be recovered from the saved vector alone.

## Setup

Training function: `scripts.pretrained_sps_worker.train_shard`.

1. Add the requested checkpoint as a separate entry in the executable registry
   in `scripts/pretrained_sps_experiments.py`. Give the run its own output
   location and target it explicitly in the launcher. Reuse the audited Qwen
   protocol and verify the ordered document hashes and token IDs against the
   archived Qwen cohort before submission.
2. Run a GPU preflight for this entry. Verify source and revision, load the SPS
   state strictly, and check reference logits and gradients with `parity`.
   Exercise optimization with frozen model weights, saving, replay and greedy
   decoding on one text. Proceed only if these checks pass.
3. Submit the full cohort through MLS and record the checkpoint identity, code
   revision and job IDs. Verify that completed shards cover each text once.
4. Run `scripts.analyze_pretrained_sps.audit_experiment` and verify exact
   teacher-forced replay of each selected prefix. Pair results by document with
   the existing Qwen SPS-CE audit. Report mean and median paired horizon change,
   mean initial correct greedy prefix, exact greedy reconstruction count and
   budget exhaustion. Use the existing original Qwen AR audit for context.

## Results

The checkpoint metadata matches the registered Qwen source and revision. The
ordered document hashes and digest of target IDs and attention masks match the
archived Qwen SPS-CE shard reports. The checkpoint-specific GPU preflight was
submitted from `e1c08f8` as `lm-mpi-job-f0e7c1a9-757e-415b-922f-43ed7d4e3e96`.
Reconstruction results are pending.

## Conclusions

A positive mean paired horizon change would support the hypothesis for this
cohort and budget. Report greedy recovery separately so that any discrepancy
between teacher-forced and free-running reconstruction is visible.

## Changelog

- 2026-11-20: Plan the paired comparison and checkpoint-specific preflight.
- 2026-11-20: Verify paired inputs and submit the targeted GPU preflight.
