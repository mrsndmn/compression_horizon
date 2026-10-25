# Standard cramming with a shared additive input vector

## Hypothesis

Repeated injection of the same learned vector may increase reconstruction
length without adding trainable scalars. Unlike SPS shared prediction, adding
the vector to ordinary inputs also changes context representations.

## Setup

`AdditiveCrammingTrainer` in `scripts/analyze_standard_additive.py` implements
the requested input sequence: one learned prefix vector followed by ordinary
token embeddings with that same vector added. Training, teacher replay and
greedy generation use the same conditioning function and ordinary causal mask.
The Python `experiment_configs` registry and `resolved_training_args` are the
design authority. They reuse the completed Standard strict-hinge control,
including exact cached texts and the saved initial vector for each text.

Use `--phase dry` to inspect the immutable MLS payloads, `--phase submit` to
submit registered shards, and `--phase merge` after successful completion.
The merge checks source snapshots, settings, raw stages, sample IDs, target IDs,
paired initialization and replay metrics before computing paired comparisons.
Artifacts live under the shared SPS artifact root in `standard_additive_20261025`.

## Results

All 308 repository tests pass; the independent portable adversarial review
returned APPROVE. A real native Standard checkpoint passes BF16 forward/backward
with a finite nonzero vector gradient and frozen model weights. All paired
source-control initializations and exact cached targets load successfully.
Compatibility evidence is saved as `standard_additive_20261025/preflight.json`.

All eight registered shards were submitted from immutable training source
`e200a381abb6f4f7cd7e64bdbed3cd2515f40ec3`:

- `lm-mpi-job-302e982d-7b7d-4ee1-8cbf-932bc818cd3b`
- `lm-mpi-job-fd5adfe2-fda5-4b2d-bf43-33b1cefdfa3b`
- `lm-mpi-job-a5456fb6-0155-4162-93bf-04aff0506a68`
- `lm-mpi-job-1efc8715-3e2c-44cc-afa9-356c1ab3a346`
- `lm-mpi-job-d3657da0-9ca8-4c20-8bdd-885a89893108`
- `lm-mpi-job-9a4f2c0f-6843-4c68-85ab-732684a270b1`
- `lm-mpi-job-dd70d473-e93e-41c2-afcb-9ee603951dd9`
- `lm-mpi-job-c1fe52f1-0103-4b1e-987f-5b40e6c97237`

The submission manifest is saved beside the shards. Arkhip durable wait group
`d0949dd7ebac4fb584146457e71d7e31`, key
`standard-additive-hinge-batch1-8x6-e200a38-20261025`, watches all eight jobs.
No reconstruction result is claimed before successful completion and merge.
On resolution run `python scripts/analyze_standard_additive.py --phase merge`
and report the paired comparison with the original Standard control. Failures
require inspecting their logs and registering a new group for remaining jobs;
do not repeat completed submissions.

## Conclusions

The primary comparison changes the conditioning interface within the same
Standard checkpoint. A gain would test repeated access to the shared vector;
it would not establish SPS-style independence of ordinary context states.

## Changelog

- 2026-10-25: Implement the user-requested shared additive intervention with
  paired initialization and matched original-budget control.
- 2026-10-25: Pass the full suite, real-checkpoint compatibility preflight and
  independent review; submit immutable shards and register the durable wait.
