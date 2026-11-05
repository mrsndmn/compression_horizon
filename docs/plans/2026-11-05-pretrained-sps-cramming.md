# Pretrained SPS cramming

The interactive interview accepted support/validation, matched original-AR
comparisons, and reproducible integration into `sps_paper/`. The executable design
is `scripts/pretrained_sps_experiments.py`. The interview record is local at
`run/deep-interview/deep-interview-pretrained-sps-cramming.md`.

- [x] Verify actual checkpoint families/revisions and existing BOS identities.
- [x] Implement differentiable native-backbone SPS interface and reference tests.
- [x] Define immutable, deduplicated preflight/main launch routes and wave limits.
- [x] Pass focused/full tests and independent review after addressing findings.
- [ ] Pass real-checkpoint GPU preflights (submitted; durable observation active).
- [ ] Submit/observe the successive main waves with durable Arkhip groups.
- [ ] Audit/merge complete cohorts, compare within families and plot results.
- [ ] Extend versioned article evidence/replay; regenerate assets and compile PDF.
- [ ] Commit/push results and verify applicable PR checks.

Artifacts live outside this repository under the source registry's absolute
`ROOT`. `data/` contains pinned raw texts and family-native IDs; `preflight/` and
`runs/` are separated by code fingerprint; `code/` contains immutable Git snapshots.
Each successful worker writes its completion marker only after validation.

Run source modules with the existing project Python and `PYTHONPATH=src:.`:

```sh
python -m scripts.run_pretrained_sps_cramming --phase dry
python -m scripts.run_pretrained_sps_cramming --phase dry --kind runs
python -m scripts.run_pretrained_sps_cramming --phase prepare
python -m scripts.run_pretrained_sps_cramming --phase preflight
# Register returned complete job UUIDs with Arkhip job_wait.
# After successful preflight completion, and after each main wave:
python -m scripts.run_pretrained_sps_cramming --phase advance
```

`advance` submits the next incomplete model's shards; it cannot bypass missing
preflights or duplicate recorded submissions. Do not run a local waiting shell.
The final article update follows completed measurement, not submission status.

## Current handoff

Implementation snapshot: `4da4f0f1af82698888ba06ecda7fc5a8a282dacd`.
Code fingerprint: `9603ad8493600a9e5577a3dad2d7abf28c2b2a41cfd241e3bfeb7f87bdcd5a4f`.
Focused tests: 17 passed. Full repository: 325 passed. Full Ruff and both dry
routes pass. Fresh independent review: APPROVE. The prepared cache checks both
IDs and masks plus family/revision/design/raw-document identity. The earlier
cache is preserved as `ROOT/data-schema1-before-review`; raw documents are equal.

Preflight jobs and the durable group are recorded in
`experiments/pretrained_sps_cramming.md`. After the outcome, require both family
completion markers and matching `preflight.json` fingerprints before running
`--phase advance`. Register a new one-shot Arkhip group for the returned complete
job UUIDs. On each main-wave completion, inspect completion evidence and advance
the next registered model; never resubmit a completed shard. Once all model waves
finish, the remaining implementation is the audited merger and article evidence,
plots, tables and replay integration. Do not report submission as measurement.

Branch `feat/pretrained-sps-cramming` is pushed. GitHub repository access and push
work, but creating the PR was rejected with HTTP 403,
`Resource not accessible by personal access token`; the credential helper has no
alternative GitHub credential. Consequently no PR was created and server CI is
unverified. Local checks above passed. No merge or unrelated branch update was
attempted. This provider permission issue does not block the authorized MLS jobs.
