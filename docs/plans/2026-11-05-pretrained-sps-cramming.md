# Pretrained SPS cramming

The interactive interview accepted support/validation, matched original-AR
comparisons, and reproducible integration into `sps_paper/`. The executable design
is `scripts/pretrained_sps_experiments.py`. The interview record is local at
`run/deep-interview/deep-interview-pretrained-sps-cramming.md`.

- [x] Verify actual checkpoint families/revisions and existing BOS identities.
- [x] Implement differentiable native-backbone SPS interface and reference tests.
- [x] Define immutable, deduplicated preflight/main launch routes and wave limits.
- [ ] Pass focused/full tests, independent review and real-checkpoint preflights.
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
