# Pretrained SPS cramming

The interactive interview accepted support/validation, matched original-AR
comparisons, and reproducible integration into `sps_paper/`. The executable design
is `scripts/pretrained_sps_experiments.py`. The interview record is local at
`run/deep-interview/deep-interview-pretrained-sps-cramming.md`.

- [x] Verify actual checkpoint families/revisions and existing BOS identities.
- [x] Implement differentiable native-backbone SPS interface and reference tests.
- [x] Define immutable, deduplicated preflight/main launch routes and wave limits.
- [x] Pass focused/full tests and independent review after addressing findings.
- [x] Pass real-checkpoint GPU preflights (both complete; all six variants checked).
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
# Audit a complete model cohort before its comparison is published:
python -m scripts.analyze_pretrained_sps --experiment qwen_ar
```

`advance` submits the next incomplete model's shards; it cannot bypass missing
preflights or duplicate recorded submissions. Do not run a local waiting shell.
The final article update follows completed measurement, not submission status.
The cohort auditor rejects incomplete/duplicate or inconsistent shard evidence
and exports scalar per-text metrics without new model inference.

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

## Completed Qwen AR wave

The preflight group resolved with both jobs completed; both output markers and
fingerprints were audited. All six upstream-reference logit errors are zero.
The bounded smoke calls did not converge within their short training budget and
are not reported as reconstruction-capacity measurements.

Original Qwen AR was submitted from snapshot
`bde700240cac62d5b5e3d6fe1691da79930d136d`. Full shard IDs are recorded in the
experiment Markdown and external job manifests. The source-vs-staged fingerprint
and all submitted offsets, complete UUIDs and resource payloads were verified.
Resolved durable group: `17b4d1437baf49d2b994784f35f02198`;
key: `pretrained-sps-qwen-ar-9603ad849360-20261105`.

Its successful outcome and complete saved sample set were audited below. Do not
repeat either completed preflight or this completed model wave. Continue through
all registered model waves before the final paired analysis and article integration.

## Completed Qwen AR and SPS-CE audits

The original Qwen AR durable group resolved with all jobs completed. Reusable
`analyze_pretrained_sps` audits the full cohort against source revisions, code
fingerprints/archives, family-native data identities, raw stages, registered
budgets and exact saved target/generated IDs. Qwen AR results are recorded in
`experiments/pretrained_sps_cramming.md` and external JSON/CSV. All texts exhausted
the total budget. Teacher-forcing success does not imply exact greedy generation.

Completed main wave: Qwen SPS-CE, submitted from
`d30039f3c7ff19f1fd2b829b3f3c91c5e4732980`.
Resolved durable group: `a5f9a4cc7568462c8450dea4bbc7f477`;
key: `pretrained-sps-qwen-ce-9603ad849360-20261105`.
The auditor with `--experiment qwen_ce` passed for all expected texts and saved
stages; its JSON/CSV outputs are complete. A separate CSV check confirmed identical
ordered document hashes across AR and CE. Both exhaust their total budgets;
teacher-forced and greedy results are recorded separately in the experiment file.
The next registered model is Qwen SPS-KD. The remaining paired analysis and
article integration still wait for all requested cohorts.

The cohort auditor and its stopping-boundary fixes passed 61 focused tests across
analysis, pretrained-model compatibility and existing SPS-paper replay. A fresh
independent adversarial review returned `APPROVE` after checking the real complete
cohort, JSON/CSV consistency and refusal to export missing inputs. The committed
auditor is `a95c70284a0b399b8ed8bef6e99f00ec61380651`; the training fingerprint
remains unchanged. This verification performs no new inference or optimization.

Operational note: the inherited HTTP proxy timed out before authentication when
submitting the next wave. Direct verified TLS and HTTP to `api.ai.cloud.ru` work.
Use the normal launcher with `NO_PROXY=api.ai.cloud.ru` in this environment. No
job submission intents or job records were created by either failed auth attempt.

## Active Qwen SPS-KD wave

Direct API access succeeded. All registered Qwen SPS-KD shards were submitted
from `545de7f9d3a399eb1b9c864197d3aa373b890642`; the complete job UUIDs are in
the experiment file and external manifests. Their payloads, offsets, unique IDs,
source commits and archived training fingerprints were audited.

Active durable group: `4f8b9d015b874a51bc106086f8f5a4c6`;
key: `pretrained-sps-qwen-kd-9603ad849360-20261105`.
On successful outcome, audit `--experiment qwen_kd`, then use the normal launcher
with the proxy exception above to advance to original SmolLM AR. Register a NEW
durable group for the returned jobs. Do not repeat either completed Qwen cohort
or the preflights. Full paired plots and article integration remain pending the
complete registered comparison.
