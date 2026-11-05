# Cramming execution policy

User instruction (2026-10-21): always launch per-sample cramming with
`per_device_train_batch_size=1`, including full, progressive, prefix-tuning,
low-dimensional cramming and future memory-optimization diagnostics. Preserve
this across sessions and models. Shared compression-head/model-weight training
is a separate task. Historical snapshots and already-submitted jobs retain
their original settings.

User instruction (2026-10-22): future SPS shared-predict cramming uses hinge
and native initialization only. Run the matched cohort as independent single-GPU
shards with batch one, then merge the complete cohort; legacy CE/random registries
remain for provenance and reproduction, not the current launch plan.

# Artifact storage policy

User instruction (2026-11-05): never relocate experiment artifacts to another
volume on your own when jobs fail because storage is full. Wait for storage
cleanup; do not requeue affected jobs while the storage condition remains.

The already-running pretrained SmolLM AR retry may finish on NFS1. Once all its
current jobs are terminal and no workers are writing this artifact tree, return
the series artifacts to their original physical NFS2 location. This return is
explicitly authorized. Verify the copy before switching paths or removing the
NFS1 copy. If NFS2 has insufficient space, wait for cleanup. Complete this return
before launching later model waves. Future relocation requires an explicit user
instruction; lack of space does not authorize it.

# Project conventions

## Building the paper

The separate SPS working preprint is in `sps_paper/`. Build it with
`make sps-paper`: this regenerates its tables/figures from versioned evidence,
checks source references and compiles `sps_paper/build/main.pdf`.
`make sps-paper-figures`, `make sps-paper-check` and `make sps-paper-clean`
operate only on this article. Original-paper commands and cohort rules below
remain scoped to `paper/`. The new article has its own matched SPS cohort.

`sps_paper/reproduce.py render` consumes versioned evidence without a GPU or
private mounts; `export` refreshes evidence from completed shared artifacts;
`plan` describes historical experiment replay and external prerequisites without
submitting jobs. Generated `sps_paper/tables/*.tex` and statistical figure PDFs
must be changed through that generator, never by hand. The manuscript explicitly
separates teacher forcing, greedy prefix lengths and exact generation counts.

Build the compiled PDF with **`make paper`** (runs `paper/lint_paper.py` first via
`paper-check`, then `latexmk -pdf` from inside `paper/`). Output lands at
`paper/build/example_paper.pdf`. Related targets: `make paper-check` (lint only),
`make paper-clean`, and `make tables` (regenerate tables from artifacts).

Figures are resolved by `latexmk` via `paper/.latexmkrc`, which puts `./figures//`
(recursive) on `TEXINPUTS`, so `\includegraphics` uses **bare filenames** and every
figure filename under `paper/figures/` must be unique (avoid same-named files in
subdirs — recursive search makes them ambiguous). `lint_paper.py` also fails on any
unreferenced file under `figures/`.

## ⚠️ Generated paper tables — DO NOT hand-edit

The `.tex` files in `paper/tables/` are **generated** by `scripts/paper/tables/`
(see `paper/tables/tables.sh` and the `make tables` target). They are
**overwritten on every regeneration**, so any in-place edit is silently lost.

- To change a generated table, edit its **generator** (the registry/spec in
  `scripts/paper/tables/progressive.py` or the relevant script), then regenerate:
  `python scripts/paper/tables/progressive.py --name <tab:label> --save` (or `make tables`).
- If a cell is wrong (e.g. `nan`), the cause is usually upstream data/cache, not
  the `.tex` — fix it there and regenerate.
- **`paper/tables/manual/` is the only hand-editable table directory.**

A `PreToolUse` hook (`.claude/hooks/block_paper_tables_edit.py`, wired in
`.claude/settings.json`) enforces this: `Edit`/`Write`/`NotebookEdit` to
`paper/tables/` (outside `manual/`) are denied. Regeneration via scripts
(`Bash`) is unaffected.

`paper/lint_paper.py` also fails if any rendered table still contains a `nan`.

## ⚠️ No raw tables in the paper body

The main text (`paper/example_paper.tex`) and appendix (`paper/appendix.tex`)
**must not contain raw `tabular` environments**. Every table's content lives in
its own file under `paper/tables/` (generated) or `paper/tables/manual/`
(hand-edited), and the body pulls it in with `\input`:

```latex
\begin{table}[t]
  \centering
  \caption{...}
  \label{tab:...}
  \begin{small}
  \input{tables/manual/<name>}   % the .tex file holds only the tabular
  \end{small}
\end{table}
```

`paper/lint_paper.py` enforces this via the `raw-tables` check, which fails if a
`\begin{tabular...}`/`\begin{longtable}` appears in the body sources
(`BODY_TEX`). `rebuttle.tex` is a standalone reviewer-response document, not part
of the compiled paper, so it is exempt.

## Building the poster

Source: `poster/poster.html`; exact portrait canvas is 594 × 840 mm.
Run `python poster/build_poster.py` (or pass `--ghostscript /path/to/gs`).
Dependencies and print checks are described in `poster/README.md`.
Outputs: CMYK FOGRA39 `poster/poster.pdf`, identical `poster/poster_preview.pdf`,
PNG preview, and `poster/print_report.json`. All assets render offline.

## Usage arms

- `scripts/activation_distillation.py` → `compression_horizon.train` full/progressive/low-dim/prefix/head trainers; `tests/test_cramming_batch_policy.py`.
- `scripts/analyze_standard_additive.py --phase dry|submit|probe|merge` → additive progressive trainer; `tests/test_standard_additive.py`.
- `scripts/analyze_standard_additive_results.py [--probe]` → paired posthoc analysis and fixed-vector CPU probes.
- `make sps-paper[-figures|-check|-clean]` → `sps_paper/reproduce.py`, `check.py`, `main.tex`; article tests.
- `sps_paper/reproduce.py export|render|plan` → completed evidence, offline assets and historical replay; `tests/test_sps_paper_reproduce.py`.
- `make paper[-check|-clean]`, `make tables` → original `paper/` build, lint and table generators.
- `poster/build_poster.py` → `poster/poster.html`, profiles/fonts/images and PDF renderer; `poster/README.md`.
- `.github/workflows/run-tests.yml`, `lint.yml` → repository pytest and Ruff checks.

## Native SPS usage arms

- `scripts/jobs/run_jobs_sps.py --suite batch1|lr_sweep|objectives [--dry]` → native checkpoint snapshot launcher; `tests/test_sps_launcher.py`.
- `models.sps_causal_lm.load_sps_checkpoint/load_sps_tokenizer` → native pilot adapters and vendored kernels; `tests/test_sps_causal_lm.py`.
- `inference.generation.generate_from_compression` → ordinary-history decoding and optional predict override; `tests/test_sps_generation_semantics.py`.
- `scripts/validation/validate_sps_port.py` → upstream real-checkpoint logit/gradient parity and long-context smoke.
- `scripts/validation/validate_sps_generation.py` → native generation reserved IDs and numerical-shape audit.
- `scripts/validation/validate_sps_batch1_run.py --run-dir` → saved exact-ID teacher/greedy replay; `check_sample`.
- `scripts/analyze_sps_cramming.py --phase summary|probe|plot|submit|dry` → native diagnostic registry, immutable submit helper and first-failure analysis.
- `scripts/analyze_sps_memory_ablation.py --phase dry|submit|probe|plot` → per-layer memory attention removal; `tests/test_sps_memory_ablation.py`.
- `scripts/analyze_sps_memory_routes.py --phase dry|submit|probe|plot` → route/KV interventions; `analysis.native_memory_interventions`.
- `scripts/analyze_sps_margin_geometry.py --phase dry|submit|probe|plot` → first-failure Jacobian geometry; `tests/test_sps_margin_geometry.py`.
- `scripts/analyze_sps_margin_rescue.py --phase dry|submit|probe|plot` → restart/controller diagnostics; `tests/test_sps_margin_rescue.py`.
- `scripts/audit_sps_objective_gradients.py --phase dry|submit|probe|plot` → saved-vector gradient audit; `tests/test_objective_gradient_audit.py`.
- `scripts/analyze_sps_objective_factorial.py` → completed objective comparison and bootstrap; `tests/test_objective_factorial_analysis.py`.
- `scripts/analyze_sps_predict_embedding.py --suite baseline|budget --phase dry|submit|probe|summary` → `PredictEmbeddingTrainer`; predict tests.
- `scripts/analyze_sps_predict_shards.py --suite budget|fixed_schedule --phase dry|submit|probe|merge` → native hinge shards; shard/optimization tests.
- `train.loss` squared-margin objective → `BaseTrainer.compute_loss/compute_convergence`; `tests/test_squared_margin_objective.py`.
- `analysis.information_gain.compute_information_gain` → embedding-based native/HF memory-context analysis.

## Pretrained SPS usage arms

- `scripts/run_pretrained_sps_cramming.py --phase dry|prepare` → `pretrained_sps_experiments.py` registry and family-native pinned data.
- `scripts/run_pretrained_sps_cramming.py --phase preflight|advance` → immutable single-GPU snapshot/wave submission.
- `scripts/run_pretrained_sps_cramming.py --phase probe` → `pretrained_sps_worker.py` reference parity, token-only AR and shared-predict trainers.
- `models.pretrained_sps.load_pretrained_sps/load_pretrained_tokenizer` → pinned Llama/Qwen3 backbones and strict adapted-state loading; `tests/test_pretrained_sps.py`.
- `python -m scripts.analyze_pretrained_sps --experiment <name>` → completed-cohort raw-stage/target/provenance audit and JSON/CSV; `tests/test_pretrained_sps_analysis.py`.
