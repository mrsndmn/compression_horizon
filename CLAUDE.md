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

# Project conventions

## Building the paper

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

- Poster editing: `poster/poster.html`, local `poster/images/`, `poster/fonts/`,
  and `poster/profiles/`; preserves English title/authors at the top.
- Poster rendering: `python poster/build_poster.py`; Playwright print emulation
  checks layout before Ghostscript converts RGB to FOGRA39 CMYK. pikepdf embeds
  the output intent; PyMuPDF verifies print dimensions and effective raster DPI.
  The renderer also enforces minimum body/caption/table sizes of 24/20/20 pt.
- Poster delivery/review: `poster/poster.pdf` is the print artifact;
  `poster/poster_preview.pdf` is identical; `poster/poster_preview.png` comes
  from the final PDF. `poster/print_report.json` records the validated dimensions,
  profile, DPI, and layout gaps. There are no training/evaluation/CI callers of
  the poster builder.
- Paper building: `make paper` invokes `paper-check` then `latexmk`;
  `make paper-check`, `make paper-clean`, and `make tables` are independent of
  the poster build. Generated paper tables are inputs for poster claims,
  never edited as part of poster generation.

## Native SPS usage arms

- `scripts/jobs/run_jobs_sps.py --dry` describes the matched native-checkpoint
  progressive cramming pair; without `--dry` it submits MLS jobs from a committed
  snapshot under `artifacts/sps_cramming/code/`. Designs and complete job IDs are
  saved beside each run; `completed` is written only after successful training.
- `scripts/activation_distillation.py --model_checkpoint <native.pt>` loads the
  checkpoint with `models.sps_causal_lm.load_sps_checkpoint` and the GPT-2 tokenizer.
  Existing HF loading paths remain in use for non-native checkpoints.
- Native inference/analysis must explicitly use `load_sps_checkpoint` and
  `load_sps_tokenizer`; these models are not registered with HF AutoModel.
  Both `input_ids` and differentiable `inputs_embeds` produce HF-shaped logits.
  SPS preserves its interleaved prediction stream and temporary-key mask.
- `tests/test_sps_causal_lm.py` covers masking, causal isolation, embedding
  gradients, strict loading and native trainer integration. Vendored source
  provenance and interface limits are in `models/sps_vendor/README.md`.
- `scripts/validation/validate_sps_port.py --source-repo <sps> --checkpoint <native.pt>`
  compares real-checkpoint logits and input gradients against the upstream GPU
  kernels. `--long-smoke` also checks a full-context differentiable forward and
  records peak GPU memory. The source checkout is used only for validation.
- `analysis.information_gain.compute_information_gain` uses embedding forwards
  for both baseline and memory contexts, keeping EOS document policy consistent
  with single-text cramming for native checkpoints and ordinary HF models.
- `tests/test_sps_launcher.py` verifies MLS's single-line shell contract,
  compute-node mount paths, and that the completion marker requires exit zero.
- `scripts/analyze_sps_cramming.py --phase summary|plot` audits paired saved
  stages and plots exact-reconstruction horizons. `--phase probe` runs bounded
  checkpoint diagnostics; `--phase dry|submit` describes/submits that probe from
  a committed snapshot. Its code contains all diagnostic variants. Shared
  artifacts are absolute paths, so worker snapshots read the original caches.
  The probe checks natural-prefix loss, memory sensitivity/attention, and Adam
  restarts at each sample's **first failed stage**, excluding later frozen rows.
- `tests/test_sps_diagnostics.py` checks first-failure selection, the exact loss
  offset/masking against the native trainer, and freezing of converged samples.
- `tests/test_sps_generation_semantics.py` compares interleaved SPS training
  logits and memory gradients with an independent ordinary-history forward
  containing one current prediction slot, and corrupts historical predict
  slots to verify isolation. Both greedy generation paths filter native
  predict/pad/unused vocabulary IDs before feeding generated text back as
  ordinary input embeddings. EOS and existing HF decoding behavior are covered.
- `scripts/validation/validate_sps_generation.py` checks reserved-ID errors at
  the saved first-failure stages, compares real SPS greedy decoding with the
  ordinary-history reference, and holds quantized weights/inputs fixed to
  distinguish BF16 arithmetic changes across batch/context shapes from mask
  semantics. Results go to `diagnostics_20261021/generation_audit.json`.
- `utils.launch.validate_cramming_batch_size` runs immediately after argument
  parsing in `scripts/activation_distillation.py`. It rejects multi-sample
  cramming before model/data/output setup; shared compression-head training is
  exempt. The SPS pair, generic grid, model grid and low-dim launchers use the
  single-sample setting. `tests/test_cramming_batch_policy.py` checks the CLI
  boundary and rendered jobs. The current SPS registry reuses the original
  matched text subset and writes new run identities, preserving earlier results.
  Its explicit `tokenized_dataset_path` pins the old saved token IDs for both
  training and audit; offset/limit selection applies to that saved dataset.
  Stage root and src are both on PYTHONPATH for executable audit imports.
  `scripts/jobs/run_jobs_sps.py --suite lr_sweep [--dry]` selects paired
  `lr_sweep_experiment_configs`, merges per-experiment LR into saved design and
  CLI args, and references completed single-sample baselines without relaunching
  them. Default `--suite batch1` retains the original pair registry.
- The current SPS launcher runs `scripts/validation/validate_sps_batch1_run.py`
  after training and writes `completed` only after that audit exits zero. It
  verifies the complete expected sample set, reuses exact original cached token
  IDs, and records teacher-forcing replay and greedy reconstruction separately
  in `single_sample_validation.json`, always one sample per forward. Early-EOS
  missing tokens count as errors. Teacher replay uses integer correct-token
  counts so floating-point mean reduction cannot misclassify an exact match.
  The audit can also be rerun with `--run-dir`.

- `scripts/analyze_sps_memory_ablation.py --phase dry|submit|probe|plot`
  extends the saved native checkpoint diagnostic flow. It evaluates the matched
  single-sample baseline memories without optimization, using common per-text
  target lengths and exact saved token IDs. Per-layer attention pre-hooks remove
  only key-0 edges from queries >=1, clone shared masks, and restore on exit.
  Single-layer, prefix, and suffix layer sets retain Standard's direct memory
  residual and SPS's original predict mask. BF16 and FP32 arithmetic share
  quantized weights/memories. Plots use equal per-text deltas on tokens 2..L;
  position heatmaps include token 1. `tests/test_sps_memory_ablation.py` checks
  mask locality, cleanup, full-block gradient isolation, and plotting. The shared
  `analyze_sps_cramming.submit` accepts entrypoint/job-kind overrides and stages
  both root/src on PYTHONPATH; its original defaults remain unchanged.

- Mechanism research uses `scripts/sps_mechanism_common.py` with the matched
  source-run registry and exact saved stages/token IDs. It does not crop memories
  trained at unequal horizons for the route comparison. All forwards are batch1;
  arithmetic is FP32 on BF16-quantized checkpoint weights and saved memories.
- `scripts/analyze_sps_memory_routes.py --phase dry|submit|probe|plot` separates
  memory-to-context and memory-to-predict edges in SPS, tests fixed-query versus
  context-sensitive queries, and patches projected memory K/V from norm-matched
  donor memories. `analysis.native_memory_interventions` keeps q0's original V
  before c_proj, preserving memory residual evolution while modifying outgoing
  key/value reads. Self-patches and hook cleanup are regression-tested in
  `tests/test_native_memory_interventions.py`. Controls retain complete causal
  and SPS temporary-key masks; replacing a memory is an intervention, not a
  clean attribution of capacity to a specific parameter family.
- `scripts/analyze_sps_margin_geometry.py --phase dry|submit|probe|plot` evaluates
  warm-start and optimized memories at the same first-failed SPS target length
  in both models. Target-vs-best-competitor margin Jacobians measure local
  controllability, old/new gradient conflicts, and minimum-norm projected steps.
  Finite perturbations check the local approximation. These are diagnostics,
  not a new optimizer or a global rank/capacity bound. Their math and actual
  native forward/backward path are checked in `tests/test_sps_margin_geometry.py`.
  Both new entrypoints reuse snapshot submission and output guards from
  `scripts.analyze_sps_cramming.submit`; plots can be redrawn without GPUs.

- `scripts/analyze_sps_margin_rescue.py --phase dry|submit|probe|plot` restarts
  the first failed SPS memories with CE-Adam, squared-margin-hinge Adam and a
  trust-region linearized-margin controller, without adding memory dimensions.
  Source experiment configs define the arms and VJP budget. Scalar-loss backward
  counts one VJP; full margin Jacobians count one per target. Forward counts and
  wall times are separate. The controller solves a row-normalized nonnegative
  dual QP, checks actual primal feasibility, backtracks on real merit and keeps
  the best measured minimum-margin memory. Initial BF16 accuracy must replay the
  saved failure; solutions are checked by FP32/BF16 teacher forcing and greedy
  generation. This is extra diagnostic optimization, not the original schedule
  or a continuation of missing Adam moments. `tests/test_sps_margin_rescue.py`
  verifies known feasible/infeasible systems and complete autograd/budget loops.

- `scripts/jobs/run_jobs_sps.py --suite objectives --dry` describes the matched
  objective/convergence factorial; omit `--dry` to submit. Its registry reuses
  completed CE/raw baselines and defines the six additional native-checkpoint
  runs. All cramming remains batch one. Defaults and the LR sweep are unchanged.
- `token_objective=squared_margin` selects a differentiable target-versus-largest-
  rival squared hinge in `BaseTrainer.compute_loss`, shared by full/progressive
  and low-dimensional decoder-only paths. `loss_margin` is its positive
  target gap; the default CE path retains existing weighted/hybrid behavior.
  Encoder-decoder, prefix-tuning, no-op, compression-head training and incompatible CE-only options reject
  this objective. Full/progressive result rows persist objective and margin fields.
  `tests/test_squared_margin_objective.py` checks real native forwards/backwards,
  offsets, masks, default CE compatibility, invalid arguments and suite coverage.
- `scripts/audit_sps_objective_gradients.py --phase dry|submit|probe|plot` reads
  completed rescue vectors without optimization, compares CE preferences and
  decomposes correct/wrong-token gradients at the same SPS checkpoint. It records
  gradient cancellation and alignment with the worst-margin gradient; these local
  quantities do not establish a global optimizer or architecture limitation.
  `tests/test_objective_gradient_audit.py` checks a known opposing-gradient case.

- `scripts/analyze_sps_objective_factorial.py` validates the completed objective
  suite against the launch registry and pinned exact target IDs; it reads saved
  validation outputs without new inference/submission. It exports per-text CSV,
  paired-text bootstrap contrasts and a plot. Teacher horizons, exact greedy
  success and the initial correct prefix for the final memory are distinct:
  the latter is not a maximal greedy horizon. Intervals exclude seed variance.
  `tests/test_objective_factorial_analysis.py` checks first-error/short-generation
  scoring, a known constant paired contrast and rejection of inconsistent IDs.

- `SPSCausalLM.forward(predict_embeddings=[B,1,H])` optionally broadcasts a
  differentiable per-text input predict vector to all SPS prediction slots.
  The default vocabulary predict embedding, masks, duplicated logical RoPE,
  ordinary embeddings and frozen tied output head are unchanged. Standard and
  invalid shapes reject the override. `generate_from_compression` and validation
  `check_sample` forward this optional vector only when explicitly supplied.
- `scripts/analyze_sps_predict_embedding.py --phase dry|submit|probe|summary`
  defines the fixed-BOS/shared-predict experiment registry and specialized
  `PredictEmbeddingTrainer`. It reuses the progressive optimizer/state machine
  but replaces the learned ordinary memory with a fixed existing vocabulary BOS,
  and routes the one latent vector exclusively to prediction inputs. Native and
  random initializations are compared with saved same-budget SPS memory controls.
  Artifacts use `predict_prefixes`, explicit `embedding_role=shared_predict`, BOS
  ID and an information-gain reference of fixed BOS/native predict. Summary
  requires every registered job completed. Probe requires `--experiment`.
  `scripts.analyze_sps_cramming.submit(probe_args=...)` stages the explicit
  experiment argument; existing diagnostic callers retain their original CLI.
  `tests/test_sps_predict_embedding.py` checks identity, gradients, ordinary-state
  invariance, independent ordinary-only decoding, invalid overrides, actual
  progressive training/artifact roles/frozen weights and registry/launch arguments.
- The same predict entrypoint accepts `--suite budget` for the registered larger
  total-budget margin runs. These start fresh with their original initializations;
  saved stage vectors do not contain resumable optimizer/scheduler state. The
  per-stage cap is unchanged; the cosine schedule spans the larger total budget.
  Their controls are the completed same-initialization predict runs. Probe requires
  an experiment from the selected suite; dry/submit propagate both CLI arguments.
  Budget summaries/plots have a `budget_` filename prefix and preserve baseline
  analysis outputs. Validation records the resolved training settings and run name.
- `scripts/analyze_sps_predict_shards.py --phase dry|submit|probe|merge` is the
  current native-hinge launch plan. Its `experiment_configs` partitions the exact
  original cached cohort into disjoint single-GPU shards. It reuses the existing
  staged diagnostic launcher and predict trainer; each job validates its own data.
  `analyze_sps_predict_embedding.resolved_training_args` pins the absolute cache;
  probe honors dataset offsets, saves global sample IDs and validates them against
  local sliced-cache indices. Default zero-offset callers retain their behavior.
  After the durable Arkhip outcome, run `--phase merge`: all completion markers,
  resolved settings, training snapshots, exact IDs, raw stages and report metrics
  must agree. The merger atomically writes combined `predict_prefixes`, validation,
  per-text CSV, provenance manifest and comparison plot under `merged/`; it keeps
  earlier outputs intact and repeated successful merges return the saved result.
  Shard submissions pass their validated immutable SHA through
  `analyze_sps_cramming.submit(snapshot_commit=...)`, including partial retries
  from an archived snapshot after the enclosing repository HEAD changes. Other
  callers retain the default HEAD behavior. Raw stages must start at zero without
  gaps and match the registered prefix lengths, token counts and convergence threshold.
  `tests/test_sps_predict_shards.py` covers the real dry payloads, cohort partition,
  complete/idempotent merge and rejection of incomplete or inconsistent inputs.
