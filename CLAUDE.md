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
