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
