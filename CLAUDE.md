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

Source: `poster/poster.html`. After editing, validate layout and render a preview:

1. **Measure gate** (validates column alignment, gaps, overflow):
   `python .claude/posterly/tools/poster_check.py measure poster/poster.html`
2. **Render PNG preview**:
   `python .claude/posterly/tools/render_preview.py poster/poster.html --png poster/poster_preview.png`

Do not read rendered images yourself — the user reviews visually.
