#!/usr/bin/env python3
"""Generate the main-text depth x size pivot table (``tab:depth_size_pivot``).

A compact overview of compression capacity: rows are model checkpoints (each name
annotated with its total decoder-layer count), columns are the number of first-$N$
decoder layers retained (then finetuned), and the last column is the full
(untruncated) model. Each cell is the mean number of perfectly crammed
(``compressed'') tokens over the PG19 progressive-cramming eval, read from the same
``progressive_prefixes`` dirs as the detailed per-metric table ``tab:layer_ablation``
(scripts/paper/tables/progressive.py). Missing configurations render as ``--``.

To extend: add a row to ``MODELS`` (display name, total layers, checkpoint base used
in the output-dir name, which first-$N$ runs exist, whether the full model was run) or
a column to ``FIRST_N_COLUMNS``. Regenerate:

    PYTHONPATH=./src python scripts/paper/tables/depth_size_pivot.py --save
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from compression_horizon.paper.tables.progressive import extract_trajectory  # noqa: E402

EXP = os.path.join(_ROOT, "artifacts", "experiments_progressive")
OUT = os.path.join(_ROOT, "paper", "tables", "depth_size_pivot.tex")
MISSING = "--"
# Columns: the first-N truncated checkpoints, then a "Full" (untruncated) column.
FIRST_N_COLUMNS = [1, 2, 4, 8]


@dataclass(frozen=True)
class Model:
    display: str  # shown name, e.g. "SmolLM2-1.7B"
    total_layers: int  # appended to the name as "(24L)"
    ckpt_base: str  # output-dir base, e.g. "SmolLM2-1.7B" or "Meta-Llama-3.1-8B"
    first_ns: tuple  # which first-N truncations were run for this model
    has_full: bool  # whether the full (untruncated) progressive run exists


# Ordered by model size. Edit this list to add models / depths.
MODELS = [
    Model("SmolLM2-1.7B", 24, "SmolLM2-1.7B", (1, 2, 4, 8), True),
    Model("SmolLM3-3B", 36, "SmolLM3-3B", (1, 2, 4, 8), False),
    Model("Qwen3-4B", 36, "Qwen3-4B", (1, 2, 4, 8), True),
    Model("Qwen3-8B", 36, "Qwen3-8B", (1, 2, 4, 8), True),
    Model("Llama-3.1-8B", 32, "Meta-Llama-3.1-8B", (1, 2, 4), True),
]


def _first_dir(base: str, n: int) -> str:
    return os.path.join(EXP, f"sl_4096_{base}-first{n}-ftw_ds_pg19_1k_limit_50_lr_0.1", "progressive_prefixes")


def _full_dir(base: str) -> str:
    return os.path.join(EXP, f"sl_4096_{base}_ds_pg19_1k_limit_50_lr_0.1", "progressive_prefixes")


def cell_value(path: str | None) -> tuple[float | None, int | None]:
    """``(mean compressed-token count, sample count)`` for one eval dir.

    Returns ``(None, None)`` when the dir is absent or extraction fails. The
    sample count feeds the provenance stamp consumed by ``paper/lint_paper.py``.
    """
    if not path or not os.path.isdir(path):
        return None, None
    try:
        _, _, stats, _ = extract_trajectory(path)
    except Exception as e:  # noqa: BLE001 - report and leave the cell empty
        print(f"  WARN: {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return None, None
    v = stats.get("num_embeddings")
    if isinstance(v, dict):
        count = int(v["count"]) if v.get("count") is not None else None
        return v.get("mean"), count
    return v, None


def provenance_stamp(sample_counts: set[int]) -> str:
    """Machine-readable provenance comment consumed by ``paper/lint_paper.py``.

    Mirrors ``provenance_stamp`` in scripts/paper/tables/progressive.py: records
    the set of per-cell PG19 sample counts so the linter can confirm this
    main-text table is built from the canonical 50-sample run.
    """
    value = ",".join(str(c) for c in sorted(sample_counts)) if sample_counts else "unknown"
    return f"% paper-lint: n_samples={value}"


def _cell(v: float | None) -> str:
    return MISSING if v is None else f"{v:.0f}"


def build_table() -> str:
    ncol = len(FIRST_N_COLUMNS)
    col_spec = "l" + "r" * (ncol + 1)  # Model + first-N columns + Full
    n_header = " & ".join(str(n) for n in FIRST_N_COLUMNS)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Model & \\multicolumn{{{ncol}}}{{c}}{{Retained first-$N$ layers}} & Full \\\\",
        f" & {n_header} & \\\\",
        "\\midrule",
    ]
    sample_counts: set[int] = set()
    for m in MODELS:
        cells = []
        for n in FIRST_N_COLUMNS:
            mean, count = cell_value(_first_dir(m.ckpt_base, n) if n in m.first_ns else None)
            if count is not None:
                sample_counts.add(count)
            cells.append(_cell(mean))
        mean, count = cell_value(_full_dir(m.ckpt_base) if m.has_full else None)
        if count is not None:
            sample_counts.add(count)
        cells.append(_cell(mean))
        lines.append(f" {m.display} ({m.total_layers}L) & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return provenance_stamp(sample_counts) + "\n" + "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true", help=f"Write the table to {os.path.relpath(OUT, _ROOT)}.")
    args = ap.parse_args()

    tex = build_table()
    print(tex)
    if args.save:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as fh:
            fh.write(tex)
        print(f"Saved 'tab:depth_size_pivot' to {os.path.relpath(OUT, _ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
