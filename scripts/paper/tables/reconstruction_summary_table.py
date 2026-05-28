"""Aggregate per-experiment JSONs from reconstruction_summary_eval.py into the
compression-reconstruction-summary table (paper/tables/manual/compression_reconstruction_summary.tex).

Default layout assumed:
    artifacts/experiments/
        Llama-3.2-1B_512_common/table_18_eval.json
        Llama-3.2-1B_512_no_bos/table_18_eval.json
        Llama-3.2-1B_512_2leading/table_18_eval.json
        Llama-3.2-3B_1024_common/table_18_eval.json
        ... (9 total)

Output: Markdown or LaTeX rendering matching the camera-ready paper layout.
Use --mismatch_mode both to render greedy and teacher-forced mismatch side by side.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# (directory-name stem, sequence length). The directory stem must match the
# basename of the HF checkpoint used in reproduction.py, e.g.
# unsloth/Meta-Llama-3.1-8B -> "Meta-Llama-3.1-8B".
MODEL_ORDER = [
    ("Llama-3.2-1B", 512),
    ("Llama-3.2-3B", 1024),
    ("Meta-Llama-3.1-8B", 1568),
]
# Pretty names for the rendered table (the official "Llama-3.1-8B" drops "Meta").
MODEL_DISPLAY = {
    "Llama-3.2-1B": "Llama-3.2-1B",
    "Llama-3.2-3B": "Llama-3.2-3B",
    "Meta-Llama-3.1-8B": "Llama-3.1-8B",
}
# Some runs were saved under an alias checkpoint name (unsloth mirrors the same
# Llama-3.2-3B base weights under "Meta-Llama-3.2-3B"). Map the canonical dir
# stem to any extra stems to look for when locating the eval JSON.
DIR_ALIASES = {
    "Llama-3.2-3B": ["Meta-Llama-3.2-3B"],
}
SETUP_ORDER = ("common", "no_bos", "2leading")
SETUP_DISPLAY = {"common": "common", "no_bos": "no BOS", "2leading": "2 leading"}


def _load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mismatch_at(summary: dict, k: int, mode: str) -> float | None:
    """Per-position mismatch at index k for the requested mode.

    teacher_forced -> tf_mismatch_curve[k] (the original Table 18 definition:
        argmax under a single forward over the full ground-truth prefix).
    greedy        -> greedy_mismatch_curve[k] (autoregressive from the
        compression token), with fallback to the legacy scalar mismatch_at_{k}.
    """
    if mode == "teacher_forced":
        curve = summary.get("tf_mismatch_curve")
        return curve[k] if curve is not None and len(curve) > k else None
    curve = summary.get("greedy_mismatch_curve")
    # Use the curve only if it actually holds a value at k; after --skip_greedy the
    # curve is [None, None, ...], so fall back to the preserved scalar mismatch_at_{k}.
    if curve is not None and len(curve) > k and curve[k] is not None:
        return curve[k]
    return summary.get(f"mismatch_at_{k}")  # legacy / preserved greedy scalar


def _pct(x) -> str:
    return f"{x*100:.1f}%" if x is not None else "—"


def _metric_cells(summary: dict, mismatch_mode: str) -> list[str]:
    """Mismatch cells after the Final/Greedy-conv columns, per mode.

    teacher_forced / greedy -> 3 cells (@0/@1/@2 of that mode).
    both                    -> 6 cells (greedy @0/@1/@2 then teacher-forced @0/@1/@2).
    """
    if mismatch_mode == "both":
        greedy = [_pct(_mismatch_at(summary, k, "greedy")) for k in (0, 1, 2)]
        tf = [_pct(_mismatch_at(summary, k, "teacher_forced")) for k in (0, 1, 2)]
        return greedy + tf
    return [_pct(_mismatch_at(summary, k, mismatch_mode)) for k in (0, 1, 2)]


def _num_metric_cols(mismatch_mode: str) -> int:
    return 6 if mismatch_mode == "both" else 3


def _row_for(model: str, seq_len: int, setup: str, summary: dict | None, mismatch_mode: str) -> list[str]:
    display_model = MODEL_DISPLAY.get(model, model)
    lead = [display_model, str(seq_len), SETUP_DISPLAY[setup]]
    if summary is None:
        return lead + ["—"] * (2 + _num_metric_cols(mismatch_mode))
    fc = summary.get("final_convergence_mean")
    gc = summary.get("greedy_match_rate_mean")
    return lead + [
        f"{fc:.4f}" if fc is not None else "—",
        f"{gc:.4f}" if gc is not None else "—",
        *_metric_cells(summary, mismatch_mode),
    ]


def _markdown_headers(mismatch_mode: str) -> list[str]:
    base = ["Model", "Tokens", "Setup", "Final conv.", "Greedy conv."]
    if mismatch_mode == "both":
        return base + ["G@0", "G@1", "G@2", "TF@0", "TF@1", "TF@2"]
    return base + ["@0", "@1", "@2"]


def _to_markdown(rows: list[list[str]], mismatch_mode: str) -> str:
    header = _markdown_headers(mismatch_mode)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _to_latex(rows: list[list[str]], mismatch_mode: str, n_samples: int | None = None) -> str:
    lines: list[str] = []
    if n_samples is not None:
        # Provenance stamp consumed by paper/lint_paper.py (body-table-samples).
        lines.append(f"% paper-lint: n_samples={n_samples}")
    if mismatch_mode == "both":
        lines += [
            r"\begin{tabular}{lll r c ccc ccc}",
            r"\toprule",
            r"Model & Tokens & Setup & Final conv. & Greedy conv. & "
            r"\multicolumn{3}{c}{Greedy mismatch (\%)} & \multicolumn{3}{c}{TF mismatch (\%)} \\",
            r"\cmidrule(lr){6-8} \cmidrule(lr){9-11}",
            r" &  &  &  &  & @0 & @1 & @2 & @0 & @1 & @2 \\",
            r"\midrule",
        ]
    else:
        lines += [
            r"\begin{tabular}{lll r c c c c}",
            r"\toprule",
            r"Model & Tokens & Setup & Final conv. & Greedy conv. & \multicolumn{3}{c}{Mismatch (\%)} \\",
            r"\cmidrule(lr){6-8}",
            r" &  &  &  &  & @0 & @1 & @2 \\",
            r"\midrule",
        ]
    for i, r in enumerate(rows):
        # Insert \midrule between model groups (every 3 rows).
        if i > 0 and i % 3 == 0:
            lines.append(r"\midrule")
        lines.append(" & ".join(r).replace("%", r"\%") + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts_dir",
        default="artifacts/experiments",
        help="Root directory containing <model>_<seqlen>_<setup>/table_18_eval.json subdirs.",
    )
    parser.add_argument("--eval_filename", default="table_18_eval.json")
    parser.add_argument(
        "--format",
        choices=("markdown", "latex", "both"),
        default="both",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="If set, write the rendered table to this file (markdown only). Stdout otherwise.",
    )
    parser.add_argument(
        "--mismatch_mode",
        choices=("teacher_forced", "greedy", "both"),
        default="teacher_forced",
        help="Which per-position mismatch to render for @0/@1/@2. 'teacher_forced' "
        "matches the original Table 18 (single forward over the ground-truth prefix); "
        "'greedy' reports autoregressive mismatch (error-compounding); 'both' shows "
        "greedy and teacher-forced side by side. Default: teacher_forced.",
    )
    args = parser.parse_args()

    rows: list[list[str]] = []
    sample_counts: set[int] = set()
    for model, seq_len in MODEL_ORDER:
        for setup in SETUP_ORDER:
            candidates = [model, *DIR_ALIASES.get(model, [])]
            summary = None
            tried = []
            for stem in candidates:
                run_dir = Path(args.artifacts_dir) / f"{stem}_{seq_len}_{setup}"
                tried.append(str(run_dir))
                summary = _load_summary(run_dir / args.eval_filename)
                if summary is not None:
                    break
            if summary is None:
                print(f"[WARN] missing eval JSON for {' or '.join(tried)}")
            else:
                n = summary.get("num_samples")
                if n is not None:
                    sample_counts.add(int(n))
            rows.append(_row_for(model, seq_len, setup, summary, args.mismatch_mode))

    # Single provenance stamp only if every loaded run shares the same sample count.
    n_samples = sample_counts.pop() if len(sample_counts) == 1 else None
    if len(sample_counts) > 1:
        print(f"[WARN] inconsistent num_samples across runs: {sorted(sample_counts)} -- omitting stamp")

    md = _to_markdown(rows, args.mismatch_mode)
    tex = _to_latex(rows, args.mismatch_mode, n_samples=n_samples)

    if args.format in ("markdown", "both"):
        print("=== Markdown ===")
        print(md)
    if args.format in ("latex", "both"):
        print("\n=== LaTeX ===")
        print(tex)

    if args.output is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            if args.format == "latex":
                f.write(tex + "\n")
            else:
                f.write(md + "\n")
        print(f"\nWrote table to: {args.output}")


if __name__ == "__main__":
    main()
