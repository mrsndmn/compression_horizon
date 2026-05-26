"""Aggregate per-experiment JSONs from scripts/eval_table_18.py into Table 18.

Default layout assumed:
    artifacts/experiments/
        Llama-3.2-1B_512_common/table_18_eval.json
        Llama-3.2-1B_512_no_bos/table_18_eval.json
        Llama-3.2-1B_512_2leading/table_18_eval.json
        Llama-3.2-3B_1024_common/table_18_eval.json
        ... (9 total)

Output: Markdown or LaTeX rendering matching the camera-ready paper layout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MODEL_ORDER = [
    ("Llama-3.2-1B", 512),
    ("Llama-3.2-3B", 1024),
    ("Llama-3.1-8B", 1568),
]
SETUP_ORDER = ("common", "no_bos", "2leading")
SETUP_DISPLAY = {"common": "common", "no_bos": "no BOS", "2leading": "2 leading"}


def _load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _row_for(model: str, seq_len: int, setup: str, summary: dict | None) -> list[str]:
    if summary is None:
        return [model, str(seq_len), SETUP_DISPLAY[setup], "—", "—", "—", "—", "—"]
    fc = summary.get("final_convergence_mean")
    gc = summary.get("greedy_match_rate_mean")
    m0 = summary.get("mismatch_at_0")
    m1 = summary.get("mismatch_at_1")
    m2 = summary.get("mismatch_at_2")

    def _pct(x):
        return f"{x*100:.1f}%" if x is not None else "—"

    return [
        model,
        str(seq_len),
        SETUP_DISPLAY[setup],
        f"{fc:.4f}" if fc is not None else "—",
        f"{gc:.4f}" if gc is not None else "—",
        _pct(m0),
        _pct(m1),
        _pct(m2),
    ]


def _to_markdown(rows: list[list[str]]) -> str:
    header = ["Model", "Tokens", "Setup", "Final conv.", "Greedy conv.", "@0", "@1", "@2"]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _to_latex(rows: list[list[str]]) -> str:
    lines = [
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
    args = parser.parse_args()

    rows: list[list[str]] = []
    for model, seq_len in MODEL_ORDER:
        for setup in SETUP_ORDER:
            run_dir = Path(args.artifacts_dir) / f"{model}_{seq_len}_{setup}"
            summary = _load_summary(run_dir / args.eval_filename)
            if summary is None:
                print(f"[WARN] missing eval JSON for {run_dir}")
            rows.append(_row_for(model, seq_len, setup, summary))

    md = _to_markdown(rows)
    tex = _to_latex(rows)

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
