"""Side-by-side print-out of our downstream_eval.json results vs paper Table 10.

Reads up to three saved ``downstream_eval.json`` files (HellaSwag, ARC-Easy,
ARC-Challenge), aligns them to the five paper-canonical PPL variants of
Table 10 (page 14 of ICML2026_Compression.pdf), and prints a single
ours-vs-paper-1.7B matrix in the same row/column layout as the paper.

Run AFTER the three downstream shell scripts:

    bash scripts/thesis_reproduction/experiments/downstream/smollm2_135m_hellaswag.sh
    bash scripts/thesis_reproduction/experiments/downstream/smollm2_135m_arc_easy.sh
    bash scripts/thesis_reproduction/experiments/downstream/smollm2_135m_arc_challenge.sh

then:

    uv run python scripts/thesis_reproduction/summarize_downstream_table10.py

Pass ``--no_paper`` to drop the paper column (useful for clean copy-pasting).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Paper Table 10 (page 14): SmolLM2-1.7B token-normalized accuracy (%) on the
# perfectly-reconstructed subset. Five canonical strategies.
PAPER_TABLE_10: dict[str, dict[str, float]] = {
    "hellaswag": {
        "baseline": 52.41,
        "baseline_endings": 53.30,
        "compression": 36.47,
        "compression_endings": 40.70,
        "compression_only": 34.57,
    },
    "arc-easy": {
        "baseline": 68.72,
        "baseline_endings": 53.88,
        "compression": 55.87,
        "compression_endings": 41.63,
        "compression_only": 30.92,
    },
    "arc-challenge": {
        "baseline": 36.66,
        "baseline_endings": 40.29,
        "compression": 28.62,
        "compression_endings": 31.36,
        "compression_only": 22.94,
    },
}

# Order of rows in the paper table.
PAPER_ROWS: tuple[str, ...] = (
    "baseline",
    "baseline_endings",
    "compression",
    "compression_endings",
    "compression_only",
)

# Extra diagnostic rows we measure but paper Table 10 does not list.
EXTRA_ROWS: tuple[str, ...] = (
    "compression_edge",
    "compression_only_edge",
    "compression_only_endings",
)


def _benchmark_paths(model_slug: str) -> dict[str, str]:
    """Construct default downstream_eval.json paths for the chosen model slug."""
    return {
        "hellaswag": f"artifacts/thesis_reproduction/downstream/{model_slug}_hellaswag/downstream_eval.json",
        "arc-easy": f"artifacts/thesis_reproduction/downstream/{model_slug}_arc_easy/downstream_eval.json",
        "arc-challenge": f"artifacts/thesis_reproduction/downstream/{model_slug}_arc_challenge/downstream_eval.json",
    }


MODEL_PRESETS: dict[str, dict[str, str]] = {
    "smollm2_135m": {
        "model_label": "SmolLM2-135M (ours)",
        "title_suffix": "SmolLM2-135M (ours) vs SmolLM2-1.7B (paper)",
    },
    "smollm2_1_7b": {
        "model_label": "SmolLM2-1.7B (ours, same model as paper)",
        "title_suffix": "SmolLM2-1.7B (ours, same model as paper) vs SmolLM2-1.7B (paper)",
    },
}

DEFAULT_MODEL = "smollm2_135m"

BENCHMARK_LABELS: dict[str, str] = {
    "hellaswag": "HellaSwag",
    "arc-easy": "ARC-Easy",
    "arc-challenge": "ARC-Challenge",
}


def _load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _fmt_cell(ours_pct: float | None, paper_pct: float | None, show_paper: bool) -> str:
    if show_paper:
        ours_str = f"{ours_pct:>6.2f}%" if ours_pct is not None else f"{'-':>7}"
        paper_str = f"{paper_pct:>6.2f}%" if paper_pct is not None else f"{'-':>7}"
        return f"{ours_str} / {paper_str}"
    return f"{ours_pct:>6.2f}%" if ours_pct is not None else f"{'-':>7}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side Table-10 summary.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        default=DEFAULT_MODEL,
        help=(
            "Model slug used to construct default paths to downstream_eval.json. "
            "Pass --hellaswag/--arc_easy/--arc_challenge to override individual paths."
        ),
    )
    # Bench-specific path overrides default to None; we fill them from --model below.
    parser.add_argument("--hellaswag", default=None, help="Path to HellaSwag downstream_eval.json")
    parser.add_argument("--arc_easy", default=None, help="Path to ARC-Easy downstream_eval.json")
    parser.add_argument(
        "--arc_challenge",
        default=None,
        help="Path to ARC-Challenge downstream_eval.json",
    )
    parser.add_argument(
        "--no_paper",
        action="store_true",
        help="Hide paper-1.7B column (useful for clean copy-paste).",
    )
    parser.add_argument(
        "--show_extras",
        action="store_true",
        help="Also print compression_edge / compression_only_edge / compression_only_endings rows.",
    )
    parser.add_argument(
        "--subset",
        choices=["summary_perfectly_reconstructed", "summary_all_samples", "summary"],
        default="summary_perfectly_reconstructed",
        help=(
            "Which saved summary view to compare against the paper. "
            "Default matches paper Table 10 (perfectly-reconstructed subset). "
            "Falls back to 'summary' when the selected view is missing in older JSONs."
        ),
    )
    args = parser.parse_args()

    model_preset = MODEL_PRESETS[args.model]
    model_paths = _benchmark_paths(args.model)
    paths = {
        "hellaswag": args.hellaswag or model_paths["hellaswag"],
        "arc-easy": args.arc_easy or model_paths["arc-easy"],
        "arc-challenge": args.arc_challenge or model_paths["arc-challenge"],
    }
    loaded: dict[str, dict | None] = {bench: _load(path) for bench, path in paths.items()}

    show_paper = not args.no_paper
    cell_width = 16 if show_paper else 8

    def _pick_summary(data: dict | None) -> dict | None:
        if data is None:
            return None
        # Prefer the requested subset; fall back to legacy `summary` if missing
        # (older JSONs only stored one view).
        for key in (args.subset, "summary"):
            if key in data:
                return data[key]
        return None

    title = f"Table 10 reproduction — {model_preset['title_suffix']}"
    print()
    print("=" * (24 + 3 * (cell_width + 2)))
    print(title)
    print(f"  view: {args.subset}")
    print("  source files:")
    for bench, data in loaded.items():
        marker = "OK" if data is not None else "MISSING"
        s = _pick_summary(data)
        n = s["num_samples_total"] if s is not None else "-"
        conv = s["num_full_convergence"] if s is not None else "-"
        print(f"    - {bench:<14} [{marker}]  n={n}, fully-converged={conv}")
    print("  Paper Table 10 row: SmolLM2-1.7B token-normalized accuracy, perfectly-reconstructed subset.")
    print("=" * (24 + 3 * (cell_width + 2)))

    # Header
    if show_paper:
        cell_hdr = "  ours / paper  "
    else:
        cell_hdr = "    ours    "
    bench_hdr = "  ".join(f"{BENCHMARK_LABELS[b]:>{cell_width}}" for b in paths)
    print(f"  {'Setup':<24}  {bench_hdr}")
    sub_hdr = "  ".join(f"{cell_hdr:>{cell_width}}" for _ in paths) if show_paper else ""
    if show_paper:
        print(f"  {'':<24}  {sub_hdr}")
    print("  " + "-" * (24 + 2 * len(paths) + len(paths) * cell_width))

    def _row_cell(row: str, bench: str) -> str:
        s = _pick_summary(loaded[bench])
        # Paper Table 10 reports token-normalized accuracy. Fall back to raw
        # accuracy for older JSONs that don't store the token-norm field.
        if s is None:
            ours_pct = None
        else:
            entry = s[row]
            ours_pct = entry.get("token_normalized_accuracy", entry["accuracy"]) * 100
        paper_pct = PAPER_TABLE_10.get(bench, {}).get(row) if show_paper else None
        return _fmt_cell(ours_pct, paper_pct, show_paper=show_paper)

    for row in PAPER_ROWS:
        cells = "  ".join(f"{_row_cell(row, b):>{cell_width}}" for b in paths)
        print(f"  {row:<24}  {cells}")

    if args.show_extras:
        print("  " + "-" * (24 + 2 * len(paths) + len(paths) * cell_width))
        print(f"  {'(diagnostic extras, not in paper Table 10)':<24}")
        for row in EXTRA_ROWS:
            cells = "  ".join(f"{_row_cell(row, b):>{cell_width}}" for b in paths)
            print(f"  {row:<24}  {cells}")

    print("=" * (24 + 3 * (cell_width + 2)))
    print()
    print("Notes:")
    if args.model == "smollm2_1_7b":
        print("  - Same model as paper (SmolLM2-1.7B). Direct quantitative comparison;")
        print("    expected.json gives ±5pp tolerance bands (std=0.025).")
    else:
        print(f"  - Paper Table 10 is SmolLM2-1.7B; our column is {model_preset['model_label']}.")
        print("    Absolute values are expected to be lower, but the ORDERING and the relative")
        print("    drops (compression vs baseline) should match qualitatively.")
    print("  - When the shell scripts use --only_full_convergence, our denominator matches")
    print("    the paper's 'perfectly reconstructed samples' subset.")
    print("  - For a single-benchmark deep dive (with z-scores and qualitative gate), use:")
    print(f"      uv run python scripts/thesis_reproduction/analyze.py --experiment downstream/{args.model}_<benchmark>")


if __name__ == "__main__":
    main()
