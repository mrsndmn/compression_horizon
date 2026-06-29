"""Build the progressive convergence-margin comparison table (tab:progressive_margin).

Compares, per model, the baseline progressive run against the decode-robust margin variants
(``--convergence_margin 0.5`` and ``--convergence_margin 0.5 --loss_margin 0.5`` -- see
scripts/jobs/run_jobs_progressive.py). Columns: compressed tokens (mean +/- std over samples,
last *converged* stage) and greedy autoregressive reconstruction accuracy (from
greedy_accuracy_cache.json, written by scripts/greedy_reconstruction_eval.py with flash-attn +
last-converged-stage selection).

The story: a positive convergence_margin trades crammed tokens for honest greedy reconstruction
(baseline embeddings reconstruct poorly under standard greedy decoding despite TF=1.0);
loss_margin claws back some of the token cost at the same robustness.

Run after the relevant greedy_accuracy_cache.json files exist:
    python scripts/paper/tables/progressive_margin_table.py --save-dir paper/tables
Rows whose greedy cache is missing render "--" (and a note is printed).
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# Load scripts/results/results.py by path (the sibling scripts/results.py shadows the package).
_RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "results.py"
_spec = importlib.util.spec_from_file_location("_pm_results_helpers", _RESULTS_PATH)
assert _spec is not None and _spec.loader is not None
_results = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_pm_results_helpers", _results)
_spec.loader.exec_module(_results)
aggregate_progressive = _results.aggregate_progressive
load_dataset_rows = _results.load_dataset_rows

from tabulate import tabulate  # noqa: E402

from compression_horizon.utils import hlines_to_booktabs, to_mean_std_cell  # noqa: E402

_EXP = "artifacts/experiments_progressive"

# (model_label, variant_label, run_dir_basename). Only models with margin runs are listed;
# add SmolLM2-1.7B / gemma-3-4b-pt blocks once their cm_0.5[/lm_0.5] runs land.
EXPERIMENTS = [
    ("Llama-3.1-8B", "baseline", "sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1"),
    ("Llama-3.1-8B", "$\\epsilon$=0.5", "sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1_cm_0.5"),
    ("Llama-3.1-8B", "$\\epsilon$=0.5 +lm", "sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1_cm_0.5_lm_0.5"),
    ("pythia-1.4b", "baseline", "sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5"),
    ("pythia-1.4b", "$\\epsilon$=0.5", "sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5_cm_0.5"),
    ("pythia-1.4b", "$\\epsilon$=0.5 +lm", "sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5_cm_0.5_lm_0.5"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tab:progressive_margin (convergence-margin comparison).")
    p.add_argument("--tablefmt", default="latex", help="Tabulate format (latex, github, grid, ...).")
    p.add_argument("--save-dir", default=None, help="If set, write <save-dir>/progressive_margin.tex.")
    p.add_argument("--greedy-precision", type=int, default=2)
    return p.parse_args()


def load_greedy(run_dir: str):
    """(mean, std) from greedy_accuracy_cache.json, or None if absent/unreadable."""
    path = os.path.join(run_dir, "greedy_accuracy_cache.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return payload.get("greedy_match_mean"), payload.get("greedy_match_std")


def main() -> None:
    args = parse_args()
    use_latex = args.tablefmt == "latex"

    def pct_cell(mean_frac, std_frac, precision=2) -> str:
        if mean_frac is None:
            return "--"
        mean_str = f"{mean_frac * 100:.{precision}f}"
        if std_frac is None:
            return mean_str
        std_str = f"{std_frac * 100:.{precision}f}"
        return f"{mean_str} {{\\small $\\pm$ {std_str}}}" if use_latex else f"{mean_str} ± {std_str}"

    columns = ["Model", "Variant", "Compressed Tokens", "Greedy Recon. (%)"]
    rows = []
    missing_greedy = []
    prev_model = None
    for model_label, variant_label, basename in EXPERIMENTS:
        run_dir = os.path.join(_EXP, basename)
        pp = os.path.join(run_dir, "progressive_prefixes")
        if not os.path.isdir(pp):
            print(f"[skip] missing progressive_prefixes: {pp}")
            continue
        summary = aggregate_progressive(pp, load_dataset_rows(pp))
        if summary is None:
            print(f"[skip] aggregate_progressive returned None: {pp}")
            continue

        tokens_cell = to_mean_std_cell(
            summary.number_of_compressed_tokens,
            summary.number_of_compressed_tokens_std,
            use_latex=use_latex,
            float_precision=0,
        )
        greedy = load_greedy(run_dir)
        if greedy is None or greedy[0] is None:
            greedy_cell = "--"
            missing_greedy.append(basename)
        else:
            greedy_cell = pct_cell(greedy[0], greedy[1], precision=args.greedy_precision)

        # Midrule between model groups (sentinel row stripped after rendering).
        if use_latex and prev_model is not None and prev_model != model_label:
            rows.append(["\\midrule REMOVE", "", "", ""])
        rows.append([model_label if model_label != prev_model else "", variant_label, tokens_cell, greedy_cell])
        prev_model = model_label

    result = tabulate(rows, headers=columns, tablefmt=args.tablefmt)
    if use_latex:
        # tabulate latex-escapes our already-LaTeX cells; undo it (same as full_cramming_table.py),
        # then strip the REMOVE midrule sentinels, then convert hlines to booktabs.
        import re

        result = result.replace("\\textbackslash{}", "\\")
        result = result.replace("\\$", "$")
        result = result.replace("\\{", "{")
        result = result.replace("\\}", "}")
        result = re.sub(r"REMOVE.+", "", result)
        result = hlines_to_booktabs(result)
    # Strip trailing whitespace (the REMOVE-sentinel midrule rows leave some) so the
    # rendered .tex is stable under the trailing-whitespace pre-commit hook.
    result = "\n".join(line.rstrip() for line in result.split("\n"))
    print(result)

    if missing_greedy:
        print("\n[note] greedy cache missing (rendered '--'); run greedy_reconstruction_eval.py --dataset-type progr for:")
        for b in missing_greedy:
            print("   ", b)

    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "progressive_margin.tex"
        stamp = "% paper-lint: n_samples=50\n"
        out_path.write_text(stamp + result + "\n", encoding="utf-8")
        print(f"\nSaved 'tab:progressive_margin' to {out_path}")


if __name__ == "__main__":
    main()
