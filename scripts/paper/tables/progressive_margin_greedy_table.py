"""Generate tab:progressive_margin_greedy (paper appendix, app:tf_vs_greedy_kernel).

Greedy autoregressive reconstruction accuracy of progressive-cramming embeddings trained
with a convergence margin epsilon, across four model families x {baseline, low-dim projection}
x epsilon in {0.5, 1.0, 2.0} x {plain CE, +loss-margin}. Both the greedy match rate and the
mean number of compressed tokens are read from each run's ``greedy_accuracy_cache.json`` (written
by scripts/greedy_reconstruction_eval.py with flash-attn + last-converged-stage selection): the
per-sample ``L`` field gives the crammed length, so no dataset reload is needed. Cells render as
``greedy% (tokens)``; a missing cache renders ``--``.

    python scripts/paper/tables/progressive_margin_greedy_table.py --save-dir paper/tables
"""

import argparse
import json
import os
from pathlib import Path

_EXP = "artifacts/experiments_progressive"

# (display label, checkpoint short-name in the run dir, low-dim size, learning rate)
MODELS = [
    ("Llama-3.1-8B", "Meta-Llama-3.1-8B", 256, "0.1"),
    ("Pythia-1.4B", "pythia-1.4b", 256, "0.5"),
    ("SmolLM2-1.7B", "SmolLM2-1.7B", 256, "0.1"),
    ("Gemma-3-4B", "gemma-3-4b-pt", 32, "0.1"),
]
MARGINS = ["0.5", "1.0", "2.0"]


def run_dir_name(short: str, lr: str, low_dim_size: int | None) -> str:
    low = f"_lowdim_{low_dim_size}_lowproj" if low_dim_size is not None else ""
    return f"sl_4096_{short}_ds_pg19_1k_limit_50{low}_lr_{lr}"


def cell(base: str, eps: str, with_lm: bool) -> str:
    """``greedy% (tokens)`` for one margin run, or ``--`` if its greedy cache is absent."""
    lm = f"_lm_{eps}" if with_lm else ""
    run = f"{base}_cm_{eps}{lm}"
    path = os.path.join(_EXP, run, "greedy_accuracy_cache.json")
    if not os.path.isfile(path):
        return "--"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "--"
    per_sample = data.get("per_sample") or []
    lengths = [s["L"] for s in per_sample if "L" in s]
    mean = data.get("greedy_match_mean")
    if not lengths or mean is None:
        return "--"
    tokens = round(sum(lengths) / len(lengths))
    greedy = "\\textbf{100}" if mean * 100 >= 99.995 else f"{mean * 100:.1f}"
    return f"{greedy} {{\\scriptsize({tokens})}}"


def render() -> str:
    lines = [
        r"\begin{tabular}{l l cc cc cc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{$\epsilon=0.5$} & \multicolumn{2}{c}{$\epsilon=1.0$} "
        r"& \multicolumn{2}{c}{$\epsilon=2.0$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Model & Proj. & CE & +LM & CE & +LM & CE & +LM \\",
        r"\midrule",
    ]
    for i, (label, short, low_dim_size, lr) in enumerate(MODELS):
        for j, (proj_label, low) in enumerate((("--", None), ("low-dim", low_dim_size))):
            base = run_dir_name(short, lr, low)
            cells = [cell(base, eps, lm) for eps in MARGINS for lm in (False, True)]
            model_col = label if j == 0 else ""
            lines.append(f"{model_col} & {proj_label} & " + " & ".join(cells) + r" \\")
        if i < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tab:progressive_margin_greedy.")
    parser.add_argument("--save-dir", default=None, help="If set, write <save-dir>/progressive_margin_greedy.tex.")
    args = parser.parse_args()

    table = render()
    print(table)

    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "progressive_margin_greedy.tex"
        out_path.write_text(table, encoding="utf-8")
        print(f"Saved 'tab:progressive_margin_greedy' to {out_path}")


if __name__ == "__main__":
    main()
