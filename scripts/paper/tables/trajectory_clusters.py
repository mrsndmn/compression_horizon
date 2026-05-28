"""Generate the trajectory cluster-structure tables (``tab:trajectory_cluster_structure*``).

Companion to the solution-diversity analysis (Appendix~\\ref{app:solution_diversity}): that section
shows a single progressive-cramming trajectory is low-dimensional because it is one warm-started
*path*, not an intrinsic low-dim solution manifold. Here we characterise the *shape* of that path --
is it a smooth low-dim curve, or a punctuated "dwell-and-jump" walk through disconnected basins?

For each run we read the per-sample converged-embedding trajectory and, against three nulls, ask:
  * smooth-curve null (Gaussian-process): is it more gapped than a smooth curve? (jump gap-ratio)
  * i.i.d. random-walk null: are jumps heavier-tailed than diffusion?
  * jump-shuffle null (decisive): does structure survive fixing the jump-size distribution? The
    lag-1 autocorrelation of jump magnitudes is ~0 under the shuffle by construction, so a real
    value above it means small/large jumps are bunched in ORDER -- genuine dwelling.

This script only *renders* the LaTeX from the JSON caches written by
``scripts/analyze_trajectory_clusters.py`` (one ``summary.json`` per run under
``artifacts/analysis/trajectory_clusters_135m/<run>/``). Regenerate those caches first by running
that script on each run (CPU only); then ``make tables`` re-renders without recomputation::

    PYTHONPATH=./src python scripts/analyze_trajectory_clusters.py --run_dir <run>   # once per run
    PYTHONPATH=./src:. python scripts/paper/tables/trajectory_clusters.py --save      # cheap re-render
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from tabulate import tabulate

_ANALYSIS = "artifacts/analysis/trajectory_clusters_135m"
_STEP_CACHE = "artifacts/paper/trajectory_steps"

# Model-scale trend (all at lr=0.1). Every row uses a 50-sample (limit_50) PG19 run so the table is
# apples-to-apples; the 135M/360M 50-sample runs are produced by
# scripts/jobs/run_jobs_trajectory_clusters_50samples.py (no 10-sample fallback -- a missing cache is
# a hard error, by design).
SCALE_ROWS: List[Tuple[str, str]] = [
    ("SmolLM2-135M", "sl_4096_SmolLM2-135M_ds_pg19_1k_limit_50_lr_0.1"),
    ("SmolLM2-360M", "sl_4096_SmolLM2-360M_ds_pg19_1k_limit_50_lr_0.1"),
    ("SmolLM2-1.7B", "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1"),
    ("Llama-3.1-8B", "sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1"),
]

# Learning-rate sweep (SmolLM2-1.7B), all 50-sample runs, apples-to-apples across LR. lr=0.1 reuses
# the scale-trend 50-sample run; lr=0.5 / lr=1.0 are produced by the same launcher above.
LR_ROWS: List[Tuple[str, str]] = [
    ("SmolLM2-1.7B {\\small lr=0.1}", "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1"),
    ("SmolLM2-1.7B {\\small lr=0.5}", "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.5"),
    ("SmolLM2-1.7B {\\small lr=1.0}", "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_1.0"),
]


def load_aggregate(run_name: str) -> dict:
    path = Path(_ANALYSIS) / run_name / "summary.json"
    if not path.exists():
        raise SystemExit(
            f"missing cache: {path}\n"
            f"Run it first:  PYTHONPATH=./src python scripts/analyze_trajectory_clusters.py "
            f"--run_dir artifacts/experiments_progressive/{run_name}"
        )
    return json.loads(path.read_text())["aggregate"]


def load_step_aggregate(run_name: str) -> dict:
    """Load the step-based jump cache written by trajectory_steps.py --compute."""
    p = Path(_STEP_CACHE) / f"{run_name}.json"
    if not p.exists():
        raise SystemExit(
            f"missing step cache: {p}\n"
            "Build it once:  PYTHONPATH=./src:. python scripts/paper/tables/trajectory_steps.py --compute"
        )
    return json.loads(p.read_text())


def _sgn(x: float) -> str:
    """Signed 2-decimal format, collapsing -0.00 to +0.00."""
    return f"{0.0 if abs(x) < 0.005 else x:+.2f}"


def _euclid_cells(agg: dict, latex: bool) -> list:
    gap = (
        f"{agg['gap_ratio_real_mean']:.1f} {{\\small ({agg['gap_ratio_null_mean']:.1f})}}"
        if latex
        else f"{agg['gap_ratio_real_mean']:.1f} ({agg['gap_ratio_null_mean']:.1f})"
    )
    r, s = _sgn(agg["jump_autocorr_real_mean"]), _sgn(agg["jump_autocorr_shuffle_mean"])
    acf = f"{r} {{\\small ({s})}}" if latex else f"{r} ({s})"
    dwell = f"{agg['fraction_dwelling_basins'] * 100:.0f}\\%" if latex else f"{agg['fraction_dwelling_basins'] * 100:.0f}%"
    return [gap, acf, dwell]


def _step_cells(st: dict, latex: bool) -> list:
    r, s = _sgn(st["autocorr_real_mean"]), _sgn(st["autocorr_shuffle_mean"])
    acf = f"{r} {{\\small ({s})}}" if latex else f"{r} ({s})"
    dwell = f"{st['fraction_dwelling'] * 100:.0f}\\%" if latex else f"{st['fraction_dwelling'] * 100:.0f}%"
    return [f"{st['gap_ratio_mean']:.1f}", acf, dwell]


def format_table(rows_spec: List[Tuple[str, str]], tablefmt: str = "latex") -> str:
    """Merged Euclidean- and step-based trajectory-shape table with grouped column headings.

    Reads both the Euclidean-jump cache (``analyze_trajectory_clusters.py``) and the step-jump cache
    (``trajectory_steps.py --compute``) for each run; the two metric families share ``Stages`` and
    ``PCA 99\\%`` and each contribute a (gap-ratio, jump-autocorrelation, dwelling-%) block.
    """
    latex = tablefmt == "latex"
    body = []
    for disp, run in rows_spec:
        e = load_aggregate(run)
        s = load_step_aggregate(run)
        body.append(
            [disp, f"{s['mean_stages']:.0f}", f"{e['pca99_mean']:.0f}", *_euclid_cells(e, latex), *_step_cells(s, latex)]
        )
    if not latex:
        headers = ["Model", "Stages", "PCA99", "E gap", "E r1(shuf)", "E dwell", "S gap", "S r1(shuf)", "S dwell"]
        return tabulate(body, headers=headers, tablefmt=tablefmt or "github", disable_numparse=True)
    lines = [
        "\\begin{tabular}{" + "l" * 9 + "}",
        "\\toprule",
        " &  &  & \\multicolumn{3}{c}{Euclidean-based} & \\multicolumn{3}{c}{Step-based} \\\\",
        "\\cmidrule(lr){4-6} \\cmidrule(lr){7-9}",
        "Model & Stages & PCA 99\\% & Gap-ratio & $r_1$ {\\small (shuffle)} & Dwelling \\% "
        "& Gap-ratio & $r_1$ {\\small (shuffle)} & Dwelling \\% \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in body]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def make_figure(out_path: Path) -> None:
    """Bar plot of the dwelling signal (jump autocorrelation) vs model scale, with the shuffle null."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggs = [load_aggregate(run) for _, run in SCALE_ROWS]
    labels = [d for d, _ in SCALE_ROWS]
    acf = [a["jump_autocorr_real_mean"] for a in aggs]
    shuf = [a["jump_autocorr_shuffle_mean"] for a in aggs]
    dwell = [a["fraction_dwelling_basins"] * 100 for a in aggs]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(x, acf, color="#4477aa", width=0.6, label="real (per-sample mean)")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.plot(x, shuf, "o--", color="#cc3311", ms=4, lw=1, label="jump-shuffle null ($\\approx 0$)")
    for xi, (a, d) in enumerate(zip(acf, dwell)):
        ax.text(xi, a + 0.012, f"{d:.0f}\\% dwell", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("jump-magnitude autocorrelation $r_1$", fontsize=9)
    ax.set_title("Dwell-and-jump basin structure emerges with scale", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(-0.05, max(acf) * 1.25 + 0.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tablefmt", default="latex", help="tabulate format for stdout preview (default: latex->github)")
    ap.add_argument("--save", action="store_true", help="Write the LaTeX tables to <save-dir>/<name>.tex")
    ap.add_argument("--save-dir", default="paper/tables")
    ap.add_argument("--figure", action="store_true", help="Also render the scale figure PDF to paper/figures/")
    args = ap.parse_args()

    preview_fmt = "github" if args.tablefmt == "latex" else args.tablefmt
    print("== Model-scale trend (tab:trajectory_cluster_structure) ==")
    print(format_table(SCALE_ROWS, tablefmt=preview_fmt))
    print("\n== Learning-rate sweep (tab:trajectory_cluster_structure_lr) ==")
    print(format_table(LR_ROWS, tablefmt=preview_fmt))

    if args.save:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, spec in (("trajectory_cluster_structure", SCALE_ROWS), ("trajectory_cluster_structure_lr", LR_ROWS)):
            tex = format_table(spec, tablefmt="latex")
            (out_dir / f"{name}.tex").write_text(tex + "\n", encoding="utf-8")
            print(f"Saved 'tab:{name}' to {out_dir / f'{name}.tex'}")

    if args.figure:
        make_figure(Path("paper/figures/trajectory_cluster_dwelling_vs_scale.pdf"))


if __name__ == "__main__":
    main()
