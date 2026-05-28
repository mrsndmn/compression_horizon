"""Generate ``tab:solution_diversity``: are equally-good cramming solutions far apart?

The learning-rate sweep runs (``sl_4096_<model>{,_lr_X}``) share the default ``--random_seed 42``
and the same ``--embedding_init_method``, so for a given sample every LR starts from a *byte-identical*
compression-embedding init ``e_0`` and differs only in optimizer step size. For each prefix
(``sample_id``, prefix length) that converges to 100% reconstruction in two or more of the runs we
therefore obtain a small set of *equally valid* solutions reached from one common starting point.

This table pools, over all matched (sample, length) groups and four model families, three things:

  * **IG CV** -- coefficient of variation (%) of the information-gain bits across the runs: are the
    solutions equally good? (They are: a few percent.)
  * **Sol. dist.** -- mean cross-LR pairwise ``L2`` distance between the converged solutions, in units
    of their mean displacement-from-init ``\\|e^*-e_0\\|`` (so it is scale-free, not a step-size artifact);
    ``> 1`` means two equally-good solutions are *farther from each other than from the shared start*.
  * **Dir. cos.** -- mean pairwise cosine of the displacement directions ``(e^*-e_0)``; the random
    high-dimensional baseline is ``~1/sqrt(d)`` (0.016-0.022 for these hidden sizes).
  * **Disp. ratio** -- mean ``\\|e^*-e_0\\|`` at the largest LR divided by that at the smallest: how much
    farther a higher LR travels.

Takeaway: equally-good solutions are far apart and near-orthogonal, so the valid-solution set for a
prefix is wide and high-dimensional. The low "PCA 99%" of a single warm-started trajectory
(Sec.~optimization-trajectories) is therefore a property of one lazy *path*, not of an intrinsic
low-dimensional solution manifold -- which is exactly why low-dimensional projection preserves quality.

The embedding-decoding pass is gated behind ``--compute`` and cached to
``artifacts/paper/solution_diversity/<key>.json`` (one file per family). Without ``--compute`` the
table is rendered from those caches, so ``make tables`` regenerates the ``.tex`` without re-reading
the (large) trajectory datasets.

    # one-time (reads the LR-sweep trajectory datasets; CPU only, memory-frugal):
    PYTHONPATH=./src:. python scripts/paper/tables/solution_diversity.py --compute --save
    # cheap re-render from cache (what tables.sh runs):
    PYTHONPATH=./src:. python scripts/paper/tables/solution_diversity.py --save
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from scripts.analyze_solution_diversity import Run, analyze_runs, sort_runs
from tabulate import tabulate

from compression_horizon.utils import hlines_to_booktabs

_EXP = "artifacts/experiments_progressive"
_CACHE_DIR = "artifacts/paper/solution_diversity"

# (display name, cache key, [run dirs: default + lr_0.1 + lr_0.5 + lr_1.0]).
FAMILIES: List[Tuple[str, str, List[str]]] = [
    (
        "Llama-3.1-8B",
        "Llama-3.1-8B",
        [f"{_EXP}/sl_4096_Meta-Llama-3.1-8B{s}/progressive_prefixes" for s in ("", "_lr_0.1", "_lr_0.5", "_lr_1.0")],
    ),
    (
        "Pythia-1.4B",
        "pythia-1.4b",
        [f"{_EXP}/sl_4096_pythia-1.4b{s}/progressive_prefixes" for s in ("", "_lr_0.1", "_lr_0.5", "_lr_1.0")],
    ),
    (
        "SmolLM2-1.7B",
        "SmolLM2-1.7B",
        [f"{_EXP}/sl_4096_SmolLM2-1.7B{s}/progressive_prefixes" for s in ("", "_lr_0.1", "_lr_0.5", "_lr_1.0")],
    ),
    (
        "Gemma-3-4B",
        "gemma-3-4b-pt",
        [f"{_EXP}/sl_4096_gemma-3-4b-pt{s}/progressive_prefixes" for s in ("", "_lr_0.1", "_lr_0.5", "_lr_1.0")],
    ),
]


def cache_path(key: str) -> str:
    return os.path.join(_CACHE_DIR, f"{key}.json")


def compute_and_cache(key: str, run_dirs: List[str], force: bool = False) -> dict:
    """Decode the LR-sweep trajectories for one family and cache the pooled diversity stats."""
    path = cache_path(key)
    if os.path.exists(path) and not force:
        print(f"  cache exists, skipping compute: {path}")
        with open(path) as f:
            return json.load(f)
    dirs = [d for d in run_dirs if os.path.isdir(d)]
    if len(dirs) < 2:
        raise SystemExit(f"{key}: need >=2 existing LR runs, found {len(dirs)} of {len(run_dirs)}")
    runs = sort_runs([Run(d) for d in dirs])
    summary, _ = analyze_runs(runs, label=key)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"  saved cache: {path}  (groups={summary['n_groups']}, sol.dist={summary['norm_cross_solution_distance'].get('median')})"
    )
    return summary


def load_cache(key: str) -> Optional[dict]:
    path = cache_path(key)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _disp_ratio(summary: dict) -> float:
    """Mean displacement-from-init at the largest LR over that at the smallest (LRs sorted ascending)."""
    disp = summary["mean_displacement_by_lr"]
    vals = [(0.01 if lr == "default" else float(lr), v) for lr, v in disp.items() if v is not None]
    vals.sort()
    if len(vals) < 2 or vals[0][1] in (0, None):
        return float("nan")
    return vals[-1][1] / vals[0][1]


def format_table(displays: List[str], summaries: List[dict], tablefmt: str = "latex") -> str:
    latex = tablefmt == "latex"
    rows = []
    for name, st in zip(displays, summaries):
        nd = st["norm_cross_solution_distance"]
        dc = st["direction_cosine"]
        cv = st["info_gain_cv_pct"]
        ratio = _disp_ratio(st)
        dist_cell = (
            f"{nd['median']:.2f} {{\\small [{nd['q25']:.2f}, {nd['q75']:.2f}]}}"
            if latex
            else f"{nd['median']:.2f} [{nd['q25']:.2f}, {nd['q75']:.2f}]"
        )
        rows.append(
            [
                name,
                f"{st['n_groups']}",
                f"{cv['median']:.1f}\\%" if latex else f"{cv['median']:.1f}%",
                dist_cell,
                f"{dc['median']:+.2f}",
                f"{ratio:.0f}$\\times$" if latex else f"{ratio:.0f}x",
            ]
        )
    headers = [
        "Model",
        "$N$",
        "IG CV",
        "Sol.\\ dist." if latex else "Sol. dist.",
        "Dir.\\ cos." if latex else "Dir. cos.",
        "Disp.\\ ratio" if latex else "Disp. ratio",
    ]
    fmt = "latex_raw" if latex else (tablefmt or "github")
    # disable_numparse: keep our pre-formatted cells (``+0.10``, ``0.9\%``, ``15$\times$``) verbatim
    # instead of letting tabulate re-parse them as numbers and drop signs / trailing zeros.
    result = tabulate(
        rows,
        headers=headers,
        tablefmt=fmt,
        colalign=["left"] * len(headers),
        disable_numparse=True,
    )
    if latex:
        result = hlines_to_booktabs(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Decode the LR-sweep trajectories and (re)write per-family caches.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a cache exists (requires --compute).",
    )
    parser.add_argument(
        "--tablefmt",
        default="latex",
        help="tabulate format for stdout (default: latex).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write the rendered LaTeX table to <save-dir>/<save-name>.tex.",
    )
    parser.add_argument("--save-dir", default="paper/tables")
    parser.add_argument("--save-name", default="solution_diversity")
    args = parser.parse_args()
    if args.force and not args.compute:
        parser.error("--force requires --compute")

    displays, summaries = [], []
    for display, key, dirs in FAMILIES:
        print(f"== {display} ==")
        if args.compute:
            st = compute_and_cache(key, dirs, force=args.force)
        else:
            st = load_cache(key)
            if st is None:
                raise SystemExit(f"No cache at {cache_path(key)}. Run once with --compute.")
        displays.append(display)
        summaries.append(st)

    print("\n" + format_table(displays, summaries, tablefmt="github"))

    if args.save:
        tex = format_table(displays, summaries, tablefmt="latex")
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.save_name}.tex"
        out_path.write_text(tex + ("\n" if not tex.endswith("\n") else ""), encoding="utf-8")
        print(f"\nSaved 'tab:{args.save_name}' to {out_path}")


if __name__ == "__main__":
    main()
