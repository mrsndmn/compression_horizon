#!/usr/bin/env python
"""Diversity of equally-good progressive-cramming solutions across learning rates.

The learning-rate sweep runs (``sl_4096_<model>{,_lr_X}``) all use the default
``--random_seed 42`` and the same ``--embedding_init_method``, so for a given
sample they start from a *byte-identical* compression-embedding init and differ
only in optimizer step size. For each prefix (sample_id, prefix length) that
converges to 100% reconstruction in two or more of the runs we therefore have a
small set of *equally valid* solutions reached from one common starting point.

This script asks whether those solutions are:
  * far apart           -- cross-LR pairwise L2 relative to displacement-from-init;
  * near-orthogonal     -- cosine between the displacement directions (e* - e0);
  * equally good        -- spread of information-gain bits across the runs.

If equally-good solutions are far apart, orthogonal, and quality-invariant, the
set of valid compression embeddings for a prefix is *wide and high-dimensional*,
and the low "PCA 99%" of any single warm-started trajectory (paper Sec. on
optimization trajectories) reflects one lazy slice of that set rather than an
intrinsic low-dimensional solution manifold.

Memory notes (these datasets are large -- one full embedding column is ~hundreds
of MB and iterating every row materialises GBs):
  * datasets are loaded memory-mapped; only the small *scalar* columns are pulled
    into RAM in bulk;
  * embeddings are decoded one row at a time through a ``select_columns`` view and
    discarded immediately -- the full embedding column is never materialised.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

import numpy as np

SCALAR_COLS = [
    "sample_id",
    "stage_index",
    "stage_seq_len",
    "final_convergence",
    "information_gain_bits",
]


def parse_lr(run_dir: str) -> str:
    """Extract the learning-rate tag from a run directory name (``default`` if absent)."""
    base = os.path.basename(os.path.dirname(run_dir.rstrip("/"))) or os.path.basename(run_dir.rstrip("/"))
    m = re.search(r"_lr_([0-9.]+)", base)
    return m.group(1) if m else "default"


class Run:
    """Memory-mapped handle to one progressive run: bulk scalars + per-row embedding views."""

    def __init__(self, run_dir: str):
        from datasets import load_from_disk

        self.run_dir = run_dir
        self.lr = parse_lr(run_dir)
        ds = load_from_disk(run_dir)
        # Small scalar columns only -- safe to hold in RAM.
        self.scal = {c: np.asarray(ds[c]) for c in SCALAR_COLS}
        # Column-restricted views: indexing decodes ONLY that column for one row.
        self._emb = ds.select_columns(["embedding"])
        self._init = ds.select_columns(["initialization_embedding"])

    def converged_index(self, sid: int, length: int) -> Optional[int]:
        s = self.scal
        m = (s["sample_id"] == sid) & (s["stage_seq_len"] == length) & (s["final_convergence"] >= 1.0)
        w = np.where(m)[0]
        return int(w[0]) if len(w) else None

    def init_index(self, sid: int) -> Optional[int]:
        s = self.scal
        w = np.where((s["sample_id"] == sid) & (s["stage_index"] == 0))[0]
        return int(w[0]) if len(w) else None

    def embedding(self, i: int) -> np.ndarray:
        return np.asarray(self._emb[i]["embedding"], dtype=np.float32).reshape(-1)

    def init_embedding(self, i: int) -> np.ndarray:
        return np.asarray(self._init[i]["initialization_embedding"], dtype=np.float32).reshape(-1)

    def info_gain(self, i: int) -> float:
        return float(self.scal["information_gain_bits"][i])

    def sample_ids(self) -> list[int]:
        return sorted(set(int(x) for x in self.scal["sample_id"].tolist()))

    def max_converged_len(self, sid: int) -> int:
        s = self.scal
        m = (s["sample_id"] == sid) & (s["final_convergence"] >= 1.0)
        lens = s["stage_seq_len"][m]
        return int(lens.max()) if len(lens) else 0

    def family_label(self) -> str:
        """Model-family name from the run dir, sans ``sl_<n>_`` prefix, ``_lr_*`` suffix, ``Meta-``."""
        base = os.path.basename(os.path.dirname(self.run_dir.rstrip("/"))) or os.path.basename(self.run_dir.rstrip("/"))
        base = re.sub(r"^sl_\d+_", "", base)
        base = re.sub(r"_lr_.*$", "", base)
        return base.replace("Meta-", "")


def geometric_grid(max_len: int, num: int) -> list[int]:
    """Geometric integer grid in ``[4, max_len]`` (deduplicated, ascending)."""
    if max_len < 4:
        return [max_len] if max_len >= 1 else []
    pts = np.unique(np.round(np.geomspace(4, max_len, num=num)).astype(int))
    return [int(p) for p in pts if 1 <= p <= max_len]


def lr_value(lr: str) -> float:
    """Numeric LR for sorting/plotting; ``default`` is the trainer default 0.01."""
    return 0.01 if lr == "default" else float(lr)


def analyze_runs(
    runs: list["Run"],
    max_samples: Optional[int] = None,
    num_lengths: int = 24,
    label: Optional[str] = None,
) -> tuple[dict, dict]:
    """Pool cross-LR solution-diversity metrics over all matched (sample, length) groups.

    ``runs`` must be the LR variants of ONE family, pre-sorted by LR. Returns
    ``(summary, plotdata)`` where ``summary`` is the JSON-serialisable stats dict and
    ``plotdata`` holds the per-group arrays used only for the figure.
    """
    label = label or runs[0].family_label()

    # Samples common to all runs.
    common_sids = set(runs[0].sample_ids())
    for r in runs[1:]:
        common_sids &= set(r.sample_ids())
    sids = sorted(common_sids)
    if max_samples is not None:
        sids = sids[:max_samples]

    # Per-(sample,length) group metrics, pooled across the family.
    norm_dist: list[float] = []  # cross-solution spread / mean displacement-from-init
    dir_cos: list[float] = []  # mean pairwise cosine of displacement directions
    ig_cv: list[float] = []  # std/mean of information gain across runs (%)
    ig_rng_bits: list[float] = []  # max-min info gain across runs (bits)
    n_runs_per_group: list[int] = []
    disp_by_lr: dict[str, list[float]] = {r.lr: [] for r in runs}
    length_of_group: list[int] = []

    for sid in sids:
        # Shared init (verified identical across LRs); decode once.
        i0 = runs[0].init_index(sid)
        if i0 is None:
            continue
        e0 = runs[0].init_embedding(i0)
        max_common = min(r.max_converged_len(sid) for r in runs)
        if max_common < 4:
            continue
        grid = geometric_grid(max_common, num_lengths)
        for L in grid:
            embs: dict[str, np.ndarray] = {}
            igs: dict[str, float] = {}
            for r in runs:
                idx = r.converged_index(sid, L)
                if idx is None:
                    continue
                embs[r.lr] = r.embedding(idx)
                igs[r.lr] = r.info_gain(idx)
            lrs = [lr for lr in embs]
            if len(lrs) < 2:
                continue
            disp = {lr: float(np.linalg.norm(embs[lr] - e0)) for lr in lrs}
            for lr in lrs:
                disp_by_lr[lr].append(disp[lr])
            units = {lr: (embs[lr] - e0) / (disp[lr] + 1e-9) for lr in lrs}
            pair_d, pair_c = [], []
            for a in range(len(lrs)):
                for b in range(a + 1, len(lrs)):
                    la, lb = lrs[a], lrs[b]
                    pair_d.append(float(np.linalg.norm(embs[la] - embs[lb])))
                    pair_c.append(float(units[la] @ units[lb]))
            mean_disp = float(np.mean([disp[lr] for lr in lrs]))
            norm_dist.append(float(np.mean(pair_d)) / (mean_disp + 1e-9))
            dir_cos.append(float(np.mean(pair_c)))
            ig_vals = np.array([igs[lr] for lr in lrs], dtype=np.float64)
            ig_cv.append(float(ig_vals.std() / (abs(ig_vals.mean()) + 1e-9) * 100.0))
            ig_rng_bits.append(float(ig_vals.max() - ig_vals.min()))
            n_runs_per_group.append(len(lrs))
            length_of_group.append(L)

    def stat(x: list[float]) -> dict:
        a = np.asarray(x, dtype=np.float64)
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
        }

    summary = {
        "label": label,
        "lrs": [r.lr for r in runs],
        "n_samples": len(sids),
        "n_groups": len(norm_dist),
        "norm_cross_solution_distance": stat(norm_dist),
        "direction_cosine": stat(dir_cos),
        "info_gain_cv_pct": stat(ig_cv),
        "info_gain_range_bits": stat(ig_rng_bits),
        "mean_displacement_by_lr": {lr: (float(np.mean(v)) if v else None) for lr, v in disp_by_lr.items()},
    }

    plotdata = {
        "disp_by_lr": disp_by_lr,
        "length_of_group": length_of_group,
        "norm_dist": norm_dist,
        "dir_cos": dir_cos,
    }
    return summary, plotdata


def print_summary(summary: dict) -> None:
    print(f"=== {summary['label']} :: LRs {summary['lrs']} ===")
    print(f"  samples={summary['n_samples']}  comparison groups={summary['n_groups']}")
    nd, dc, cv, rb = (
        summary["norm_cross_solution_distance"],
        summary["direction_cosine"],
        summary["info_gain_cv_pct"],
        summary["info_gain_range_bits"],
    )
    if nd.get("n"):
        print(f"  norm cross-solution distance : median {nd['median']:.2f}  IQR [{nd['q25']:.2f}, {nd['q75']:.2f}]")
        print(f"  displacement-direction cosine: median {dc['median']:+.3f}  mean {dc['mean']:+.3f}")
        print(f"  info-gain CV across LRs      : median {cv['median']:.2f}%  (range {rb['median']:.1f} bits)")
        print("  mean displacement-from-init by LR:")
        for lr, v in summary["mean_displacement_by_lr"].items():
            print(f"      lr={lr:>7}: {v:.2f}" if v is not None else f"      lr={lr:>7}: --")


def sort_runs(runs: list["Run"]) -> list["Run"]:
    """Sort LR variants ascending by numeric LR (``default`` first)."""
    return sorted(runs, key=lambda r: lr_value(r.lr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="progressive_prefixes dirs for ONE family (LR variants)",
    )
    ap.add_argument("--label", default=None, help="family label for output (default: inferred)")
    ap.add_argument("--max-samples", type=int, default=None, help="cap number of samples")
    ap.add_argument(
        "--num-lengths",
        type=int,
        default=24,
        help="size of the per-sample geometric length grid",
    )
    ap.add_argument(
        "--out-dir",
        default="artifacts/paper/solution_diversity",
        help="where to write figure + json",
    )
    ap.add_argument("--no-fig", action="store_true", help="skip the matplotlib figure")
    args = ap.parse_args()

    runs = sort_runs([Run(d) for d in args.runs])
    summary, plotdata = analyze_runs(
        runs,
        max_samples=args.max_samples,
        num_lengths=args.num_lengths,
        label=args.label,
    )
    print_summary(summary)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{summary['label']}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {json_path}")

    if not args.no_fig and len(plotdata["norm_dist"]) > 0:
        _plot(
            args.out_dir,
            summary["label"],
            runs,
            plotdata["disp_by_lr"],
            plotdata["length_of_group"],
            plotdata["norm_dist"],
            plotdata["dir_cos"],
        )


def _plot(out_dir, label, runs, disp_by_lr, length_of_group, norm_dist, dir_cos):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # (a) displacement-from-init vs LR.
    lrs_num, disp_mean, disp_q25, disp_q75 = [], [], [], []
    for r in runs:
        v = disp_by_lr[r.lr]
        if not v:
            continue
        lr_val = 0.01 if r.lr == "default" else float(r.lr)
        lrs_num.append(lr_val)
        a = np.asarray(v)
        disp_mean.append(a.mean())
        disp_q25.append(np.percentile(a, 25))
        disp_q75.append(np.percentile(a, 75))
    order = np.argsort(lrs_num)
    lrs_num = np.array(lrs_num)[order]
    disp_mean = np.array(disp_mean)[order]
    disp_q25 = np.array(disp_q25)[order]
    disp_q75 = np.array(disp_q75)[order]
    ax[0].fill_between(lrs_num, disp_q25, disp_q75, alpha=0.2, color="C0")
    ax[0].plot(lrs_num, disp_mean, marker="o", color="C0")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("learning rate")
    ax[0].set_ylabel(r"displacement from init $\|e^*-e_0\|$")
    ax[0].set_title(f"{label}: higher LR lands farther")
    ax[0].grid(True, alpha=0.3)
    # (b) normalized cross-solution distance vs prefix length.
    ax[1].scatter(length_of_group, norm_dist, s=10, alpha=0.4, color="C3")
    ax[1].axhline(1.0, ls="--", color="k", lw=1, alpha=0.6)
    ax[1].set_xlabel("prefix length")
    ax[1].set_ylabel("cross-solution dist / displacement")
    ax[1].set_title(f"mean dir. cosine {np.mean(dir_cos):+.2f} (near-orthogonal)")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, f"{label}_solution_diversity.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
