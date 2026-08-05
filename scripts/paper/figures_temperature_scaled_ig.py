#!/usr/bin/env python
"""Temperature-SCALED information-gain version of fig:ce_temperature.

The published figure (``scripts/habr/build_temperature_figures.py`` ->
``ce_temperature_tokens_ig.pdf``) plots the IG column of ``tab:progressive_temperature``, which the
trainer computes on the **raw** model logits (a plain ``T=1`` cross-entropy) in every run of the
sweep -- the temperature never enters the IG measurement, only the training loss. Because a run
trained under ``CE(z/T)`` ends with reconstruction logits scaled as ``z ~ T``, reading them back at
``T=1`` makes cold runs look under-confident and understates their IG.

This script redraws the same tokens-vs-IG figure with the IG recomputed under temperature scaling
by ``scripts/paper/tables/temperature_information_gain.py`` (cache:
``artifacts/paper/temperature_information_gain.json``). Two rows, one per scaling convention:

  * top    ``loss units``  H_T(base) - H_T(mem): both sides read through the softmax the loss
                           optimizes. Literal "divide the logits by T" -- but the tempered base LM
                           is itself a poor code at cold T, which dominates the curve.
  * bottom ``comp only``   H_1(base) - H_T(mem): only the crammed side is put back on the scale its
                           own loss shaped it at, keeping the base LM's ordinary T=1 code as the
                           reference. This is the apples-to-apples "bits saved" reading; the
                           published raw-logit curve is overlaid dotted for comparison.

Compressed-token counts (left axis, blue) are parsed from the committed table, so they are identical
to the published figure. Run:  python scripts/paper/figures_temperature_scaled_ig.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "habr"))
from build_temperature_figures import C_IG, C_TOK, MODELS, _line_with_band, parse_table  # noqa: E402

CACHE = REPO / "artifacts" / "paper" / "temperature_information_gain.json"
DEFAULT_TABLE = REPO / "paper" / "tables" / "progressive_temperature.tex"
OUT_PDF = REPO / "artifacts" / "paper" / "figures" / "ce_temperature_tokens_ig_scaled.pdf"
OUT_PNG = REPO / "artifacts" / "paper" / "figures" / "ce_temperature_tokens_ig_scaled.png"

C_RAWREF = "0.45"  # published raw-logit IG, drawn as a grey dotted reference

# (cache key stem, row title, whether to overlay the published raw-logit IG)
ROWS = [
    ("ig_loss_units", r"IG in loss units:  $H_T(\mathrm{base}) - H_T(\mathrm{mem})$", False),
    ("ig_comp_only", r"IG, mem side rescaled:  $H_1(\mathrm{base}) - H_T(\mathrm{mem})$", True),
]


def load_ig(cache_path: Path) -> dict:
    """(prefix, T, arm) -> entry, from the recomputed-IG cache."""
    with open(cache_path) as f:
        cache = json.load(f)
    return {(e["model"], e["T"], e["arm"]): e for e in cache.values()}


def _attach(rows: list[dict], ig: dict, prefix: str, arm: str, key: str) -> list[dict]:
    """Rows from the table, with the IG mean/std swapped for the recomputed ``key`` variant."""
    out = []
    for r in rows:
        e = ig.get((prefix, r["T"], arm))
        if e is None:
            continue
        out.append({**r, "ig": e[f"{key}_mean"], "ig_std": e[f"{key}_std"]})
    return out


def make_figure(raw: dict, hi: dict, ig: dict, out: Path, dpi: int):
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(len(ROWS), len(MODELS), figsize=(7.5 * len(MODELS), 6 * len(ROWS)))
    for row_i, (key, row_title, show_raw_ref) in enumerate(ROWS):
        for col_i, (prefix, name) in enumerate(MODELS):
            ax = axes[row_i][col_i]
            # The T=1.0 control row is stored under arm "control"; every other T under "raw".
            rows = _attach(raw[prefix], ig, prefix, "raw", key) + _attach(raw[prefix], ig, prefix, "control", key)
            rows.sort(key=lambda r: r["T"])
            ts = [r["T"] for r in rows]

            l1 = _line_with_band(
                ax, ts, [r["tokens"] for r in rows], [r["tokens_std"] for r in rows], C_TOK, "o", "Compressed Tokens"
            )
            ax.set_xlabel(r"Cross-Entropy Temperature $T$")
            ax.set_ylabel("Compressed Tokens", color=C_TOK)
            ax.tick_params(axis="y", labelcolor=C_TOK)
            ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

            axr = ax.twinx()
            handles = [l1]
            if show_raw_ref:  # unscaled (raw-logit, T=1) IG for comparison, same units
                unscaled = _attach(raw[prefix], ig, prefix, "raw", "ig_raw")
                unscaled += _attach(raw[prefix], ig, prefix, "control", "ig_raw")
                unscaled = {r["T"]: r["ig"] for r in unscaled}
                handles.append(
                    _line_with_band(
                        axr,
                        ts,
                        [unscaled[t] for t in ts],
                        [0.0] * len(ts),
                        C_RAWREF,
                        "s",
                        "Information Gain (raw logits, unscaled)",
                        linestyle=":",
                        fillstyle="none",
                        band=False,
                    )
                )
            handles.append(
                _line_with_band(
                    axr,
                    ts,
                    [r["ig"] for r in rows],
                    [r["ig_std"] for r in rows],
                    C_IG,
                    "s",
                    "Information Gain (scaled)",
                )
            )
            axr.set_ylabel("Information Gain (bits)", color=C_IG)
            axr.tick_params(axis="y", labelcolor=C_IG)
            axr.set_ylim(bottom=0)
            axr.relim()
            axr.autoscale_view()

            # Raised-budget (50k-step) low-T re-run overlay, where present (pythia).
            hrows = _attach(hi.get(prefix, []), ig, prefix, "raw 50k", key)
            if hrows:
                hts = [r["T"] for r in hrows]
                handles.append(
                    _line_with_band(
                        ax,
                        hts,
                        [r["tokens"] for r in hrows],
                        [r["tokens_std"] for r in hrows],
                        C_TOK,
                        "o",
                        "Compressed Tokens (50k budget)",
                        linestyle="--",
                        fillstyle="none",
                        band=False,
                    )
                )
                handles.append(
                    _line_with_band(
                        axr,
                        hts,
                        [r["ig"] for r in hrows],
                        [r["ig_std"] for r in hrows],
                        C_IG,
                        "s",
                        "Information Gain (scaled, 50k budget)",
                        linestyle="--",
                        fillstyle="none",
                        band=False,
                    )
                )

            ax.set_title(name if row_i == 0 else "")
            ax.legend(handles=handles, loc="lower center")
            if col_i == 0:
                ax.text(
                    -0.16,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=15,
                )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[ok] wrote {out.relative_to(REPO)} ({out.stat().st_size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    raw, hi = parse_table(args.table)
    ig = load_ig(args.cache)
    make_figure(raw, hi, ig, OUT_PDF, dpi=args.dpi)
    make_figure(raw, hi, ig, OUT_PNG, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
