#!/usr/bin/env python
"""Build the CE-temperature figure for the Habr article and the paper.

Parses the committed paper table ``paper/tables/progressive_temperature.tex``
(the published CE-temperature sweep values, kept in sync with
``tab:progressive_temperature``) and renders a single twin-Y-axis figure with one
panel per model (pythia-1.4b, Llama-3.1-8B). Left axis = compressed tokens
(reversed-U, peaking at a cold temperature); right axis = information gain
(peaking at a warmer temperature). The ``raw`` CE(z/T) arm (incl. the T=1 control)
is plotted with std error bars; the raw and T^2 arms agree to within noise (see
tab:progressive_temperature), so only the raw arm is drawn to keep the twin axes
readable.

Style matches the paper's matplotlib figures (default color cycle, larger fonts,
mean line + shaded std band, grid alpha 0.3, no in-figure title -- the caption
carries it). Outputs (regenerate deterministically from the committed table):
  * habr/assets/ce_temperature_tokens_ig.png   -- for the Habr article
  * paper/figures/ce_temperature_tokens_ig.pdf -- for the paper

Run:  python scripts/habr/build_temperature_figures.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = REPO / "paper" / "tables" / "progressive_temperature.tex"
HABR_PNG = REPO / "habr" / "assets" / "ce_temperature_tokens_ig.png"
PAPER_PDF = REPO / "paper" / "figures" / "ce_temperature_tokens_ig.pdf"

# Table prefix -> display name (order = panel order, left to right).
MODELS = [("P1.4b", "Pythia-1.4b"), ("L8b", "Llama-3.1-8B")]

# Match the paper's figure palette (default matplotlib color cycle): C0 blue, C1 orange.
_CYCLE = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
C_TOK = _CYCLE[0]  # compressed tokens (left axis)
C_IG = _CYCLE[1]  # information gain (right axis)

ROW_RE = re.compile(r"^\s*(P1\.4b|L8b)\s+T=([0-9.]+)\s+(raw 50k|raw|t2|control)\s*&(.*?)\\\\")


def _first_float(cell: str) -> float:
    m = re.search(r"[-+]?\d+\.?\d*", cell)
    if not m:
        raise ValueError(f"no number in cell: {cell!r}")
    return float(m.group(0))


def _std(cell: str) -> float:
    m = re.search(r"\\pm\s*\$?\s*([-+]?\d+\.?\d*)", cell)
    return float(m.group(1)) if m else 0.0


def parse_table(path: Path) -> tuple[dict, dict]:
    """Return (raw, hi): raw[prefix] = default-budget raw-arm dicts (T=1 control merged in), sorted by
    T; hi[prefix] = raised-budget ``raw 50k`` re-run dicts (low-T only, overlaid as a second series)."""
    raw: dict[str, list[dict]] = {p: [] for p, _ in MODELS}
    hi: dict[str, list[dict]] = {p: [] for p, _ in MODELS}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        prefix, t_str, arm, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        if arm == "t2":  # raw + control fully describe the plotted curve
            continue
        cells = rest.split("&")
        rec = {
            "T": float(t_str),
            "tokens": _first_float(cells[0]),
            "tokens_std": _std(cells[0]),
            "ig": _first_float(cells[1]),
            "ig_std": _std(cells[1]),
        }
        (hi if arm == "raw 50k" else raw)[prefix].append(rec)
    for d in (raw, hi):
        for prefix in d:
            d[prefix].sort(key=lambda r: r["T"])
    return raw, hi


def _line_with_band(ax, ts, mean, std, color, marker, label, linestyle="-", fillstyle="full", band=True):
    """Mean line + shaded +/-1 std band, matching the paper figures' aesthetic. The raised-budget
    overlay passes linestyle='--', open markers, and band=False to stay visually distinct/uncluttered."""
    if band:
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.fill_between(ts, lo, hi, alpha=0.18, color=color, zorder=1)
    (line,) = ax.plot(
        ts,
        mean,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.5,
        markersize=8,
        color=color,
        label=label,
        fillstyle=fillstyle,
        zorder=3,
    )
    return line


def make_figure(raw: dict, hi: dict, out: Path, dpi: int):
    # Paper-style typography: larger fonts, default color cycle, clean grid.
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 6))
    for ax, (prefix, name) in zip(axes, MODELS):
        rows = raw[prefix]
        ts = [r["T"] for r in rows]

        # Left axis: compressed tokens (reversed-U).
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

        # Right axis: information gain.
        axr = ax.twinx()
        l2 = _line_with_band(axr, ts, [r["ig"] for r in rows], [r["ig_std"] for r in rows], C_IG, "s", "Information Gain")
        axr.set_ylabel("Information Gain (bits)", color=C_IG)
        axr.tick_params(axis="y", labelcolor=C_IG)
        axr.set_ylim(bottom=0)

        # Raised-budget (50k-step) low-T re-run overlay, where present (pythia). Dashed + open markers,
        # no band, so it reads as a distinct series against the default-budget curve.
        handles = [l1, l2]
        hrows = hi.get(prefix, [])
        if hrows:
            hts = [r["T"] for r in hrows]
            l3 = _line_with_band(
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
            l4 = _line_with_band(
                axr,
                hts,
                [r["ig"] for r in hrows],
                [r["ig_std"] for r in hrows],
                C_IG,
                "s",
                "Information Gain (50k budget)",
                linestyle="--",
                fillstyle="none",
                band=False,
            )
            handles += [l3, l4]

        ax.set_title(name)
        ax.legend(handles=handles, loc="lower center")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[ok] wrote {out.relative_to(REPO)} ({out.stat().st_size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    raw, hi = parse_table(args.table)
    make_figure(raw, hi, HABR_PNG, dpi=args.dpi)
    make_figure(raw, hi, PAPER_PDF, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
