#!/usr/bin/env python
"""Build CE-temperature figures for the Habr article.

Parses the committed paper table ``paper/tables/progressive_temperature.tex``
(the published CE-temperature sweep values, kept in sync with
``tab:progressive_temperature``) and renders two figures into ``habr/assets/``:

  * ``ce_temperature_compressed_tokens.png`` -- reversed-U: compressed tokens vs T
  * ``ce_temperature_information_gain.png``  -- information gain (bits) vs T

Each figure has one panel per model (pythia-1.4b, Llama-3.1-8B). Within a panel we
draw the ``raw`` loss CE(z/T) and the ``T^2``-compensated arm, both sharing the
single ``T=1.0`` control point, with std error bars. The per-model peak is marked.

Regenerates deterministically from the committed table, so the figures always
match the paper numbers. Run:

    python scripts/habr/build_temperature_figures.py
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
DEFAULT_OUTDIR = REPO / "habr" / "assets"

# Table prefix -> display name (order = panel order, left to right).
MODELS = [("P1.4b", "Pythia-1.4b"), ("L8b", "Llama-3.1-8B")]

# AXXX-ish palette.
C_RAW = "#0689D4"  # raw CE(z/T)
C_T2 = "#C0392B"  # T^2-compensated
C_PEAK = "#687A86"  # peak marker

ROW_RE = re.compile(r"^\s*(P1\.4b|L8b)\s+T=([0-9.]+)\s+(raw|t2|control)\s*&(.*?)\\\\")


def _first_float(cell: str) -> float:
    m = re.search(r"[-+]?\d+\.?\d*", cell)
    if not m:
        raise ValueError(f"no number in cell: {cell!r}")
    return float(m.group(0))


def _std(cell: str) -> float:
    m = re.search(r"\\pm\s*\$?\s*([-+]?\d+\.?\d*)", cell)
    return float(m.group(1)) if m else 0.0


def parse_table(path: Path) -> dict:
    """Return data[prefix][arm] = list of dicts sorted by T (control merged in)."""
    raw: dict[str, dict[str, list[dict]]] = {p: {"raw": [], "t2": []} for p, _ in MODELS}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        prefix, t_str, arm, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        cells = [c for c in rest.split("&")]
        rec = {
            "T": float(t_str),
            "tokens": _first_float(cells[0]),
            "tokens_std": _std(cells[0]),
            "ig": _first_float(cells[1]),
            "ig_std": _std(cells[1]),
        }
        if arm == "control":  # shared by both arms
            raw[prefix]["raw"].append(rec)
            raw[prefix]["t2"].append(dict(rec))
        else:
            raw[prefix][arm].append(rec)
    for prefix in raw:
        for arm in raw[prefix]:
            raw[prefix][arm].sort(key=lambda r: r["T"])
    return raw


def _series(rows: list[dict], key: str):
    xs = [r["T"] for r in rows]
    ys = [r[key] for r in rows]
    es = [r[f"{key}_std"] for r in rows]
    return xs, ys, es


def make_figure(data: dict, metric: str, ylabel: str, suptitle: str, out: Path, dpi: int):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.2))
    for ax, (prefix, name) in zip(axes, MODELS):
        raw_rows = data[prefix]["raw"]
        t2_rows = data[prefix]["t2"]

        xr, yr, er = _series(raw_rows, metric)
        xt, yt, et = _series(t2_rows, metric)
        ax.errorbar(
            xr,
            yr,
            yerr=er,
            color=C_RAW,
            marker="o",
            ms=6,
            lw=2.2,
            capsize=3,
            label=r"raw  $\mathrm{CE}(\mathbf{z}/T)$",
            zorder=3,
        )
        ax.errorbar(
            xt,
            yt,
            yerr=et,
            color=C_T2,
            marker="s",
            ms=5,
            lw=1.6,
            ls="--",
            capsize=3,
            alpha=0.85,
            label=r"$T^2$-компенсир.",
            zorder=2,
        )

        # Mark the per-model peak (of the raw arm) with a vertical guide + label.
        peak = max(raw_rows, key=lambda r: r[metric])
        ax.axvline(peak["T"], color=C_PEAK, ls=":", lw=1.4, zorder=1)
        ax.annotate(
            f"пик: T={peak['T']:g}\n({peak[metric]:.0f})",
            xy=(peak["T"], peak[metric]),
            xytext=(6, -4),
            textcoords="offset points",
            fontsize=9,
            color=C_PEAK,
            fontweight="bold",
            ha="left",
            va="top",
        )
        # Highlight the T=1 control point (plain cross-entropy baseline).
        ctrl = next(r for r in raw_rows if r["T"] == 1.0)
        ax.scatter([1.0], [ctrl[metric]], s=90, facecolors="none", edgecolors="#111111", linewidths=1.4, zorder=4)
        ax.annotate(
            "T=1 (обычная CE)",
            xy=(1.0, ctrl[metric]),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=8,
            color="#111111",
            ha="center",
        )

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Температура кросс-энтропии $T$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.04)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out.relative_to(REPO)} ({out.stat().st_size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    data = parse_table(args.table)
    make_figure(
        data,
        "tokens",
        ylabel="Число сжатых токенов",
        suptitle="Число сжатых токенов vs температура: перевёрнутая парабола с пиком при холодном $T$",
        out=args.outdir / "ce_temperature_compressed_tokens.png",
        dpi=args.dpi,
    )
    make_figure(
        data,
        "ig",
        ylabel="Information gain (бит)",
        suptitle=r"Information gain vs температура: пик при более тёплом $T\approx0.75$",
        out=args.outdir / "ce_temperature_information_gain.png",
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
