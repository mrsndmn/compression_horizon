#!/usr/bin/env python3
"""Render a schematic of the downstream likelihood evaluation protocol.

Shows a single horizontal chain:
  Compression Embedding + Original Prefix + [stacked suffixes] → Frozen LM → argmin NLL

    python scripts/paper/figures_evaluation_protocol.py
    python scripts/paper/figures_evaluation_protocol.py --poster
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --- Palette (matches poster CSS / visual_abstract) ---
C_ACCENT_DEEP = "#0E2F44"
C_ACCENT = "#1B4F72"
C_ACCENT_MID = "#2E86C1"
C_ACCENT_LIGHT = "#EBF5FB"
C_GOLD = "#D4AC0D"
C_BG = "#F4F6F7"
C_WHITE = "#FFFFFF"
C_BORDER = "#BDC3C7"
C_TEXT = "#1A1A1A"
C_GREEN = "#16a34a"


def _rounded_box(
    ax,
    x,
    y,
    w,
    h,
    label,
    color,
    text_color="white",
    fontsize=13,
    border_color=None,
    linewidth=1.5,
    fontweight="bold",
    zorder=2,
    alpha=1.0,
):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=border_color or color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)
    if label:
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color=text_color,
            zorder=zorder + 1,
        )
    return box


def draw_evaluation_protocol(ax, fontscale=1.0):
    """Draw the evaluation protocol schematic on the given axes."""
    fs = lambda s: s * fontscale  # noqa: E731

    ax.set_xlim(-0.3, 16.5)
    ax.set_ylim(-0.1, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")

    bh = 0.75  # box height
    mid_y = 0.6  # vertical center for the chain

    # --- 1. Compression Embedding ---
    emb_x, emb_w = 0.0, 2.6
    _rounded_box(ax, emb_x, mid_y, emb_w, bh, "Embedding $\\mathbf{e}^*$", C_ACCENT, fontsize=fs(11))

    # "+"
    p1 = emb_x + emb_w + 0.22
    ax.text(
        p1, mid_y + bh / 2, "+", ha="center", va="center", fontsize=fs(16), fontweight="bold", color=C_ACCENT_DEEP, zorder=5
    )

    # --- 2. Original Prefix ---
    pfx_x = p1 + 0.28
    pfx_w = 2.6
    _rounded_box(ax, pfx_x, mid_y, pfx_w, bh, "Original Prefix", C_ACCENT_MID, fontsize=fs(11))

    # "+"
    p2 = pfx_x + pfx_w + 0.22
    ax.text(
        p2, mid_y + bh / 2, "+", ha="center", va="center", fontsize=fs(16), fontweight="bold", color=C_ACCENT_DEEP, zorder=5
    )

    # --- 3. Stacked suffix cards (4 overlapping, back-to-front) ---
    stack_x = p2 + 0.28
    stack_w = 3.0
    stack_h = bh
    n = 4
    # Offsets: back cards peek out from behind the front card
    dx, dy = 0.18, 0.12
    suffix_labels = ["$c_4$", "$c_3$", "$c_2$", "$c_1$"]
    suffix_colors = ["#FCF3CF", "#F0E68C", "#E8D44D", C_GOLD]

    for i in range(n):
        sx = stack_x + (n - 1 - i) * dx
        sy = mid_y + (n - 1 - i) * dy
        is_front = i == n - 1
        _rounded_box(
            ax,
            sx,
            sy,
            stack_w,
            stack_h,
            suffix_labels[i] if is_front else "",
            suffix_colors[i],
            text_color=C_ACCENT_DEEP,
            fontsize=fs(12),
            border_color=C_ACCENT if is_front else C_BORDER,
            linewidth=2.0 if is_front else 1.0,
            zorder=2 + i,
        )

    # Label on front card
    front_x = stack_x
    front_y = mid_y
    ax.text(
        front_x + stack_w / 2,
        front_y + stack_h / 2,
        "Suffix $c_i$",
        ha="center",
        va="center",
        fontsize=fs(12),
        fontweight="bold",
        color=C_ACCENT_DEEP,
        zorder=2 + n + 1,
    )

    # Small count badge
    badge_x = stack_x + stack_w - 0.15
    badge_y = mid_y + stack_h + (n - 1) * dy + 0.05
    ax.text(badge_x, badge_y, "×4", ha="center", va="bottom", fontsize=fs(9), fontweight="bold", color=C_ACCENT, zorder=10)

    # --- Arrow → Frozen LM ---
    arrow_start = stack_x + stack_w + 0.1
    lm_x = arrow_start + 0.55
    lm_w = 2.3
    ax.annotate(
        "",
        xy=(lm_x - 0.05, mid_y + bh / 2),
        xytext=(arrow_start, mid_y + bh / 2),
        arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=2.5),
        zorder=5,
    )

    _rounded_box(ax, lm_x, mid_y, lm_w, bh, "Frozen LM", C_ACCENT_DEEP, fontsize=fs(11))

    # --- Arrow → argmin result ---
    res_arrow_start = lm_x + lm_w + 0.1
    res_x = res_arrow_start + 0.55
    res_w = 3.2
    ax.annotate(
        "",
        xy=(res_x - 0.05, mid_y + bh / 2),
        xytext=(res_arrow_start, mid_y + bh / 2),
        arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=2.5),
        zorder=5,
    )

    _rounded_box(
        ax,
        res_x,
        mid_y,
        res_w,
        bh,
        "$\\hat{c} = \\arg\\min_i \\mathrm{NLL}(c_i)$",
        C_ACCENT_LIGHT,
        text_color=C_ACCENT_DEEP,
        fontsize=fs(10.5),
        border_color=C_ACCENT,
        linewidth=2.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poster", action="store_true", help="Generate poster-sized PNG")
    parser.add_argument("--output", default=None, help="Output path (default: auto)")
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    if args.poster:
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        figsize = (16, 2.5)
        fontscale = 1.4
        out = args.output or "poster/images/evaluation_protocol.png"
    else:
        figsize = (14, 2.2)
        fontscale = 1.0
        out = args.output or "paper/figures/evaluation_protocol.pdf"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    draw_evaluation_protocol(ax, fontscale=fontscale)

    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
