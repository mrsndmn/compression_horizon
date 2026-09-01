"""Explainer figures for the CE-temperature reconstruction logit geometry (tab:temperature_logit_stats).

Emits two standalone SVGs into ``docs/figures/``:

  * ``temperature_logit_geometry_idea.svg``     -- what the experiment does and why the
    normalized/unnormalized top-1 distinction exists.
  * ``temperature_logit_geometry_results.svg``  -- what the 42 runs show.

These are documentation figures, NOT paper figures: ``paper/lint_paper.py`` fails on any
unreferenced file under ``paper/figures/``, so they deliberately live under ``docs/``.

Both read the committed-by-regeneration caches (no GPU):
  * ``artifacts/paper/temperature_logit_stats.json``       -- margin/top-1 geometry
  * ``artifacts/paper/temperature_information_gain.json``  -- compressed tokens + IG

Usage:
    python scripts/paper/figures_temperature_logit_geometry.py
"""

import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_LOGIT = os.path.join(REPO, "artifacts", "paper", "temperature_logit_stats.json")
CACHE_IG = os.path.join(REPO, "artifacts", "paper", "temperature_information_gain.json")
OUT_DIR = os.path.join(REPO, "docs", "figures")

# Reference data-viz palette (validated: 3 categorical slots pass all-pairs in light mode;
# aqua sits below 3:1 on the surface, so every series carries a visible direct label).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
POS = "#2a78d6"  # diverging warm/cool pair, blue <-> red
NEG = "#e34948"
ZERO = "#f0efec"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

MODELS = [
    ("P1.4b", "pythia-1.4b", "#2a78d6"),  # slot 1 blue
    ("L8b", "Llama-3.1-8B", "#eb6834"),  # slot 2 orange
    ("Q4b", "Qwen3-4B", "#1baf7a"),  # slot 3 aqua
]
TEMPS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def log_scale(d0, d1, r0, r1):
    l0, l1 = math.log10(d0), math.log10(d1)
    return lambda v: r0 + (math.log10(v) - l0) / (l1 - l0) * (r1 - r0)


def lin_scale(d0, d1, r0, r1):
    return lambda v: r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def load():
    """Main-arm series per model (arm 'raw', or 'control' at T=1.0) + the 50k budget arm."""
    logit = json.load(open(CACHE_LOGIT, encoding="utf-8"))
    ig = json.load(open(CACHE_IG, encoding="utf-8"))
    series = {}
    for prefix, _, _ in MODELS:
        pts = []
        for T in TEMPS:
            for arm in ("raw", "control"):
                e = logit.get(f"{prefix} T={T} {arm}")
                if e:
                    pts.append(
                        {
                            "T": T,
                            "margin_zt": e["margin_raw"] / T,
                            "top1_zt": e["top1_raw"] / T,
                            "p_top1": math.exp(e["top1_logp_norm"]),
                            "steps": e["avg_steps"],
                        }
                    )
                    break
        series[prefix] = pts

    # 50k-step budget arm (pythia only, cold temperatures): 5x budget vs the 10k default.
    budget = []
    for T in TEMPS:
        base_l, long_l = logit.get(f"P1.4b T={T} raw"), logit.get(f"P1.4b T={T} raw 50k")
        base_i, long_i = ig.get(f"P1.4b T={T} raw"), ig.get(f"P1.4b T={T} raw 50k")
        if not (base_l and long_l and base_i and long_i):
            continue
        budget.append(
            {
                "T": T,
                "tokens": (base_i["n_tokens_mean"], long_i["n_tokens_mean"]),
                "ig": (base_i["ig_comp_only_mean"], long_i["ig_comp_only_mean"]),
                "margin": (base_l["margin_raw"] / T, long_l["margin_raw"] / T),
                "p_top1": (math.exp(base_l["top1_logp_norm"]), math.exp(long_l["top1_logp_norm"])),
                "steps": (base_l["avg_steps"], long_l["avg_steps"]),
            }
        )
    return series, budget


# --------------------------------------------------------------------------- #
# Shared SVG chrome
# --------------------------------------------------------------------------- #
def panel(x, y, w, h, title, subtitle):
    o = [
        f'<text x="{x}" y="{y - 26}" font-family="{FONT}" font-size="16" font-weight="600" fill="{INK}">{esc(title)}</text>',
        f'<text x="{x}" y="{y - 8}" font-family="{FONT}" font-size="12.5" fill="{INK2}">{esc(subtitle)}</text>',
    ]
    return o


def t_axis(x0, x1, ybase, sx):
    o = [f'<line x1="{x0}" y1="{ybase}" x2="{x1}" y2="{ybase}" stroke="{AXIS}" stroke-width="1"/>']
    for T in (0.1, 0.25, 0.5, 1.0, 2.0):
        px = sx(T)
        o.append(f'<line x1="{px:.1f}" y1="{ybase}" x2="{px:.1f}" y2="{ybase + 5}" stroke="{AXIS}" stroke-width="1"/>')
        o.append(
            f'<text x="{px:.1f}" y="{ybase + 19}" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
            f'fill="{MUTED}" style="font-variant-numeric:tabular-nums">{T:g}</text>'
        )
    o.append(
        f'<text x="{(x0 + x1) / 2:.1f}" y="{ybase + 38}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="12" fill="{INK2}">CE temperature T (log scale)</text>'
    )
    return o


def y_grid(x0, x1, sy, ticks, fmt="{:g}"):
    o = []
    for v in ticks:
        py = sy(v)
        o.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x1}" y2="{py:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(
            f'<text x="{x0 - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="{FONT}" font-size="11.5" '
            f'fill="{MUTED}" style="font-variant-numeric:tabular-nums">{fmt.format(v)}</text>'
        )
    return o


def series_lines(pts_by_model, sx, sy, key, x1, label_dy=None):
    """2px polylines + 8px ringed markers + direct endpoint labels (relief for low-contrast hues)."""
    o = []
    for prefix, _, color in MODELS:
        pts = pts_by_model[prefix]
        d = " ".join(f"{sx(p['T']):.1f},{sy(p[key]):.1f}" for p in pts)
        o.append(
            f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for p in pts:
            o.append(
                f'<circle cx="{sx(p["T"]):.1f}" cy="{sy(p[key]):.1f}" r="4" fill="{color}" '
                f'stroke="{SURFACE}" stroke-width="2"/>'
            )
        last = pts[-1]
        dy = (label_dy or {}).get(prefix, 0)
        lx, ly = sx(last["T"]) + 12, sy(last[key]) + 4 + dy
        o.append(f'<circle cx="{lx + 4:.1f}" cy="{ly - 4:.1f}" r="3.5" fill="{color}"/>')
        o.append(
            f'<text x="{lx + 13:.1f}" y="{ly:.1f}" font-family="{FONT}" font-size="11.5" '
            f'font-weight="600" fill="{INK2}">{esc(prefix)}</text>'
        )
    return o


def legend(x, y):
    o = []
    for i, (prefix, full, color) in enumerate(MODELS):
        cx = x + i * 186
        o.append(f'<line x1="{cx}" y1="{y}" x2="{cx + 22}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        o.append(f'<circle cx="{cx + 11}" cy="{y}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        o.append(
            f'<text x="{cx + 30}" y="{y + 4}" font-family="{FONT}" font-size="12.5" fill="{INK2}">'
            f'<tspan font-weight="600">{esc(prefix)}</tspan> {esc(full)}</text>'
        )
    return o


# --------------------------------------------------------------------------- #
# Figure 1: results
# --------------------------------------------------------------------------- #
def fig_results(series, budget):
    W, H = 1280, 1000
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" ' f'font-family="{FONT}">',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="56" y="52" font-size="24" font-weight="700" fill="{INK}">'
        f"Reconstruction logit geometry vs. CE temperature</text>",
        f'<text x="56" y="78" font-size="14" fill="{INK2}">'
        f"Converged [mem] embedding, final stage, 50 PG19 samples per run &#183; 42 runs "
        f"(3 models &#215; 7 temperatures &#215; arms)</text>",
    ]
    o += legend(56, 106)

    PW, PH = 520, 250
    # --- A: unnormalized top-1 ------------------------------------------------
    ax, ay = 116, 176
    sx = log_scale(0.1, 2.0, ax, ax + PW - 66)
    sy = log_scale(10, 220, ay + PH, ay)
    o += panel(ax, ay, PW, PH, "A. Top-1 z/T — UNNORMALIZED", "raw winning logit over T; spans ~10×")
    o += y_grid(ax, ax + PW, sy, [10, 20, 50, 100, 200])
    o += series_lines(series, sx, sy, "top1_zt", ax + PW)
    o += t_axis(ax, ax + PW, ay + PH, sx)

    # --- B: normalized top-1 --------------------------------------------------
    bx, by = 716, 176
    sx2 = log_scale(0.1, 2.0, bx, bx + PW - 66)
    sy2 = lin_scale(0.5, 1.0, by + PH, by)
    o += panel(bx, by, PW, PH, "B. Top-1 probability — NORMALIZED", "softmax(z/T) mass on the winner; near-flat")
    o += y_grid(bx, bx + PW, sy2, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "{:.1f}")
    o += series_lines(series, sx2, sy2, "p_top1", bx + PW, {"L8b": -12, "Q4b": 4})
    o += t_axis(bx, bx + PW, by + PH, sx2)
    o.append(
        f'<text x="{bx + 12}" y="{by + 30}" font-size="12" fill="{MUTED}">'
        f"argmax convergence keeps every crammed token on top at every T</text>"
    )

    # --- C: margin ------------------------------------------------------------
    cx, cy = 116, 556
    sx3 = log_scale(0.1, 2.0, cx, cx + PW - 66)
    sy3 = log_scale(3, 30, cy + PH, cy)
    o += panel(cx, cy, PW, PH, "C. Margin z/T", "(z_true − runner-up)/T; flat for T ≳ 0.5, floor rises when cold")
    o += y_grid(cx, cx + PW, sy3, [3, 5, 10, 20, 30])
    o += series_lines(series, sx3, sy3, "margin_zt", cx + PW)
    o += t_axis(cx, cx + PW, cy + PH, sx3)

    # --- D: 5x step budget ----------------------------------------------------
    dx, dy0 = 716, 556
    o += panel(
        dx,
        dy0,
        PW,
        PH,
        "D. 5× step budget (pythia, T ≤ 0.5)",
        "change from 10k to 50k steps/sample, averaged over T = 0.1/0.25/0.5",
    )
    metrics = [
        ("Compressed tokens", "tokens"),
        ("Information gain", "ig"),
        ("Margin z/T", "margin"),
        ("Top-1 probability", "p_top1"),
    ]
    deltas = []
    for label, key in metrics:
        vals = [(b[key][1] / b[key][0] - 1.0) * 100.0 for b in budget]
        deltas.append((label, sum(vals) / len(vals)))
    span = 20.0
    zero_x = dx + 250
    sxd = lin_scale(-span, span, dx + 40, dx + PW - 30)
    row_h = 46
    for i, (label, pct) in enumerate(deltas):
        ry = dy0 + 34 + i * row_h
        color = POS if pct >= 0 else NEG
        x_from, x_to = sxd(min(0, pct)), sxd(max(0, pct))
        o.append(f'<text x="{dx + 40}" y="{ry - 9}" font-size="12.5" fill="{INK2}">{esc(label)}</text>')
        o.append(
            f'<rect x="{x_from:.1f}" y="{ry}" width="{max(2.0, x_to - x_from):.1f}" height="14" ' f'rx="4" fill="{color}"/>'
        )
        tx = x_to + 8 if pct >= 0 else x_from - 8
        anchor = "start" if pct >= 0 else "end"
        o.append(
            f'<text x="{tx:.1f}" y="{ry + 12}" text-anchor="{anchor}" font-size="12.5" font-weight="600" '
            f'fill="{INK}" style="font-variant-numeric:tabular-nums">{pct:+.1f}%</text>'
        )
    ytop, ybot = dy0 + 20, dy0 + 34 + len(deltas) * row_h - 12
    o.append(f'<line x1="{zero_x}" y1="{ytop}" x2="{zero_x}" y2="{ybot}" stroke="{AXIS}" stroke-width="1"/>')
    o.append(f'<text x="{zero_x}" y="{ybot + 18}" text-anchor="middle" font-size="11.5" fill="{MUTED}">' f"no change</text>")
    o.append(
        f'<text x="{dx + 40}" y="{ybot + 44}" font-size="12" fill="{INK2}">'
        f"5× more steps buys ~12% more crammed tokens — and no extra per-token confidence.</text>"
    )

    # --- takeaway strip -------------------------------------------------------
    ty = 940
    o.append(f'<rect x="56" y="{ty - 26}" width="{W - 112}" height="44" rx="8" fill="#f2f6fc"/>')
    o.append(
        f'<text x="76" y="{ty + 2}" font-size="13.5" fill="{INK}">'
        f'<tspan font-weight="700">Takeaway:</tspan> temperature rescales logit '
        f'<tspan font-style="italic">separation</tspan>, not the solution. The unnormalized top-1 fans out '
        f"~10× purely because the pedestal is T-independent; normalized, every model stays a "
        f"near-deterministic argmax.</text>"
    )
    o.append("</svg>")
    return "\n".join(o)


# --------------------------------------------------------------------------- #
# Figure 2: the idea
# --------------------------------------------------------------------------- #
def box(x, y, w, h, fill, stroke, rx=10):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'


def arrow(x0, y0, x1, y1, color, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="2" '
        f'marker-end="url(#ah-{color.lstrip("#")})"{d}/>'
    )


def fig_idea():
    W, H = 1280, 760
    heads = {POS: "ah-" + POS.lstrip("#"), MUTED: "ah-" + MUTED.lstrip("#"), NEG: "ah-" + NEG.lstrip("#")}
    defs = "".join(
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for c, mid in heads.items()
    )
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
        f"<defs>{defs}</defs>",
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="56" y="52" font-size="24" font-weight="700" fill="{INK}">'
        f"Does CE temperature change the crammed solution — or just rescale its logits?</text>",
        f'<text x="56" y="78" font-size="14" fill="{INK2}">' f"The experiment behind tab:temperature_logit_stats</text>",
    ]

    # ---- Stage 1: the cramming loop -----------------------------------------
    o.append(f'<text x="56" y="126" font-size="13" font-weight="700" fill="{MUTED}">1 &#183; WHAT IS OPTIMIZED</text>')
    o.append(box(56, 142, 210, 96, "#ffffff", GRID))
    o.append(f'<text x="72" y="170" font-size="13.5" font-weight="600" fill="{INK}">PG19 passage</text>')
    o.append(f'<text x="72" y="192" font-size="12" fill="{INK2}">up to ~1.5k tokens</text>')
    o.append(f'<text x="72" y="214" font-size="12" fill="{INK2}">the text to reconstruct</text>')

    o.append(arrow(276, 190, 322, 190, MUTED))

    o.append(box(332, 142, 200, 96, "#eaf2fd", POS))
    o.append(f'<text x="348" y="170" font-size="13.5" font-weight="700" fill="{INK}">[mem] embedding</text>')
    o.append(f'<text x="348" y="192" font-size="12" fill="{INK2}">one vector, per sample</text>')
    o.append(f'<text x="348" y="214" font-size="12" font-weight="600" fill="{POS}">the ONLY trainable thing</text>')

    o.append(arrow(542, 190, 588, 190, MUTED))

    o.append(box(598, 142, 190, 96, "#ffffff", GRID))
    o.append(f'<text x="614" y="170" font-size="13.5" font-weight="600" fill="{INK}">frozen LM &#10052;</text>')
    o.append(f'<text x="614" y="192" font-size="12" fill="{INK2}">pythia-1.4b / Llama-8B</text>')
    o.append(f'<text x="614" y="214" font-size="12" fill="{INK2}">Qwen3-4B</text>')

    o.append(arrow(798, 190, 844, 190, MUTED))

    o.append(box(854, 142, 170, 96, "#ffffff", GRID))
    o.append(f'<text x="870" y="170" font-size="13.5" font-weight="600" fill="{INK}">logits z</text>')
    o.append(f'<text x="870" y="192" font-size="12" fill="{INK2}">what decoding</text>')
    o.append(f'<text x="870" y="214" font-size="12" fill="{INK2}">actually sees</text>')

    o.append(arrow(1034, 190, 1080, 190, MUTED))
    o.append(box(1090, 142, 134, 96, "#fdeeea", "#eb6834"))
    o.append(f'<text x="1106" y="176" font-size="14" font-weight="700" fill="{INK}">CE(z / T)</text>')
    o.append(f'<text x="1106" y="200" font-size="12" fill="{INK2}">the knob</text>')
    o.append(f'<text x="1106" y="220" font-size="12" fill="{INK2}">T: 0.1 → 2.0</text>')

    # gradient feedback arrow back to the embedding only
    o.append(
        f'<path d="M1157,238 L1157,272 L432,272 L432,244" fill="none" stroke="{POS}" stroke-width="2" '
        f'stroke-dasharray="5 4" marker-end="url(#{heads[POS]})"/>'
    )
    o.append(
        f'<text x="700" y="266" text-anchor="middle" font-size="12" font-weight="600" fill="{POS}">'
        f"gradient updates the embedding only — weights never move</text>"
    )

    # ---- Stage 2: the invariant ---------------------------------------------
    o.append(f'<text x="56" y="330" font-size="13" font-weight="700" fill="{MUTED}">2 &#183; WHY T IS A CLEAN KNOB</text>')
    o.append(box(56, 346, 1168, 74, "#f4faf7", "#1baf7a"))
    o.append(
        f'<text x="80" y="378" font-size="14" font-weight="700" fill="{INK}">'
        f"Stopping rule: argmax(z) = label for every token — independent of T.</text>"
    )
    o.append(
        f'<text x="80" y="402" font-size="12.5" fill="{INK2}">'
        f"So T cannot move the finish line. Any difference between runs is temperature acting on the "
        f"optimization path and the loss geometry — not on the definition of success.</text>"
    )

    # ---- Stage 3: three readouts --------------------------------------------
    o.append(
        f'<text x="56" y="472" font-size="13" font-weight="700" fill="{MUTED}">'
        f"3 &#183; WHAT WE MEASURE ON THE CONVERGED z</text>"
    )
    cards = [
        (
            56,
            "Margin z/T",
            "(z_true − runner-up) / T",
            "How separated the right token is,\nin the loss's own units.",
            "shift-invariant",
            GRID,
            "#ffffff",
        ),
        (
            446,
            "Top-1 z/T — unnormalized",
            "(max_j z_j) / T",
            "The absolute logit pedestal.\nNOT shift-invariant.",
            "inflates as T → 0",
            "#eb6834",
            "#fdeeea",
        ),
        (
            836,
            "Top-1 log p — normalized",
            "log softmax(z/T)[argmax]",
            "The winner's actual probability\nmass. Bounded by 0.",
            "the honest confidence",
            POS,
            "#eaf2fd",
        ),
    ]
    for x, title, formula, body, tag, stroke, fill in cards:
        o.append(box(x, 488, 388, 154, fill, stroke))
        o.append(f'<text x="{x + 20}" y="516" font-size="13.5" font-weight="700" fill="{INK}">{esc(title)}</text>')
        o.append(
            f'<text x="{x + 20}" y="542" font-size="13" fill="{INK}" font-family="ui-monospace, monospace">'
            f"{esc(formula)}</text>"
        )
        for j, line in enumerate(body.split("\n")):
            o.append(f'<text x="{x + 20}" y="{568 + j * 19}" font-size="12" fill="{INK2}">{esc(line)}</text>')
        o.append(
            f'<text x="{x + 20}" y="{626}" font-size="12" font-weight="700" fill="{stroke if stroke != GRID else MUTED}">'
            f"{esc(tag)}</text>"
        )

    # ---- Stage 4: the point --------------------------------------------------
    o.append('<rect x="56" y="666" width="1168" height="60" rx="8" fill="#f2f6fc"/>')
    o.append(
        f'<text x="80" y="694" font-size="13.5" fill="{INK}">'
        f'<tspan font-weight="700">The distinction that matters:</tspan> '
        f"both top-1 columns describe the same logits. Dividing by T alone leaves the pedestal in, so "
        f"cold runs look ~10× more confident.</text>"
    )
    o.append(
        f'<text x="80" y="714" font-size="13.5" fill="{INK}">'
        f"Subtracting logsumexp(z/T) removes it and reports the probability a reader would actually "
        f"quote — which barely moves with T.</text>"
    )
    o.append("</svg>")
    return "\n".join(o)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    series, budget = load()
    outputs = {
        "temperature_logit_geometry_idea.svg": fig_idea(),
        "temperature_logit_geometry_results.svg": fig_results(series, budget),
    }
    for name, svg in outputs.items():
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg + "\n")
        print(f"Wrote {os.path.relpath(path, REPO)} ({len(svg)} bytes)")
    print(f"\nbudget arm: {len(budget)} temperature(s) with both 10k and 50k runs")


if __name__ == "__main__":
    main()
