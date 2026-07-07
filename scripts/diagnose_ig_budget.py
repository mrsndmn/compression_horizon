"""Fungibility diagnostic for the IG budget-rebalancing hypothesis (GPU-free).

The hypothesis behind budget rebalancing (cap/dual) is that a plain-CE memory embedding spends a
*fixed* information-gain budget, over-investing in some tokens (margin >> eps) while others fall
short -- so reclaiming surplus bits should let MORE tokens clear eps. The cap/dual runs refuted the
outcome (they clear FEWER tokens than CE). This script tests the *premise* directly, using only the
per-stage scalars already saved by every plain-CE progressive run (no model forward needed).

For a linear progressive run each sample has one row per prefix length L=1..horizon with the saved
TOTAL information gain IG(L) = information_gain_bits (sum of per-token bits-saved over the L crammed
tokens). Two signals discriminate the competing explanations:

  * per-token IG  = IG(L) / L        -- if a fixed budget B were being spread over more tokens this
                                        FALLS ~ B/L; if each token brings its own bits it stays flat.
  * marginal IG   = IG(L) - IG(L-1)  -- the bits the L-th token adds (net of any change to tokens
                                        1..L-1). Exhaustion of a fixed budget => marginal -> 0 near
                                        the horizon. CONTENTION for a shared resource (adding a token
                                        pushes earlier tokens down) => marginal < 0.

Reads the six plain-CE runs (full / lowdim256 x eps in {0.5,1.0,2.0}); +LM (`_lm_`) and rebalancing
(`_brl_`) dirs are excluded.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from datasets import load_from_disk

BASE = "artifacts/experiments_progressive"
PREFIX = "sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_"


def ce_runs() -> list[str]:
    """Plain-CE run dirs: `..._cm_{eps}` with no `_lm_` / `_brl_` / hybrid / other suffix after."""
    out = []
    for d in sorted(glob.glob(os.path.join(BASE, PREFIX + "*_cm_*"))):
        name = os.path.basename(d)
        tail = name.split("_cm_")[1]  # e.g. "1.0" (CE) vs "1.0_lm_1.0" / "1.0_brl_cap"
        if "_" in tail:  # anything after the eps value => not plain CE
            continue
        out.append(d)
    return out


def per_sample_trajectories(d: str):
    """Yield (L_sorted, ig_sorted, horizon) per sample for consecutive (+1) linear stages."""
    ds = load_from_disk(os.path.join(d, "progressive_prefixes")).select_columns(
        ["sample_id", "stage_seq_len", "information_gain_bits", "final_convergence", "convergence_threshold"]
    )
    sid = np.array(ds["sample_id"])
    L = np.array(ds["stage_seq_len"], dtype=np.int64)
    ig = np.array(ds["information_gain_bits"], dtype=np.float64)
    conv = np.array(ds["final_convergence"], dtype=np.float64)
    thr = float(ds[0]["convergence_threshold"])
    for s in np.unique(sid):
        m = sid == s
        order = np.argsort(L[m])
        Ls, igs, cs = L[m][order], ig[m][order], conv[m][order]
        conv_mask = cs >= thr
        horizon = int(Ls[conv_mask].max()) if conv_mask.any() else 0
        yield Ls, igs, horizon


def summarize(d: str) -> dict:
    name = os.path.basename(d).replace(PREFIX, "")
    ptok_L, ptok_v = [], []      # per-token IG vs L
    marg_relpos, marg_v = [], []  # marginal IG vs relative prefix position (L/horizon)
    neg_marg = tot_marg = 0
    last_decile_marg, first_decile_marg = [], []
    for Ls, igs, horizon in per_sample_trajectories(d):
        if horizon < 10:
            continue
        # restrict to the converged frontier (L <= horizon)
        keep = Ls <= horizon
        Ls, igs = Ls[keep], igs[keep]
        ptok_L.extend(Ls.tolist())
        ptok_v.extend((igs / Ls).tolist())
        # marginal on consecutive stages
        d_ig = np.diff(igs)
        d_L = np.diff(Ls)
        cons = d_L == 1
        marg = d_ig[cons]
        relpos = (Ls[1:][cons]) / horizon
        marg_relpos.extend(relpos.tolist())
        marg_v.extend(marg.tolist())
        neg_marg += int((marg < 0).sum())
        tot_marg += int(marg.size)
        first_decile_marg.extend(marg[relpos <= 0.1].tolist())
        last_decile_marg.extend(marg[relpos >= 0.9].tolist())

    ptok_L = np.array(ptok_L)
    ptok_v = np.array(ptok_v)
    # slope of per-token IG vs L (bits/token per additional token of prefix)
    slope = np.polyfit(ptok_L, ptok_v, 1)[0] if ptok_L.size > 2 else float("nan")
    return {
        "name": name,
        "n_points": ptok_L.size,
        "ptok_early": float(np.mean(ptok_v[ptok_L <= 20])) if (ptok_L <= 20).any() else float("nan"),
        "ptok_late": float(np.mean(ptok_v[ptok_L >= ptok_L.max() * 0.9])) if ptok_L.size else float("nan"),
        "ptok_slope_per_tok": slope,
        "marg_first_decile": float(np.mean(first_decile_marg)) if first_decile_marg else float("nan"),
        "marg_last_decile": float(np.mean(last_decile_marg)) if last_decile_marg else float("nan"),
        "frac_marginal_negative": (neg_marg / tot_marg) if tot_marg else float("nan"),
        "_ptok_L": ptok_L,
        "_ptok_v": ptok_v,
        "_marg_relpos": np.array(marg_relpos),
        "_marg_v": np.array(marg_v),
    }


def main() -> None:
    runs = ce_runs()
    print(f"Plain-CE runs found: {len(runs)}")
    rows = [summarize(d) for d in runs]

    hdr = ("run", "per-tok IG early", "per-tok IG late", "slope/tok", "marg 1st-dec", "marg last-dec", "% marg<0")
    print("\n{:<28} {:>16} {:>15} {:>10} {:>12} {:>13} {:>9}".format(*hdr))
    for r in rows:
        print(
            "{:<28} {:>16.3f} {:>15.3f} {:>10.4f} {:>12.3f} {:>13.3f} {:>8.1f}%".format(
                r["name"], r["ptok_early"], r["ptok_late"], r["ptok_slope_per_tok"],
                r["marg_first_decile"], r["marg_last_decile"], 100 * r["frac_marginal_negative"],
            )
        )

    # ---- plot: per-token IG and marginal IG vs prefix position, full vs lowdim (avg over eps) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"\n(skip plot: {e})")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    groups = {"full": [r for r in rows if "lowdim" not in r["name"]],
              "lowdim256": [r for r in rows if "lowdim" in r["name"]]}
    colors = {"full": "#1f77b4", "lowdim256": "#d62728"}
    for gname, grp in groups.items():
        if not grp:
            continue
        # per-token IG vs L (binned by absolute L)
        allL = np.concatenate([r["_ptok_L"] for r in grp])
        allv = np.concatenate([r["_ptok_v"] for r in grp])
        bins = np.linspace(1, np.percentile(allL, 99), 30)
        idx = np.digitize(allL, bins)
        bx = [allL[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
        by = [allv[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
        ax1.plot(bx, by, "-o", ms=3, color=colors[gname], label=gname)
        # marginal IG vs relative position (binned)
        rp = np.concatenate([r["_marg_relpos"] for r in grp])
        mv = np.concatenate([r["_marg_v"] for r in grp])
        rb = np.linspace(0, 1, 21)
        ri = np.digitize(rp, rb)
        rx = [rp[ri == i].mean() for i in range(1, len(rb)) if (ri == i).any()]
        ry = [mv[ri == i].mean() for i in range(1, len(rb)) if (ri == i).any()]
        ax2.plot(rx, ry, "-o", ms=3, color=colors[gname], label=gname)

    ax1.set_xlabel("prefix length L (crammed tokens)")
    ax1.set_ylabel("per-token IG  = IG(L)/L  (bits)")
    ax1.set_title("Is a fixed budget spread thinner?\n(flat = each token brings its own bits)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("relative prefix position  L / horizon")
    ax2.set_ylabel("marginal IG  = IG(L) - IG(L-1)  (bits)")
    ax2.set_title("Bits the new token adds\n(->0 = budget exhausts; <0 = contention)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("IG budget fungibility diagnostic (plain-CE progressive, Pythia-1.4B)", fontweight="bold")
    fig.tight_layout()
    out = "artifacts/results/ig_budget_diagnostic.png"
    fig.savefig(out, dpi=130)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
