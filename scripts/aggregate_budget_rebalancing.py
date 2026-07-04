"""Aggregate the Pythia-1.4B information-gain budget-rebalancing deep dive.

Compares the four loss arms at a matched, FIXED convergence margin epsilon:

* ``CE``   -- plain cross-entropy at margin epsilon        (dir ``..._cm_{eps}``)
* ``+LM``  -- margin-deficit CE reweighting (raises floor)  (dir ``..._cm_{eps}_lm_{eps}``)
* ``cap``  -- Arm C: floor + reclaim of over-budget bits    (dir ``..._cm_{eps}_brl_cap``)
* ``dual`` -- Arm D: dual water-filling under a bits budget  (dir ``..._cm_{eps}_brl_dual``)

The hypothesis: reclaiming information-gain budget from over-margined tokens lets MORE tokens
clear the fixed epsilon (a longer crammed span) than plain CE / +LM. Every metric is read straight
from each run's ``progressive_prefixes`` stage-row dataset (no GPU, no model):

* ``crammed_tokens``  -- per-sample horizon = max ``stage_seq_len`` reached (the crammed span).
* ``cleared_eps``     -- per-sample max ``stage_seq_len`` whose stage fully converged
                          (``final_convergence >= convergence_threshold``) => tokens that actually
                          clear the fixed epsilon margin. This is the pre-registered headline metric.
* ``true_bits_saved`` -- per-sample ``information_gain_bits`` at the horizon (H_base - H_comp, the
                          true bits-saved built on the cached per-stage H_base; paper Eq. 9).
* ``tf_convergence``  -- per-sample final-stage teacher-forcing match ratio (reconstruction quality).
* ``greedy_match``    -- optional: mean greedy autoregressive reconstruction match from
                          ``greedy_accuracy_cache.json`` if a prior eval left one next to the run.

Rows = arms (grouped by base variant and epsilon); columns = metrics, per project table convention.

Usage:
    PYTHONPATH=./src python scripts/aggregate_budget_rebalancing.py
    PYTHONPATH=./src python scripts/aggregate_budget_rebalancing.py --csv artifacts/results/budget_rebalancing.csv
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from datasets import load_from_disk

ARTIFACT_ROOT = "artifacts/experiments_progressive"

# (label, base_suffix) — the two Pythia base variants of the deep dive.
BASES = [
    ("full", "sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5"),
    ("lowdim256", "sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.5"),
]
EPSILONS = [0.5, 1.0, 2.0]

# (arm label, dir-suffix builder from base+eps). Order defines row order within a block.
ARMS = [
    ("CE", lambda base, eps: f"{base}_cm_{eps}"),
    ("+LM", lambda base, eps: f"{base}_cm_{eps}_lm_{eps}"),
    ("cap (C)", lambda base, eps: f"{base}_cm_{eps}_brl_cap"),
    ("dual (D)", lambda base, eps: f"{base}_cm_{eps}_brl_dual"),
]


def _per_sample_finals(run_dir: str) -> dict | None:
    """Read progressive_prefixes; return per-sample horizon / cleared-eps / bits / convergence.

    Returns ``None`` if the run has not produced a stage-row dataset yet (job still pending/running).
    """
    ds_path = os.path.join(run_dir, "progressive_prefixes")
    if not os.path.isdir(ds_path) or not os.path.exists(os.path.join(ds_path, "dataset_info.json")):
        return None
    ds = load_from_disk(ds_path)
    # Only the light metadata columns are needed; drop the big embedding blobs so this stays cheap.
    drop = [
        c
        for c in ds.column_names
        if c in ("embedding", "orig_embedding", "initialization_embedding", "pca_coefficients_to_save", "text")
    ]
    if drop:
        ds = ds.remove_columns(drop)
    sid = np.asarray(ds["sample_id"])
    ssl = np.asarray(ds["stage_seq_len"], dtype=np.int64)
    conv = np.asarray(ds["final_convergence"], dtype=np.float64)
    ig = np.asarray(ds["information_gain_bits"], dtype=np.float64)
    thr = float(np.asarray(ds["convergence_threshold"], dtype=np.float64)[0])

    crammed, cleared, bits, tfconv = [], [], [], []
    for s in np.unique(sid):
        m = sid == s
        ssl_s, conv_s, ig_s = ssl[m], conv[m], ig[m]
        top = int(np.argmax(ssl_s))  # the horizon stage (largest span reached)
        crammed.append(int(ssl_s[top]))
        bits.append(float(ig_s[top]))
        tfconv.append(float(conv_s[top]))
        ok = conv_s >= (thr - 1e-6)
        cleared.append(int(ssl_s[ok].max()) if ok.any() else 0)
    return {
        "n": len(crammed),
        "crammed_tokens": np.array(crammed, dtype=np.float64),
        "cleared_eps": np.array(cleared, dtype=np.float64),
        "true_bits_saved": np.array(bits, dtype=np.float64),
        "tf_convergence": np.array(tfconv, dtype=np.float64),
        "conv_threshold": thr,
    }


def _greedy_match(run_dir: str) -> float | None:
    path = os.path.join(run_dir, "greedy_accuracy_cache.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return float(json.load(f).get("greedy_match_mean", float("nan")))


def collect() -> list[dict]:
    rows = []
    for base_label, base_suffix in BASES:
        for eps in EPSILONS:
            for arm_label, build in ARMS:
                suffix = build(base_suffix, eps)
                run_dir = os.path.join(ARTIFACT_ROOT, suffix)
                finals = _per_sample_finals(run_dir)
                row = {"base": base_label, "eps": eps, "arm": arm_label, "dir": suffix}
                if finals is None:
                    row["status"] = "pending" if os.path.isdir(run_dir) else "missing"
                    rows.append(row)
                    continue
                row["status"] = "done"
                row["n"] = finals["n"]
                row["cleared_eps"] = float(finals["cleared_eps"].mean())
                row["crammed_tokens"] = float(finals["crammed_tokens"].mean())
                row["crammed_std"] = float(finals["crammed_tokens"].std())
                row["true_bits_saved"] = float(finals["true_bits_saved"].mean())
                row["bits_per_token"] = row["true_bits_saved"] / max(row["crammed_tokens"], 1e-9)
                row["tf_convergence"] = float(finals["tf_convergence"].mean())
                row["greedy_match"] = _greedy_match(run_dir)
                rows.append(row)
    return rows


def _fmt(v, spec="{:.1f}"):
    if v is None:
        return "-"
    if isinstance(v, float) and np.isnan(v):
        return "nan"
    return spec.format(v)


def render_table(rows: list[dict]) -> str:
    cols = [
        ("arm", "arm", "{}"),
        ("n", "n", "{}"),
        ("cleared_eps", "cleared_eps", "{:.1f}"),
        ("crammed_tokens", "crammed_tok", "{:.1f}"),
        ("true_bits_saved", "true_bits", "{:.0f}"),
        ("bits_per_token", "bits/tok", "{:.2f}"),
        ("tf_convergence", "tf_conv", "{:.4f}"),
        ("greedy_match", "greedy", "{:.4f}"),
    ]
    out = []
    for base_label, _ in BASES:
        for eps in EPSILONS:
            block = [r for r in rows if r["base"] == base_label and r["eps"] == eps]
            if not block:
                continue
            out.append(f"\n=== base={base_label}  epsilon={eps} ===")
            header = "  ".join(f"{h:>12}" if k != "arm" else f"{h:<10}" for k, h, _ in cols)
            out.append(header)
            for r in block:
                if r.get("status") != "done":
                    out.append(f"{r['arm']:<10}  {r.get('status','?'):>12}")
                    continue
                cells = []
                for k, _, spec in cols:
                    if k == "arm":
                        cells.append(f"{r[k]:<10}")
                    else:
                        cells.append(f"{_fmt(r.get(k), spec):>12}")
                out.append("  ".join(cells))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="Optional path to write the flat CSV.")
    args = ap.parse_args()

    rows = collect()
    print(render_table(rows))

    done = sum(1 for r in rows if r.get("status") == "done")
    print(
        f"\n{done}/{len(rows)} runs aggregated "
        f"({sum(1 for r in rows if r.get('status')=='pending')} pending, "
        f"{sum(1 for r in rows if r.get('status')=='missing')} missing)."
    )

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        field_order = [
            "base",
            "eps",
            "arm",
            "status",
            "n",
            "cleared_eps",
            "crammed_tokens",
            "crammed_std",
            "true_bits_saved",
            "bits_per_token",
            "tf_convergence",
            "greedy_match",
            "dir",
        ]
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
