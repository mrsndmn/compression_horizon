"""Validate the hypothesis: per-token surprisal predicts steps-to-converge in progressive cramming.

The progressive-cramming trainer grows the target prefix one token at a time (``--progressive_step
1``) and, at each stage, optimizes the per-sample compression embedding until it reconstructs the
prefix (argmax match == ``--progressive_convergence_threshold``). The number of optimizer steps the
stage at prefix length ``L`` needs *on top of* the previous (already-converged) stage is the marginal
cost of absorbing the token at position ``L``:

    incr_steps(L) = steps_taken(L) - steps_taken(L-1)        # both stages converged

The hypothesis is that this marginal cost is driven by how *surprising* the newly added token is to
the frozen base model, i.e. its next-token surprisal

    s(L) = -log2 p(x_L | x_<L)   bits            # the per-token quantity in plot_surprisal_curve.py

This script reads a finished progressive run's ``progressive_prefixes`` dataset (for the steps), runs
one frozen forward pass per sample to get the surprisal curve (identical tokenization to the trainer),
aligns them position-by-position, and reports the correlation plus a figure.

Why the *marginal* (incremental) test and not the cumulative one: ``steps_taken`` and the cumulative
description length ``DL(n) = sum_{i<=n} s_i`` (== ``information_gain_bits`` at convergence) are both
monotonically increasing in ``L``, so their correlation is ~1.0 trivially. That is a spurious
length confound. Differencing both removes it and tests the actual claim.

Example:
    python scripts/analyze_surprisal_vs_steps.py \\
        --run_dir artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1
"""

import argparse
import math
import os
import shlex

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_dataset, load_from_disk  # noqa: E402
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

_TRAIN_SPLIT_DATASETS = {"LarryLovestein/pg19_1k", "LarryLovestein/fanfics_1k"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="Progressive run dir (with progressive_prefixes/ and cmd.txt).")
    p.add_argument("--model_checkpoint", default=None, help="Override base LM (default: parsed from cmd.txt).")
    p.add_argument("--dataset_name", default=None, help="Override dataset (default: parsed from cmd.txt).")
    p.add_argument("--max_length", type=int, default=None, help="Override max_sequence_length (default: from cmd.txt).")
    p.add_argument("--limit", type=int, default=None, help="Override number of samples (default: from cmd.txt).")
    p.add_argument("--text_column", default="text")
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None, help="Figure path (default: <run_dir>/surprisal_vs_steps.png).")
    return p.parse_args()


def parse_cmd_txt(run_dir: str) -> dict:
    """Recover the run's key knobs from the persisted CLI (cmd.txt)."""
    path = os.path.join(run_dir, "cmd.txt")
    cfg: dict = {}
    if not os.path.exists(path):
        return cfg
    with open(path, encoding="utf-8") as f:
        toks = shlex.split(f.read().strip())
    flag_keys = {
        "--model_checkpoint": "model_checkpoint",
        "--dataset_name": "dataset_name",
        "--max_sequence_length": "max_sequence_length",
        "--limit_dataset_items": "limit",
        "--progressive_step": "progressive_step",
        "--progressive_min_seq_len": "progressive_min_seq_len",
        "--no_bos_token": "no_bos_token",
    }
    i = 0
    while i < len(toks):
        if toks[i] in flag_keys:
            key = flag_keys[toks[i]]
            cfg[key] = toks[i + 1] if i + 1 < len(toks) else True
            i += 2
        else:
            i += 1
    return cfg


@torch.no_grad()
def per_token_surprisal_bits(model, input_ids: torch.Tensor) -> np.ndarray:
    """s_i = -log2 p(x_i | x_<i) in bits; entry j predicts token at 0-index j+1."""
    logits = model(input_ids=input_ids).logits
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = input_ids[0, 1:]
    nll = -logp[torch.arange(targets.shape[0], device=targets.device), targets]
    return (nll / math.log(2)).cpu().numpy()


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman partial correlation of x,y controlling for z (rank-residualized)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    zc = np.c_[np.ones_like(rz), rz]

    def resid(a):
        beta = np.linalg.lstsq(zc, a, rcond=None)[0]
        return a - zc @ beta

    return float(pearsonr(resid(rx), resid(ry)).statistic)


def collect_pairs(run_dir: str, surp: dict, progressive_step: int):
    """Pair each converged token's marginal step-cost with the surprisal of the token it added.

    ``steps_taken`` is stored two ways across runs: cumulative-across-stages (recent runs,
    monotonically non-decreasing within a sample) or per-stage (older runs). We auto-detect by the
    fraction of consecutive converged pairs that decrease: a cumulative sequence never decreases.
    The marginal cost of the token growing the prefix to length ``L`` is ``steps[L]-steps[L-1]``
    (cumulative) or ``steps[L]`` (per-stage); it is paired with that token's surprisal entry
    ``L-2``. Returns arrays (surprisal, cost, position_L, sample_id, ig_increment, is_cumulative).
    """
    ds = load_from_disk(os.path.join(run_dir, "progressive_prefixes"))
    drop = [
        c
        for c in ("embedding", "orig_embedding", "initialization_embedding", "pca_coefficients_to_save")
        if c in ds.column_names
    ]
    ds = ds.remove_columns(drop)
    from collections import defaultdict

    rows = defaultdict(list)
    for r in ds:
        rows[r["sample_id"]].append((r["stage_seq_len"], r["steps_taken"], r["final_convergence"], r["information_gain_bits"]))
    for rr in rows.values():
        rr.sort()

    dec = comp = 0
    for rr in rows.values():
        for k in range(1, len(rr)):
            (L, st, c, _), (Lp, stp, cp, _) = rr[k], rr[k - 1]
            if L - Lp == progressive_step and c == 1.0 and cp == 1.0:
                comp += 1
                dec += int(st < stp)
    cumulative = comp > 0 and (dec / comp) < 0.02

    S, STEPS, POS, SID, IGINC = [], [], [], [], []
    for sid, rr in rows.items():
        if sid not in surp:
            continue
        sarr = surp[sid]
        for k in range(1, len(rr)):
            (L, st, c, ig), (Lp, stp, cp, igp) = rr[k], rr[k - 1]
            if L - Lp != progressive_step or c != 1.0:
                continue
            idx = L - 2  # token added at 0-index L-1 is predicted by surprisal entry L-2
            if not (0 <= idx < len(sarr)):
                continue
            if cumulative:
                if cp != 1.0:
                    continue
                cost = st - stp
            else:
                cost = st
            if cost < 1:
                continue
            S.append(float(sarr[idx]))
            STEPS.append(float(cost))
            POS.append(float(L))
            SID.append(int(sid))
            IGINC.append(float(ig - igp))
    arrays = tuple(np.array(a) for a in (S, STEPS, POS, SID, IGINC))
    return (*arrays, cumulative)


def main() -> None:
    args = parse_args()
    cfg = parse_cmd_txt(args.run_dir)
    model_ckpt = args.model_checkpoint or cfg.get("model_checkpoint")
    dataset_name = args.dataset_name or cfg.get("dataset_name")
    max_length = args.max_length or int(cfg.get("max_sequence_length", 4096))
    limit = args.limit or (int(cfg["limit"]) if "limit" in cfg else None)
    progressive_step = int(cfg.get("progressive_step", 1))
    no_bos = bool(cfg.get("no_bos_token", False))
    if model_ckpt is None or dataset_name is None:
        raise SystemExit("Could not determine model/dataset; pass --model_checkpoint and --dataset_name.")
    if progressive_step != 1:
        print(
            f"[warn] progressive_step={progressive_step}: each stage adds a block of tokens; "
            "the surprisal used is that of the *last* token in the block, which is an approximation."
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = "train" if dataset_name in _TRAIN_SPLIT_DATASETS else "test"

    print(f"Loading {dataset_name} (split={split}) and {model_ckpt} ...")
    raw = load_dataset(dataset_name, split=split)
    n = limit or len(raw)
    raw = raw.select(range(min(n, len(raw))))
    tok = AutoTokenizer.from_pretrained(model_ckpt)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if no_bos and hasattr(tok, "add_bos_token"):
        tok.add_bos_token = False
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, dtype=torch.bfloat16).to(device).eval()

    surp: dict[int, np.ndarray] = {}
    for i in range(len(raw)):
        enc = tok(raw[i][args.text_column], truncation=True, max_length=max_length, add_special_tokens=not no_bos)
        ids = torch.tensor(enc["input_ids"]).unsqueeze(0).to(device)
        surp[i] = per_token_surprisal_bits(model, ids)

    S, STEPS, POS, SID, IGINC, cumulative = collect_pairs(args.run_dir, surp, progressive_step)
    if len(S) == 0:
        raise SystemExit("No aligned (surprisal, steps) pairs found.")
    print(f"steps_taken convention detected: {'cumulative' if cumulative else 'per-stage'}")

    pear = pearsonr(S, STEPS)
    spear = spearmanr(S, STEPS)
    pear_log = pearsonr(S, np.log1p(STEPS))
    partial = partial_spearman(S, STEPS, POS)
    per_sample = []
    for sid in np.unique(SID):
        m = SID == sid
        if m.sum() >= 10:
            rho = spearmanr(S[m], STEPS[m]).statistic
            if not np.isnan(rho):
                per_sample.append(rho)
    per_sample = np.array(per_sample)

    print(f"\n=== Surprisal vs steps-to-converge | {os.path.basename(args.run_dir.rstrip('/'))} ===")
    print(f"token-level pairs (converged, aligned): {len(S)}   samples: {len(np.unique(SID))}")
    print(f"surprisal bits : mean={S.mean():.2f} sd={S.std():.2f} max={S.max():.1f}")
    print(f"incr steps     : mean={STEPS.mean():.1f} median={np.median(STEPS):.0f} max={STEPS.max():.0f}")
    print(f"\nPOOLED   Pearson r            = {pear.statistic:.3f} (p={pear.pvalue:.1e})")
    print(f"POOLED   Pearson r (log steps)= {pear_log.statistic:.3f}")
    print(f"POOLED   Spearman rho         = {spear.statistic:.3f} (p={spear.pvalue:.1e})")
    print(
        f"PARTIAL  Spearman | position  = {partial:.3f}   (position-vs-surprisal rho="
        f"{spearmanr(POS, S).statistic:.3f}, position-vs-steps rho={spearmanr(POS, STEPS).statistic:.3f})"
    )
    print(
        f"PER-SAMPLE Spearman rho       : mean={per_sample.mean():.3f} median={np.median(per_sample):.3f} "
        f"sd={per_sample.std():.3f} frac>0={np.mean(per_sample > 0):.2f} (n={len(per_sample)})"
    )
    print(
        f"CHECK    surprisal vs IG-incr = {pearsonr(S, IGINC).statistic:.3f} "
        f"(IGinc mean={IGINC.mean():.2f} ~ surprisal mean {S.mean():.2f})"
    )

    # ---- figure ----
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    ax0.scatter(S, STEPS, s=4, alpha=0.12, color="tab:blue", rasterized=True)
    # binned means of steps within surprisal bins
    bins = np.linspace(0, np.percentile(S, 99.5), 25)
    idx = np.digitize(S, bins)
    bx, by = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 20:
            bx.append(0.5 * (bins[b - 1] + bins[b]))
            by.append(np.median(STEPS[m]))
    ax0.plot(bx, by, "o-", color="tab:red", lw=2, label="median steps per surprisal bin")
    ax0.set_yscale("log")
    ax0.set_xlabel("per-token surprisal s(L) = -log2 p(x_L | x_<L)  [bits]")
    ax0.set_ylabel("incremental steps to converge token L  (log)")
    ax0.set_title(f"Spearman rho = {spear.statistic:.3f}  (partial | L = {partial:.3f})")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(alpha=0.3)

    ax1.hist(per_sample, bins=20, color="tab:green", alpha=0.8)
    ax1.axvline(per_sample.mean(), ls="--", color="black", label=f"mean {per_sample.mean():.2f}")
    ax1.axvline(0, ls=":", color="gray")
    ax1.set_xlabel("per-sample Spearman rho (surprisal vs steps)")
    ax1.set_ylabel("# samples")
    ax1.set_title(f"per-sample correlation ({np.mean(per_sample > 0) * 100:.0f}% positive)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    fig.suptitle(f"Per-token surprisal predicts steps-to-converge — {os.path.basename(args.run_dir.rstrip('/'))}", fontsize=12)
    output = args.output or os.path.join(args.run_dir, "surprisal_vs_steps.png")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150)
    print(f"\nSaved figure to {output}")


if __name__ == "__main__":
    main()
