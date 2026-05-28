"""Generate ``tab:surprisal_steps_correlation``: does base-model surprisal predict cramming effort?

Progressive cramming with ``--progressive_step 1`` grows the target prefix one token at a time and,
at each stage, optimizes the per-sample compression embedding (warm-started from the previous,
already-converged stage) until it reconstructs the prefix. The optimizer steps the stage at prefix
length ``L`` spends *on top of* the previous stage are the marginal cost of absorbing the token at
position ``L``::

    incr_steps(L) = steps_taken(L) - steps_taken(L-1)        # both stages converged

This table tests whether that marginal cost is governed by how *surprising* the newly added token is
to the frozen base model, i.e. its next-token surprisal ``s(L) = -log2 p(x_L | x_<L)`` (bits) -- the
per-token quantity plotted by ``scripts/plot_surprisal_curve.py``. For every checkpoint we read the
saved ``progressive_prefixes`` trajectory (for the steps), run one frozen forward pass per sample
(for the surprisal, using the trainer's tokenization), align them position-by-position over the
converged stages, and report the rank correlation plus reliability metrics.

Why the *marginal* test and not the cumulative one: ``steps_taken`` and the cumulative description
length ``DL(n) = sum_{i<=n} s_i`` (== ``information_gain_bits`` at convergence) both increase
monotonically with ``L``, so their correlation is ~1.0 trivially. Differencing both removes that
length confound and tests the actual claim.

Heavy model forwards are gated behind ``--compute`` and cached to ``<run>/surprisal_steps_cache.json``
inside each checkpoint dir (like ``attn_hijacking.py``). Without ``--compute`` the table is rendered
from those caches, so ``make tables`` regenerates the ``.tex`` without a GPU.

    # one-time (needs a GPU + the four base models):
    PYTHONPATH=./src:. python scripts/paper/tables/surprisal_steps_correlation.py --compute --save
    # cheap re-render from cache (what tables.sh runs):
    PYTHONPATH=./src:. python scripts/paper/tables/surprisal_steps_correlation.py --save
"""

import argparse
import json
import math
import os
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from tabulate import tabulate

from compression_horizon.utils import hlines_to_booktabs

_EXP = "artifacts/experiments_progressive"

# (checkpoint dir, display name). Four base-model families at their main PG19 learning rate.
CHECKPOINTS: List[Tuple[str, str]] = [
    (f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1", "Llama-3.1-8B {\\small lr=0.1}"),
    (f"{_EXP}/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5", "Pythia-1.4B {\\small lr=0.5}"),
    (f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1", "SmolLM2-1.7B {\\small lr=0.1}"),
    (f"{_EXP}/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lr_0.1", "Gemma-3-4B {\\small lr=0.1}"),
]

CACHE_NAME = "surprisal_steps_cache.json"
_TRAIN_SPLIT_DATASETS = {"LarryLovestein/pg19_1k", "LarryLovestein/fanfics_1k"}
_BOOTSTRAP = 2000
_SEED = 0


# ----------------------------------------------------------------------------
# Run-config recovery + alignment (shared with scripts/analyze_surprisal_vs_steps.py).
# ----------------------------------------------------------------------------
def parse_cmd_txt(run_dir: str) -> dict:
    """Recover the run's tokenization-relevant knobs from the persisted CLI (cmd.txt)."""
    path = os.path.join(os.path.dirname(run_dir.rstrip("/")), "cmd.txt")
    cfg: dict = {}
    if not os.path.exists(path):
        return cfg
    with open(path, encoding="utf-8") as f:
        toks = shlex.split(f.read().strip())
    keys = {
        "--model_checkpoint": "model_checkpoint",
        "--dataset_name": "dataset_name",
        "--max_sequence_length": "max_sequence_length",
        "--limit_dataset_items": "limit",
        "--progressive_step": "progressive_step",
        "--no_bos_token": "no_bos_token",
    }
    i = 0
    while i < len(toks):
        if toks[i] in keys:
            cfg[keys[toks[i]]] = toks[i + 1] if i + 1 < len(toks) else True
            i += 2
        else:
            i += 1
    return cfg


def _detect_cumulative(rows: Dict[int, list], progressive_step: int) -> bool:
    """Decide whether ``steps_taken`` is cumulative-across-stages or per-stage.

    The trainer's ``steps_taken`` field has had two conventions across runs: recent runs store the
    running cumulative optimizer-step count (monotonically non-decreasing within a sample), while
    older runs store the per-stage count (the steps that one stage alone spent). A cumulative
    sequence never decreases between consecutive converged stages; a per-stage one decreases roughly
    half the time. We classify by the fraction of consecutive converged pairs that decrease.
    """
    dec = comp = 0
    for rr in rows.values():
        for k in range(1, len(rr)):
            (L, st, c), (Lp, stp, cp) = rr[k], rr[k - 1]
            if L - Lp != progressive_step or c != 1.0 or cp != 1.0:
                continue
            comp += 1
            dec += int(st < stp)
    return comp > 0 and (dec / comp) < 0.02


def collect_pairs(run_dir: str, surp: Dict[int, np.ndarray], progressive_step: int):
    """Pair each converged token's marginal step-cost with the surprisal of the token it added.

    The marginal cost of absorbing the token that grows the prefix to length ``L`` is ``steps[L] -
    steps[L-1]`` when ``steps_taken`` is cumulative, but ``steps[L]`` directly when it is already
    per-stage (see :func:`_detect_cumulative`). Either way it is paired with the surprisal of that
    same token, ``surprisal[L-2]`` (the token at 0-index ``L-1`` is predicted by entry ``L-2``).
    Returns ``(surprisal, cost, position, sample_id, is_cumulative)``.
    """
    from datasets import load_from_disk

    ds = load_from_disk(run_dir)
    drop = [
        c
        for c in ("embedding", "orig_embedding", "initialization_embedding", "pca_coefficients_to_save")
        if c in ds.column_names
    ]
    ds = ds.remove_columns(drop)
    rows = defaultdict(list)
    for r in ds:
        rows[r["sample_id"]].append((r["stage_seq_len"], r["steps_taken"], r["final_convergence"]))
    for rr in rows.values():
        rr.sort()
    cumulative = _detect_cumulative(rows, progressive_step)

    S, STEPS, POS, SID = [], [], [], []
    for sid, rr in rows.items():
        if sid not in surp:
            continue
        sarr = surp[sid]
        for k in range(1, len(rr)):
            (L, st, c), (Lp, stp, cp) = rr[k], rr[k - 1]
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
            if cost < 1:  # a converged stage always spends >=1 step; drop rare bad rows
                continue
            S.append(float(sarr[idx]))
            STEPS.append(float(cost))
            POS.append(float(L))
            SID.append(int(sid))
    arrays = tuple(np.array(a) for a in (S, STEPS, POS, SID))
    return (*arrays, cumulative)


# ----------------------------------------------------------------------------
# Statistics.
# ----------------------------------------------------------------------------
def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman partial correlation of x,y controlling for z (rank-residualized)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    zc = np.c_[np.ones_like(rz), rz]

    def resid(a):
        beta = np.linalg.lstsq(zc, a, rcond=None)[0]
        return a - zc @ beta

    return float(pearsonr(resid(rx), resid(ry)).statistic)


def _cluster_bootstrap_ci(S: np.ndarray, STEPS: np.ndarray, SID: np.ndarray) -> Tuple[float, float]:
    """95% CI for the pooled Spearman rho, resampling whole samples (clusters) with replacement.

    Tokens within a sample are correlated, so a naive per-pair bootstrap understates the CI; we
    resample the ~50 samples instead, which is the honest unit of independence here.
    """
    rng = np.random.default_rng(_SEED)
    by_sid = {sid: np.where(SID == sid)[0] for sid in np.unique(SID)}
    sids = list(by_sid)
    rhos = []
    for _ in range(_BOOTSTRAP):
        pick = rng.choice(sids, size=len(sids), replace=True)
        idx = np.concatenate([by_sid[s] for s in pick])
        rho = spearmanr(S[idx], STEPS[idx]).statistic
        if not np.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan"), float("nan")
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def compute_stats(S: np.ndarray, STEPS: np.ndarray, POS: np.ndarray, SID: np.ndarray) -> dict:
    """All correlation + reliability metrics for one checkpoint."""
    per_sample = []
    for sid in np.unique(SID):
        m = SID == sid
        if m.sum() >= 10:
            rho = spearmanr(S[m], STEPS[m]).statistic
            if not np.isnan(rho):
                per_sample.append(float(rho))
    per_sample = np.array(per_sample)
    ci_lo, ci_hi = _cluster_bootstrap_ci(S, STEPS, SID)
    return {
        "n_pairs": int(len(S)),
        "n_samples": int(len(np.unique(SID))),
        "mean_surprisal": float(S.mean()),
        "median_steps": float(np.median(STEPS)),
        "spearman": float(spearmanr(S, STEPS).statistic),
        "spearman_ci": [ci_lo, ci_hi],
        "pearson_log_steps": float(pearsonr(S, np.log1p(STEPS)).statistic),
        "partial_spearman_pos": _partial_spearman(S, STEPS, POS),
        "pos_vs_surprisal": float(spearmanr(POS, S).statistic),
        "per_sample_mean": float(per_sample.mean()) if len(per_sample) else float("nan"),
        "per_sample_sd": float(per_sample.std()) if len(per_sample) else float("nan"),
        "frac_positive": float(np.mean(per_sample > 0)) if len(per_sample) else float("nan"),
    }


# ----------------------------------------------------------------------------
# Compute (model forwards) + cache.
# ----------------------------------------------------------------------------
def compute_and_cache(run_dir: str, device: Optional[str] = None, force: bool = False) -> dict:
    """Run the base model over each sample, align with steps, cache stats to the checkpoint dir."""
    cache_path = os.path.join(run_dir, CACHE_NAME)
    if os.path.exists(cache_path) and not force:
        print(f"  cache exists, skipping compute: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = parse_cmd_txt(run_dir)
    model_ckpt = cfg.get("model_checkpoint")
    dataset_name = cfg.get("dataset_name")
    max_length = int(cfg.get("max_sequence_length", 4096))
    limit = int(cfg["limit"]) if "limit" in cfg else None
    progressive_step = int(cfg.get("progressive_step", 1))
    no_bos = bool(cfg.get("no_bos_token", False))
    if model_ckpt is None or dataset_name is None:
        raise SystemExit(f"{run_dir}: could not parse model/dataset from cmd.txt")
    print(f"  model={model_ckpt} dataset={dataset_name} max_len={max_length} limit={limit} step={progressive_step}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = "train" if dataset_name in _TRAIN_SPLIT_DATASETS else "test"
    raw = load_dataset(dataset_name, split=split)
    raw = raw.select(range(min(limit or len(raw), len(raw))))
    tok = AutoTokenizer.from_pretrained(model_ckpt)
    tok.pad_token = tok.eos_token
    if no_bos and hasattr(tok, "add_bos_token"):
        tok.add_bos_token = False
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, dtype=torch.bfloat16).to(dev).eval()

    @torch.no_grad()
    def surprisal_bits(ids):
        logits = model(input_ids=ids).logits
        logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = ids[0, 1:]
        nll = -logp[torch.arange(tgt.shape[0], device=tgt.device), tgt]
        return (nll / math.log(2)).cpu().numpy()

    surp: Dict[int, np.ndarray] = {}
    for i in range(len(raw)):
        enc = tok(raw[i]["text"], truncation=True, max_length=max_length, add_special_tokens=not no_bos)
        ids = torch.tensor(enc["input_ids"]).unsqueeze(0).to(dev)
        surp[i] = surprisal_bits(ids)

    model = None  # drop the reference so the next checkpoint's model can reclaim GPU memory
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    S, STEPS, POS, SID, cumulative = collect_pairs(run_dir, surp, progressive_step)
    if len(S) == 0:
        raise SystemExit(f"{run_dir}: no aligned (surprisal, steps) pairs")
    stats = compute_stats(S, STEPS, POS, SID)
    stats["model_checkpoint"] = model_ckpt
    stats["dataset_name"] = dataset_name
    stats["steps_convention"] = "cumulative" if cumulative else "per_stage"
    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(
        f"  saved cache: {cache_path}  (steps={stats['steps_convention']}, "
        f"spearman={stats['spearman']:.3f}, N={stats['n_pairs']})"
    )
    return stats


def load_cache(run_dir: str) -> Optional[dict]:
    cache_path = os.path.join(run_dir, CACHE_NAME)
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Rendering.
# ----------------------------------------------------------------------------
def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.2f}, {hi:.2f}]"


def format_table(names: List[str], stats_list: List[dict], tablefmt: str = "latex") -> str:
    """Render the correlation + reliability table for the given checkpoints."""
    rows = []
    for name, st in zip(names, stats_list):
        rho = st["spearman"]
        lo, hi = st["spearman_ci"]
        per_sample = (
            f"{st['per_sample_mean']:.2f} {{\\small $\\pm$ {st['per_sample_sd']:.2f}}}"
            if tablefmt == "latex"
            else f"{st['per_sample_mean']:.2f} ± {st['per_sample_sd']:.2f}"
        )
        rows.append(
            [
                name,
                f"{st['n_pairs']:,}".replace(",", "{,}") if tablefmt == "latex" else f"{st['n_pairs']:,}",
                f"{rho:.2f} {{\\small {_fmt_ci(lo, hi)}}}" if tablefmt == "latex" else f"{rho:.2f} {_fmt_ci(lo, hi)}",
                f"{st['partial_spearman_pos']:.2f}",
                per_sample,
                f"{st['frac_positive'] * 100:.0f}\\%" if tablefmt == "latex" else f"{st['frac_positive'] * 100:.0f}%",
            ]
        )
    latex = tablefmt == "latex"
    headers = [
        "Model",
        "$N$",
        "Spearman $\\rho$ [95\\% CI]" if latex else "Spearman rho [95% CI]",
        "Partial $\\rho \\mid n$" if latex else "Partial rho | n",
        "Per-sample $\\bar\\rho$" if latex else "Per-sample mean rho",
        "\\% $\\rho{>}0$" if latex else "% rho>0",
    ]
    # ``latex_raw`` passes cell/header strings through verbatim (no metachar escaping), so our
    # math and \small macros render as written; force all-left columns to match the house style.
    fmt = "latex_raw" if latex else (tablefmt or "github")
    result = tabulate(rows, headers=headers, tablefmt=fmt, colalign=["left"] * len(headers))
    if latex:
        result = hlines_to_booktabs(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--compute", action="store_true", help="Run base-model forwards and (re)write the per-checkpoint cache."
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if a cache exists (requires --compute).")
    parser.add_argument("--device", default=None, help="torch device for --compute (default: cuda if available).")
    parser.add_argument("--tablefmt", default="latex", help="tabulate format for stdout (default: latex).")
    parser.add_argument("--save", action="store_true", help="Write the rendered LaTeX table to <save-dir>/<save-name>.tex.")
    parser.add_argument("--save-dir", default="paper/tables")
    parser.add_argument("--save-name", default="surprisal_steps_correlation")
    args = parser.parse_args()
    if args.force and not args.compute:
        parser.error("--force requires --compute")

    names, stats_list = [], []
    for run, display in CHECKPOINTS:
        prefixes = os.path.join(run, "progressive_prefixes")
        if not os.path.isdir(prefixes):
            raise FileNotFoundError(f"missing progressive_prefixes: {prefixes}")
        print(f"== {display} ==")
        if args.compute:
            stats = compute_and_cache(prefixes, device=args.device, force=args.force)
        else:
            stats = load_cache(prefixes)
            if stats is None:
                raise SystemExit(f"No cache at {os.path.join(prefixes, CACHE_NAME)}. Run once with --compute (needs a GPU).")
        names.append(display)
        stats_list.append(stats)

    print("\n" + format_table(names, stats_list, tablefmt=args.tablefmt if args.tablefmt != "latex" else "github"))

    if args.save:
        tex = format_table(names, stats_list, tablefmt="latex")
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.save_name}.tex"
        out_path.write_text(tex + ("\n" if not tex.endswith("\n") else ""), encoding="utf-8")
        print(f"\nSaved 'tab:{args.save_name}' to {out_path}")


if __name__ == "__main__":
    main()
