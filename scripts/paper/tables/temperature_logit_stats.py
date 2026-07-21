"""Reconstruction logit-geometry table for the CE-temperature ablation (tab:temperature_logit_stats).

For each temperature run we report, over the final stage's reconstructed tokens:
  * Avg Steps   -- mean total optimization steps per sample (cumulative steps_taken -> final value;
                   capped at max_optimization_steps_per_sample). Not a logit quantity, shown as-is.
  * Margin z/T  -- mean (logit[true] - runner_up) / T  (the margin in the loss's own units).
  * Top-1 z/T   -- mean (max_j logit_j) / T   (UNNORMALIZED: the raw winning logit in z/T units).
  * Top-1 logp  -- mean log_softmax(z/T)[argmax]  (NORMALIZED: the winning token's log-probability
                   under the temperature softmax the loss uses; equals Top-1 z/T minus logsumexp(z/T),
                   so the two columns differ exactly by the mean log-partition).

The reconstruction logits z are the raw model outputs (what decoding sees); dividing by T expresses
them in the units the CE(z/T) loss actually optimizes, factoring out the trivial z ~ T rescaling and
leaving the residual "confidence" the optimizer reaches at argmax-convergence. The UNNORMALIZED Top-1
z/T tracks the absolute logit pedestal (nearly T-independent, so /T inflates it as T->0); the
NORMALIZED Top-1 logp is bounded above by 0 and reads as the actual probability mass on the winner.

Two-step, mirroring the other cache-backed tables (e.g. surprisal_steps_correlation):
    # 1) recompute the cache (needs a GPU + the two base models):
    HF_HOME=... PYTHONPATH=./src:. python scripts/paper/tables/temperature_logit_stats.py --compute
    # 2) render the .tex from the committed cache (no GPU):
    PYTHONPATH=./src:. python scripts/paper/tables/temperature_logit_stats.py --save
"""

import argparse
import json
import os

import numpy as np

EXP = "artifacts/experiments_progressive"
CACHE = "artifacts/paper/temperature_logit_stats.json"
OUT_TEX = "paper/tables/temperature_logit_stats.tex"
TEMPS = ["0.1", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0"]
MODELS = [
    # (dir_short, lr, checkpoint, row_prefix)
    ("pythia-1.4b", "0.5", "EleutherAI/pythia-1.4b", "P1.4b"),
    ("Meta-Llama-3.1-8B", "0.1", "unsloth/Meta-Llama-3.1-8B", "L8b"),
]
N_SAMPLES_STAMP = 50


def run_rows():
    """Ordered (label, arm, T, dir_short, lr, checkpoint, prefix, dir) mirroring the main table."""
    rows = []
    for dir_short, lr, ckpt, prefix in MODELS:
        for T in TEMPS:
            arms = [("", "control")] if T == "1.0" else [("", "raw"), ("_comp_t2", "t2")]
            for suf, arm in arms:
                d = f"{EXP}/sl_4096_{dir_short}_ds_pg19_1k_limit_50_lr_{lr}_temp_{T}{suf}/progressive_prefixes"
                rows.append((f"{prefix} T={T} {arm}", arm, T, dir_short, lr, ckpt, prefix, d))
    return rows


# --------------------------------------------------------------------------- #
# Compute (GPU): raw margin + raw top-1 (final stage) and total steps per run.
# --------------------------------------------------------------------------- #
def compute():
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compression_horizon.paper.tables.progressive import flatten_embedding

    @torch.no_grad()
    def sample_stats(model, tok, row, device, temperature):
        text = row.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            return None
        num_comp = int(row.get("num_compression_tokens", 1))
        emb = torch.tensor(flatten_embedding(row), dtype=torch.bfloat16, device=device).reshape(
            1, num_comp, model.config.hidden_size
        )
        enc = tok(text, truncation=True, padding=False, return_tensors="pt")
        input_ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        united = torch.cat([emb, model.get_input_embeddings()(input_ids)], dim=1)
        umask = torch.cat([torch.ones((1, num_comp), device=device, dtype=attn.dtype), attn], dim=1)
        logits = model(inputs_embeds=united, attention_mask=umask).logits
        pred = logits[:, num_comp:, :][:, :-1, :]  # predict input_ids[:,1:]
        labels, mask = input_ids[:, 1:], attn[:, 1:].bool()
        if mask.sum() == 0:
            return None
        top2 = pred.topk(2, dim=-1).values
        true_logit = pred.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        is_top1 = pred.argmax(-1) == labels
        runner_up = torch.where(is_top1, top2[..., 1], top2[..., 0])
        margin = (true_logit - runner_up)[mask].float().mean().item()
        top1 = top2[..., 0][mask].float().mean().item()
        # NORMALIZED top-1: log-prob of the winner under the temperature softmax the loss uses,
        # log_softmax(z/T)[argmax] = (max_j z_j)/T - logsumexp(z/T). Needs the full vocab logits.
        logp = torch.log_softmax(pred.float() / temperature, dim=-1)
        top1_logp = logp.max(dim=-1).values[mask].mean().item()
        return margin, top1, top1_logp

    device = torch.device("cuda")
    cache = {}
    by_ckpt = {}
    for row in run_rows():
        by_ckpt.setdefault(row[5], []).append(row)
    for ckpt, rows in by_ckpt.items():
        rows = [r for r in rows if os.path.isdir(r[-1])]
        if not rows:
            continue
        print(f"\n=== {ckpt} ({len(rows)} runs) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(ckpt)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device).eval()
        for label, arm, T, dir_short, lr, _, prefix, d in rows:
            ds = load_from_disk(d)
            # total steps per sample = final (max) cumulative steps_taken; + global index of final stage.
            light = ds.select_columns(["sample_id", "stage_index", "stage_seq_len", "steps_taken"])
            steps_max, final_idx = {}, {}
            for i, r in enumerate(light):
                s = int(r["sample_id"])
                steps_max[s] = max(steps_max.get(s, 0), int(r["steps_taken"] or 0))
                key = (int(r.get("stage_index", 0) or 0), int(r.get("stage_seq_len", 0) or 0))
                if s not in final_idx or key > final_idx[s][1]:
                    final_idx[s] = (i, key)
            margins, top1s, top1_logps = [], [], []
            for s, (gi, _) in final_idx.items():
                st = sample_stats(model, tok, ds[gi], device, float(T))
                if st is not None:
                    margins.append(st[0])
                    top1s.append(st[1])
                    top1_logps.append(st[2])
            cache[label] = {
                "model": prefix,
                "T": float(T),
                "arm": arm,
                "n": len(margins),
                "avg_steps": float(np.mean(list(steps_max.values()))),
                "margin_raw": float(np.mean(margins)),
                "top1_raw": float(np.mean(top1s)),
                "top1_logp_norm": float(np.mean(top1_logps)),
            }
            print(
                f"  {label:22s} steps={cache[label]['avg_steps']:8.1f} "
                f"margin_raw={cache[label]['margin_raw']:6.3f} top1_raw={cache[label]['top1_raw']:7.3f} "
                f"top1_logp_norm={cache[label]['top1_logp_norm']:7.3f}",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    print(f"\nWrote {CACHE} ({len(cache)} runs)")


# --------------------------------------------------------------------------- #
# Render (no GPU): read cache, emit z/T columns as a LaTeX tabular.
# --------------------------------------------------------------------------- #
def _arm_tex(arm):
    return {"raw": "raw", "t2": "$t^2$", "control": "control"}.get(arm, arm)


def render():
    with open(CACHE) as f:
        cache = json.load(f)
    rows = run_rows()
    n_min = min(int(cache[lbl]["n"]) for lbl, *_ in rows if lbl in cache)

    lines = [
        f"% paper-lint: n_samples={n_min}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        " Configuration & & Avg.\\ Steps & Margin $\\mathbf{z}/T$ & " "Top-1 $\\mathbf{z}/T$ & Top-1 $\\log p$ \\\\",
        " & & & & (unnorm.) & (norm.) \\\\",
        "\\midrule",
    ]
    prev_prefix, prev_T = None, None
    for label, arm, T, dir_short, lr, ckpt, prefix, d in rows:
        if label not in cache:
            continue
        e = cache[label]
        Tf = float(T)
        if prev_prefix is not None and (prefix != prev_prefix):
            lines.append("\\midrule")
        elif prev_T is not None and T != prev_T:
            lines.append("\\addlinespace[2pt]")
        prev_prefix, prev_T = prefix, T
        margin_zt = e["margin_raw"] / Tf
        top1_zt = e["top1_raw"] / Tf
        top1_logp = e["top1_logp_norm"]
        lines.append(
            f" {prefix} $T{{=}}{T}$ & {_arm_tex(arm)} & {e['avg_steps']:.0f} & "
            f"{margin_zt:.2f} & {top1_zt:.2f} & {top1_logp:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines))
    n_rows = sum(1 for ln in lines if ln.strip().endswith("\\\\") and "$T" in ln)
    print(f"Wrote {OUT_TEX} ({n_rows} data rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compute", action="store_true", help="Recompute the cache (needs GPU + base models).")
    ap.add_argument("--save", action="store_true", help="Render the .tex from the cache.")
    args = ap.parse_args()
    if args.compute:
        compute()
    if args.save or not args.compute:
        render()


if __name__ == "__main__":
    main()
