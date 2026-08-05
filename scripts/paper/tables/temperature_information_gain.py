"""Temperature-scaled information gain for the CE-temperature ablation.

The ``information_gain_bits`` stored by ``ProgressiveCrammingTrainer`` (and hence the IG column of
``tab:progressive_temperature`` and the red curve of ``fig:ce_temperature``) is computed by
``compression_horizon.analysis.information_gain.compute_information_gain``, which takes a plain
``F.cross_entropy`` over the *raw* model logits -- i.e. at ``T=1``, with no temperature applied, in
every run including the temperature sweep. That is a confound for the cold runs: the reconstruction
logits of a run trained under ``CE(z/T)`` come out scaled as ``z ~ T`` (see
``tab:temperature_logit_stats``), so reading them back through a ``T=1`` softmax makes a cold run's
predictive distribution look artificially flat and understates its IG.

This script recomputes IG from the saved final-stage embeddings under three conventions, from the
same two forward passes per sample (base LM, and LM conditioned on the ``[mem]`` embedding):

  * ``raw``        H_1(base) - H_1(mem)    -- reproduces the stored value (validation control).
  * ``loss_units`` H_T(base) - H_T(mem)    -- both sides read through the softmax the loss actually
                                             optimizes, ``CE(z/T)``: IG in the loss's own units.
  * ``comp_only``  H_1(base) - H_T(mem)    -- only the crammed side is rescaled, undoing the trivial
                                             ``z ~ T`` inflation while keeping the base LM's code
                                             fixed: "how many bits does mem save under the model's
                                             normal T=1 code, once the reconstruction is put back on
                                             the scale its own loss shaped it at".

All values are TOTAL bits over the crammed sequence (matching the paper's IG definition), averaged
over the 50 PG19 samples. Rows mirror ``temperature_logit_stats.run_rows()`` exactly.

Two-step, like the other cache-backed tables:
    # 1) recompute the cache (needs a GPU + the three base models):
    HF_HOME=... PYTHONPATH=./src:. python scripts/paper/tables/temperature_information_gain.py --compute
    # 2) inspect / render from the committed cache (no GPU):
    PYTHONPATH=./src:. python scripts/paper/tables/temperature_information_gain.py --report
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temperature_logit_stats import run_rows  # noqa: E402  (shared row definition)

CACHE = "artifacts/paper/temperature_information_gain.json"
CHUNK = 256  # sequence positions per CE chunk (keeps the float32 logit copy small)


# --------------------------------------------------------------------------- #
# Compute (GPU)
# --------------------------------------------------------------------------- #
def compute(only_missing=False):
    import torch
    import torch.nn.functional as F
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compression_horizon.paper.tables.progressive import flatten_embedding

    def ce_bits(pred_logits, labels, temperatures):
        """Total next-token CE in bits at each temperature, over already-shifted logits/labels."""
        totals = {T: 0.0 for T in temperatures}
        n = labels.size(1)
        vocab = pred_logits.size(-1)
        for i in range(0, n, CHUNK):
            z = pred_logits[:, i : i + CHUNK].float().reshape(-1, vocab)
            lab = labels[:, i : i + CHUNK].reshape(-1)
            for T in temperatures:
                zt = z if T == 1.0 else z / T
                totals[T] += F.cross_entropy(zt, lab, reduction="sum").item()
        return {T: v / math.log(2) for T, v in totals.items()}, n

    @torch.no_grad()
    def sample_gains(model, tok, row, device, temperature):
        """Return (ig_raw, ig_loss_units, ig_comp_only, n_scored) in bits for one final-stage row."""
        text = row.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            return None
        num_comp = int(row.get("num_compression_tokens", 1))
        emb = torch.tensor(flatten_embedding(row), dtype=torch.bfloat16, device=device).reshape(
            1, num_comp, model.config.hidden_size
        )
        enc = tok(text, truncation=True, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        if input_ids.size(1) < 2:
            return None
        labels = input_ids[:, 1:]
        temps = [1.0] if temperature == 1.0 else [1.0, temperature]

        # H_lm: base model over the same tokens, no [mem].
        base_logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
        h_base, n_scored = ce_bits(base_logits, labels, temps)
        del base_logits

        # H_comp_lm: same tokens, conditioned on the converged [mem] embedding.
        united = torch.cat([emb, model.get_input_embeddings()(input_ids)], dim=1)
        umask = torch.cat([torch.ones((1, num_comp), device=device, dtype=attn.dtype), attn], dim=1)
        comp_logits = model(inputs_embeds=united, attention_mask=umask).logits[:, num_comp:, :][:, :-1, :]
        h_comp, _ = ce_bits(comp_logits, labels, temps)
        del comp_logits, united

        return (
            h_base[1.0] - h_comp[1.0],  # raw (stored convention)
            h_base[temperature] - h_comp[temperature],  # loss units
            h_base[1.0] - h_comp[temperature],  # comp-side rescale only
            n_scored,
        )

    device = torch.device("cuda")
    cache = {}
    if only_missing and os.path.exists(CACHE):
        with open(CACHE) as f:
            cache = json.load(f)

    by_ckpt = {}
    for row in run_rows():
        by_ckpt.setdefault(row[5], []).append(row)

    for ckpt, rows in by_ckpt.items():
        rows = [r for r in rows if os.path.isdir(r[-1])]
        if only_missing:
            rows = [r for r in rows if r[0] not in cache]
        if not rows:
            continue
        print(f"\n=== {ckpt} ({len(rows)} runs) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(ckpt)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device).eval()

        for label, arm, T, dir_short, lr, _, prefix, d in rows:
            ds = load_from_disk(d)
            light = ds.select_columns(["sample_id", "stage_index", "stage_seq_len", "information_gain_bits"])
            final_idx, stored = {}, {}
            for i, r in enumerate(light):
                s = int(r["sample_id"])
                key = (int(r.get("stage_index", 0) or 0), int(r.get("stage_seq_len", 0) or 0))
                if s not in final_idx or key > final_idx[s][1]:
                    final_idx[s] = (i, key)
                    stored[s] = float(r.get("information_gain_bits") or 0.0)
            ig_raw, ig_loss, ig_comp, n_toks, ig_stored = [], [], [], [], []
            for s, (gi, _) in sorted(final_idx.items()):
                g = sample_gains(model, tok, ds[gi], device, float(T))
                if g is None:
                    continue
                ig_raw.append(g[0])
                ig_loss.append(g[1])
                ig_comp.append(g[2])
                n_toks.append(g[3])
                ig_stored.append(stored[s])
            cache[label] = {
                "model": prefix,
                "T": float(T),
                "arm": arm,
                "dir": d,  # keyed on by the table generator's rescaled-IG override
                "n": len(ig_raw),
                "n_tokens_mean": float(np.mean(n_toks)),
                "ig_stored_mean": float(np.mean(ig_stored)),
                "ig_raw_mean": float(np.mean(ig_raw)),
                "ig_raw_std": float(np.std(ig_raw)),
                "ig_loss_units_mean": float(np.mean(ig_loss)),
                "ig_loss_units_std": float(np.std(ig_loss)),
                "ig_comp_only_mean": float(np.mean(ig_comp)),
                "ig_comp_only_std": float(np.std(ig_comp)),
            }
            e = cache[label]
            print(
                f"  {label:22s} n={e['n']:2d} stored={e['ig_stored_mean']:8.1f} "
                f"raw={e['ig_raw_mean']:8.1f} loss_units={e['ig_loss_units_mean']:9.1f} "
                f"comp_only={e['ig_comp_only_mean']:9.1f}",
                flush=True,
            )
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            with open(CACHE, "w") as f:
                json.dump(cache, f, indent=2, sort_keys=True)

        del model
        torch.cuda.empty_cache()

    print(f"\nWrote {CACHE} ({len(cache)} runs)")


# --------------------------------------------------------------------------- #
# Report (no GPU)
# --------------------------------------------------------------------------- #
def report():
    with open(CACHE) as f:
        cache = json.load(f)
    hdr = f"{'run':<22} {'tokens':>7} {'stored':>9} {'raw(=T1)':>9} {'loss units':>11} {'comp only':>10}"
    print(hdr)
    print("-" * len(hdr))
    prev = None
    for label, arm, T, *_ in run_rows():
        if label not in cache:
            continue
        e = cache[label]
        if prev is not None and e["model"] != prev:
            print("-" * len(hdr))
        prev = e["model"]
        print(
            f"{label:<22} {e['n_tokens_mean']:7.1f} {e['ig_stored_mean']:9.1f} {e['ig_raw_mean']:9.1f} "
            f"{e['ig_loss_units_mean']:11.1f} {e['ig_comp_only_mean']:10.1f}"
        )
    drift = [abs(e["ig_raw_mean"] - e["ig_stored_mean"]) / max(1.0, abs(e["ig_stored_mean"])) for e in cache.values()]
    print(f"\nrecomputed-vs-stored raw IG: max relative drift {max(drift) * 100:.2f}% over {len(drift)} runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compute", action="store_true", help="Recompute the cache (needs GPU + base models).")
    ap.add_argument("--only-missing", action="store_true", help="With --compute: only labels absent from the cache.")
    ap.add_argument("--report", action="store_true", help="Print the cached values as a text table.")
    args = ap.parse_args()
    if args.compute:
        compute(only_missing=args.only_missing)
    if args.report or not args.compute:
        report()


if __name__ == "__main__":
    main()
