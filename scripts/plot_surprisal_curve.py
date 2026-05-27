"""Plot a model's next-token surprisal curve for one dataset sample.

For the chosen ``--model_checkpoint`` and the ``--sample_index``-th example of
``--dataset_name``, this runs a single forward pass of the (frozen) base model and
plots:

* per-token surprisal  s_i = -log2 p(x_i | x_<i)  in bits, and
* the cumulative description length  DL(n) = sum_{i<n} s_i  in bits,

i.e. the running number of bits the model "spends" to encode the prefix. DL(n) is
the same H_LM quantity used for information gain, so it is also the curve that
determines the progressive-cramming horizon: the largest prefix a single
compression token can reconstruct carries a roughly fixed number of bits, so the
prefix length at which DL(n) crosses that budget is the predicted horizon.

Example:
    python scripts/plot_surprisal_curve.py \\
        --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \\
        --dataset_name LarryLovestein/pg19_1k --sample_index 0 \\
        --max_length 768 --output artifacts/figures/surprisal_pg19_s0.png
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Datasets whose only split is ``train`` (mirrors scripts/activation_distillation.py).
_TRAIN_SPLIT_DATASETS = {"LarryLovestein/pg19_1k", "LarryLovestein/fanfics_1k"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_checkpoint", required=True, help="HF model id or local path (frozen base LM).")
    p.add_argument("--dataset_name", required=True, help="HF dataset id (e.g. LarryLovestein/pg19_1k).")
    p.add_argument("--sample_index", type=int, required=True, help="Index of the example to plot.")
    p.add_argument("--split", default=None, help="Dataset split (default: 'train' for the *_1k sets, else 'test').")
    p.add_argument("--text_column", default="text", help="Column holding the raw text (default: text).")
    p.add_argument("--max_length", type=int, default=1024, help="Max tokens to score (default: 1024).")
    p.add_argument(
        "--budget_bits", type=float, default=None, help="Optional IG budget; draws a horizon marker where DL crosses it."
    )
    p.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu).")
    p.add_argument("--output", default=None, help="Figure path (default: artifacts/figures/surprisal_<...>.png).")
    return p.parse_args()


def resolve_split(dataset_name: str, split: str | None) -> str:
    if split is not None:
        return split
    return "train" if dataset_name in _TRAIN_SPLIT_DATASETS else "test"


def default_output(args: argparse.Namespace) -> str:
    model_short = args.model_checkpoint.rstrip("/").split("/")[-1]
    ds_short = args.dataset_name.rstrip("/").split("/")[-1]
    return os.path.join("artifacts", "figures", f"surprisal_{model_short}_{ds_short}_sample{args.sample_index}.png")


@torch.no_grad()
def per_token_surprisal_bits(model, input_ids: torch.Tensor) -> np.ndarray:
    """Return s_i = -log2 p(x_i | x_<i) in bits for i = 1 .. T-1 (length T-1)."""
    logits = model(input_ids=input_ids).logits  # [1, T, V]
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)  # predictions for tokens 1..T-1
    targets = input_ids[0, 1:]
    nll_nats = -logp[torch.arange(targets.shape[0], device=targets.device), targets]
    return (nll_nats / math.log(2)).cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = resolve_split(args.dataset_name, args.split)

    print(f"Loading dataset {args.dataset_name} (split={split}) ...")
    dataset = load_dataset(args.dataset_name, split=split)
    if not (0 <= args.sample_index < len(dataset)):
        raise IndexError(f"sample_index {args.sample_index} out of range for {len(dataset)} examples.")
    text = dataset[args.sample_index][args.text_column]

    print(f"Loading model {args.model_checkpoint} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=torch.bfloat16).to(device).eval()

    input_ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=args.max_length, return_tensors="pt")[
        "input_ids"
    ].to(device)

    surprisal = per_token_surprisal_bits(model, input_ids)  # [T-1]
    dl = np.cumsum(surprisal)  # dl[k] = description length of the prefix of length k+2 tokens
    prefix_len = np.arange(2, 2 + len(dl))  # prefix length n for each cumulative value
    total_bits = float(dl[-1]) if len(dl) else 0.0
    mean_bpt = float(surprisal.mean()) if len(surprisal) else 0.0
    print(f"Tokens scored: {len(surprisal)} | total description length: {total_bits:.1f} bits | mean {mean_bpt:.2f} bits/token")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    title_model = args.model_checkpoint.rstrip("/").split("/")[-1]
    title_ds = args.dataset_name.rstrip("/").split("/")[-1]
    fig.suptitle(f"Surprisal curve — {title_model} on {title_ds}[{args.sample_index}]", fontsize=12)

    ax_top.plot(np.arange(1, 1 + len(surprisal)), surprisal, lw=0.8, color="tab:blue")
    ax_top.axhline(mean_bpt, ls="--", lw=1.0, color="gray", label=f"mean {mean_bpt:.2f} bits/token")
    ax_top.set_ylabel("per-token surprisal (bits)")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(alpha=0.3)

    ax_bot.plot(prefix_len, dl, lw=1.4, color="tab:red", label="cumulative description length DL(n)")
    ax_bot.set_xlabel("prefix length n (tokens)")
    ax_bot.set_ylabel("DL(n) (bits)")
    if args.budget_bits is not None:
        ax_bot.axhline(args.budget_bits, ls="--", lw=1.0, color="black", label=f"budget {args.budget_bits:.0f} bits")
        crossings = np.where(dl >= args.budget_bits)[0]
        if len(crossings):
            n_star = int(prefix_len[crossings[0]])
            ax_bot.axvline(n_star, ls=":", lw=1.0, color="green", label=f"horizon n={n_star}")
            print(f"DL crosses {args.budget_bits:.0f} bits at prefix length n = {n_star}")
    ax_bot.legend(loc="upper left", fontsize=8)
    ax_bot.grid(alpha=0.3)

    output = args.output or default_output(args)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=150)
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
