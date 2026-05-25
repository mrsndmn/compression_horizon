#!/usr/bin/env python3
"""Create depth-ablated checkpoints that keep only the first N and last N
transformer layers, for each N in a sweep (default {1, 2, 4, 8}).

For each N we load the full pretrained model, keep decoder layers
``[0 .. N-1] + [L-N .. L-1]`` (``2N`` layers total), renumber each retained
layer's ``layer_idx`` to its new position (needed for KV-cache / position
bookkeeping if the in-memory model is ever run), set
``config.num_hidden_layers = 2N``, and save a standalone HF checkpoint plus its
tokenizer.

The resulting checkpoints are plain base models (e.g. ``LlamaForCausalLM`` for
SmolLM2) loadable by ``scripts/activation_distillation.py --model_checkpoint
<path>`` for progressive cramming.

Usage:
    python scripts/checkpoints/make_first_last_layers_ckpt.py \
        --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \
        --keep 1 2 4 8 \
        --output_root artifacts/checkpoints
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn


# Mirrors compression_horizon.analysis.attention_intervention.get_decoder_layers
# but inlined to keep this preprocessing utility dependency-light.
def get_decoder_layers(model) -> nn.ModuleList:
    """Return the decoder-layer ModuleList for supported architectures."""
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(f"Unknown architecture: {type(model)}. Cannot locate decoder layers.")


def set_decoder_layers(model, new_layers: nn.ModuleList) -> None:
    """Replace the decoder-layer ModuleList in place."""
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        model.model.language_model.layers = new_layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = new_layers
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        model.gpt_neox.layers = new_layers
    else:
        raise ValueError(f"Unknown architecture: {type(model)}. Cannot set decoder layers.")


def reindex_layers(layers: nn.ModuleList) -> None:
    """Renumber per-layer ``layer_idx`` fields to their new positions."""
    for new_idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx


def first_last_indices(num_layers: int, n: int) -> list[int]:
    """Indices of the first ``n`` and last ``n`` layers (deduped, sorted)."""
    if 2 * n > num_layers:
        raise ValueError(
            f"keep N={n} requires 2N={2 * n} layers but the model only has {num_layers}; "
            "first and last blocks would overlap."
        )
    return sorted(set(list(range(n)) + list(range(num_layers - n, num_layers))))


def first_indices(num_layers: int, n: int) -> list[int]:
    """Indices of the first ``n`` layers only (``[0 .. n-1]``)."""
    if n > num_layers:
        raise ValueError(f"keep N={n} requires {n} layers but the model only has {num_layers}.")
    return list(range(n))


def last_indices(num_layers: int, n: int) -> list[int]:
    """Indices of the last ``n`` layers only (``[L-n .. L-1]``)."""
    if n > num_layers:
        raise ValueError(f"keep N={n} requires {n} layers but the model only has {num_layers}.")
    return list(range(num_layers - n, num_layers))


# keep_mode -> (indices fn, total kept layers given n, output-dir suffix template).
KEEP_MODES = {
    "first_last": (first_last_indices, lambda n: 2 * n, "firstlast{n}"),
    "first": (first_indices, lambda n: n, "first{n}"),
    "last": (last_indices, lambda n: n, "last{n}"),
}


def build_truncated(model_checkpoint: str, n: int, dtype: torch.dtype, keep_mode: str = "first_last"):
    """Load a fresh full model and slice it down per ``keep_mode`` (first-N+last-N or first-N)."""
    from transformers import AutoModelForCausalLM

    indices_fn = KEEP_MODES[keep_mode][0]
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=dtype)
    layers = get_decoder_layers(model)
    num_layers = len(layers)
    keep = indices_fn(num_layers, n)

    kept = nn.ModuleList([layers[i] for i in keep])
    reindex_layers(kept)
    set_decoder_layers(model, kept)
    model.config.num_hidden_layers = len(kept)
    return model, num_layers, keep


def verify_forward(model) -> None:
    """Run a tiny forward pass and assert the logits are finite."""
    model.eval()
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, 8))
    with torch.no_grad():
        out = model(input_ids)
    if not torch.isfinite(out.logits).all():
        raise RuntimeError("Truncated model produced non-finite logits in verification forward pass.")
    print(f"    verify: forward OK, logits {tuple(out.logits.shape)} finite")


def copy_tokenizer(model_checkpoint: str, dst: str) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_checkpoint)
    tok.save_pretrained(dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_checkpoint", default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument(
        "--keep",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Layer counts N (see --keep_mode for how N maps to kept layers).",
    )
    parser.add_argument(
        "--keep_mode",
        choices=list(KEEP_MODES),
        default="first_last",
        help="first_last: keep first N + last N layers (2N total) -> '<model>-firstlast<N>'. "
        "first: keep only the first N layers (N total) -> '<model>-first<N>'.",
    )
    parser.add_argument("--output_root", default="artifacts/checkpoints")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Dtype to load/save weights in (float32 keeps it lossless).",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip the post-build forward-pass check.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if the output dir exists.")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    model_short = args.model_checkpoint.rstrip("/").split("/")[-1]
    _, total_fn, suffix_tmpl = KEEP_MODES[args.keep_mode]

    print(f"Model: {args.model_checkpoint}  (short: {model_short})")
    print(f"keep_mode={args.keep_mode}  N: {args.keep}  -> total layers {[total_fn(n) for n in args.keep]}")

    for n in args.keep:
        expected = total_fn(n)
        dst = os.path.join(args.output_root, f"{model_short}-{suffix_tmpl.format(n=n)}")
        if os.path.isdir(dst) and not args.overwrite:
            print(f"[N={n}] exists, skip: {dst}")
            continue

        print(f"[N={n}] building ({args.keep_mode}, {expected} layers) ...")
        model, num_layers, keep = build_truncated(args.model_checkpoint, n, dtype, args.keep_mode)
        assert len(get_decoder_layers(model)) == expected == len(keep)
        assert model.config.num_hidden_layers == expected
        print(f"    kept original layer indices {keep} of {num_layers}")
        print(f"    params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

        if not args.no_verify:
            verify_forward(model)

        os.makedirs(dst, exist_ok=True)
        model.save_pretrained(dst)
        copy_tokenizer(args.model_checkpoint, dst)
        print(f"[N={n}] saved -> {dst}")

    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
