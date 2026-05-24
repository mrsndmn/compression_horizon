#!/usr/bin/env python3
"""Create initialization-ablated checkpoints: randomly re-initialize exactly one
component of a base model while keeping the rest pretrained.

Components
----------
- ``layers``     -- re-init the decoder transformer layers (``model.model.layers``);
                    the input embedding, output projection, and final norm stay
                    pretrained.
- ``lm_head``    -- re-init only the output projection. The model ties its
                    embeddings to ``lm_head`` (``tie_word_embeddings=True``), so we
                    first UNTIE (clone the shared weight into an independent
                    ``lm_head`` and set ``tie_word_embeddings=False``) and then
                    randomize the ``lm_head`` only -- the input embedding stays
                    pretrained.
- ``embeddings`` -- re-init only the input token embeddings. Likewise UNTIE first
                    so the (now independent) ``lm_head`` keeps pretrained weights.

Randomization uses the model's own ``_init_weights`` (``config.initializer_range``),
matching how the model would be initialized from scratch; norm gains inside a
re-initialized subtree are reset to 1.0.

Usage:
    python scripts/checkpoints/make_random_init_ckpt.py \
        --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \
        --components layers lm_head embeddings \
        --output_root artifacts/checkpoints
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

COMPONENTS = ("layers", "lm_head", "embeddings")
# Distinct seeds per component so the random draws are independent.
SEEDS = {"layers": 42, "lm_head": 43, "embeddings": 44}


# Mirrors compression_horizon.analysis.attention_intervention.get_decoder_layers
# but inlined to keep this preprocessing utility dependency-light.
def get_decoder_layers(model) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(f"Unknown architecture: {type(model)}. Cannot locate decoder layers.")


def reinit_subtree(model, module: nn.Module) -> None:
    """Randomly re-initialize every parameter under ``module``.

    Writes the tensors directly (Linear/Embedding ~ N(0, initializer_range),
    norm gains -> 1.0). We deliberately do NOT route through
    ``model._init_weights``: in current transformers it early-returns on modules
    flagged ``_is_hf_initialized`` (set when pretrained weights are loaded), so
    ``module.apply(model._init_weights)`` silently leaves loaded layers/embeddings
    untouched. Direct writes mirror the repo precedent
    (``.omc/autoresearch/n1-135m-100tok/make_random_layers_only_ckpt.py``).
    """
    std = float(getattr(model.config, "initializer_range", 0.02))
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.padding_idx is not None:
                    m.weight[m.padding_idx].zero_()
            elif "norm" in type(m).__name__.lower() and hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
                m.weight.fill_(1.0)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)


def untie_embeddings(model) -> None:
    """Give the output projection an independent (cloned) weight and mark the
    config untied, so one of the two tied matrices can be randomized alone."""
    out = model.get_output_embeddings()
    out.weight = nn.Parameter(out.weight.detach().clone())
    model.config.tie_word_embeddings = False
    # Defensive: drop the tied-weight bookkeeping so save/reload keeps both.
    if hasattr(model, "_tied_weights_keys"):
        model._tied_weights_keys = []


def build_ablated(model_checkpoint: str, component: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=dtype)
    torch.manual_seed(SEEDS[component])

    if component == "layers":
        reinit_subtree(model, get_decoder_layers(model))
    elif component == "lm_head":
        untie_embeddings(model)
        reinit_subtree(model, model.get_output_embeddings())
    elif component == "embeddings":
        untie_embeddings(model)
        reinit_subtree(model, model.get_input_embeddings())
    else:
        raise ValueError(f"Unknown component: {component}")

    return model


def verify_forward(model) -> None:
    model.eval()
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, 8))
    with torch.no_grad():
        out = model(input_ids)
    if not torch.isfinite(out.logits).all():
        raise RuntimeError("Ablated model produced non-finite logits in verification forward pass.")
    print(f"    verify: forward OK, logits {tuple(out.logits.shape)} finite")


def copy_tokenizer(model_checkpoint: str, dst: str) -> None:
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model_checkpoint).save_pretrained(dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_checkpoint", default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--components", nargs="+", default=list(COMPONENTS), choices=COMPONENTS)
    parser.add_argument("--output_root", default="artifacts/checkpoints")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    model_short = args.model_checkpoint.rstrip("/").split("/")[-1]
    # Filesystem-friendly dir tag per component (lm_head -> lmhead).
    tag = {"layers": "layers", "lm_head": "lmhead", "embeddings": "embeddings"}

    print(f"Model: {args.model_checkpoint}  (short: {model_short})")
    print(f"Components: {args.components}")

    for component in args.components:
        dst = os.path.join(args.output_root, f"{model_short}-randinit-{tag[component]}")
        if os.path.isdir(dst) and not args.overwrite:
            print(f"[{component}] exists, skip: {dst}")
            continue

        print(f"[{component}] building (random init of {component}) ...")
        model = build_ablated(args.model_checkpoint, component, dtype)
        print(f"    tie_word_embeddings={model.config.tie_word_embeddings}")
        print(f"    params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

        if not args.no_verify:
            verify_forward(model)

        os.makedirs(dst, exist_ok=True)
        model.save_pretrained(dst)
        copy_tokenizer(args.model_checkpoint, dst)
        print(f"[{component}] saved -> {dst}")

    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
