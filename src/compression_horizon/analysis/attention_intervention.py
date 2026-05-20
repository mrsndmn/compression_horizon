"""Attention intervention utilities for compression token analysis."""

import math
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from compression_horizon.analysis.perplexity import estimate_token_perplexity
from compression_horizon.utils.launch import get_device

# Note: `from compression_horizon.train.inputs import build_united_input` is
# imported lazily inside the functions below to avoid a circular import path
# (analysis → train.inputs → train → trainers/full_cramming → analysis).

# ---------------------------------------------------------------------------
# Model architecture helpers
# ---------------------------------------------------------------------------


def get_decoder_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """Return the list of decoder layers for supported model architectures."""
    # Gemma3 (ConditionalGeneration): model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    # Llama, SmolLM2, Gemma2: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Pythia / GPT-NeoX: model.gpt_neox.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(f"Unknown model architecture: {type(model)}. Cannot locate decoder layers.")


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


class EagerAttentionContext:
    """Context manager that temporarily switches model to eager attention and restores original on exit."""

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
        self._original_attn_implementation = None

    def __enter__(self):
        self._original_attn_implementation = getattr(self.model.config, "_attn_implementation", None)
        self.model.set_attn_implementation("eager")
        return self

    def __exit__(self, *args):
        if self._original_attn_implementation is not None:
            self.model.set_attn_implementation(self._original_attn_implementation)


class AttentionKnockoutContext:
    """Context manager that masks attention to compression token positions at specified layers."""

    def __init__(
        self,
        model: PreTrainedModel,
        knockout_layers: list[int],
        num_compression_tokens: int,
    ):
        self.model = model
        self.knockout_layers = knockout_layers
        self.num_compression_tokens = num_compression_tokens
        self.hooks: list[torch.utils.hooks.RemovableHook] = []
        self.layers = get_decoder_layers(model)
        self._eager_ctx = EagerAttentionContext(model)

    def _make_hook(self):
        num_compression_tokens = self.num_compression_tokens

        def hook_fn(module, args, kwargs):
            mask = kwargs.get("attention_mask", None)
            if mask is not None and mask.dim() == 4:
                mask = mask.clone()
                mask[:, :, :, :num_compression_tokens] = torch.finfo(mask.dtype).min
                kwargs["attention_mask"] = mask
            return args, kwargs

        return hook_fn

    def __enter__(self):
        self._eager_ctx.__enter__()
        hook_fn = self._make_hook()
        for layer_idx in self.knockout_layers:
            handle = self.layers[layer_idx].register_forward_pre_hook(hook_fn, with_kwargs=True)
            self.hooks.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self._eager_ctx.__exit__(*args)


# ---------------------------------------------------------------------------
# Attention analysis
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_attention_mass_per_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    compression_token_embeddings: torch.Tensor,
    context: str,
    num_compression_tokens: int = 1,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[float]:
    """Compute per-layer attention mass on compression token positions."""
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    encoded = tokenizer(context, truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    embed_fn = model.get_input_embeddings()
    token_embeddings = embed_fn(input_ids)  # [1, sequence, hidden]

    compression_token_embeddings = compression_token_embeddings.unsqueeze(0)
    compression_attention_mask = torch.ones((1, num_compression_tokens), dtype=attention_mask.dtype, device=device)
    from compression_horizon.train.inputs import build_united_input  # deferred (circular)

    united_token_embeddings, united_attention_mask = build_united_input(
        compression_token_embeddings,
        compression_attention_mask,
        token_embeddings,
        attention_mask,
    )

    with EagerAttentionContext(model):
        outputs = model(
            inputs_embeds=united_token_embeddings,
            attention_mask=united_attention_mask,
            output_attentions=True,
        )

    # attentions: tuple of [1, head, sequence, sequence] per layer
    attention_mass = []
    for attn_layer in outputs.attentions:
        avg_over_heads = attn_layer.mean(dim=1)  # [1, sequence, sequence]
        seq_len = int(compression_attention_mask.sum().item())
        comp_attn = avg_over_heads[0, :seq_len, :num_compression_tokens].sum(dim=-1)  # [sequence]
        mass_pct = comp_attn.mean().item() * 100.0
        attention_mass.append(mass_pct)
    return attention_mass


# ---------------------------------------------------------------------------
# PPL with knockout
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_ppl_with_compression_and_knockout_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    compression_token_embeddings: list[torch.Tensor],
    contexts: list[str],
    endings: list[str],
    knockout_layers: list[int],
    num_compression_tokens: int = 1,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[float]:
    """Compute PPL with compression tokens prepended and attention knockout at specified layers."""
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    if len(contexts) == 0:
        return []

    # Combine contexts and endings
    full_texts = [f"{ctx} {end}" for ctx, end in zip(contexts, endings)]

    # Tokenize with padding
    encoded = tokenizer(
        full_texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    token_embeddings = model.get_input_embeddings()(input_ids)
    num_compression_tokens = compression_token_embeddings[0].shape[0]
    batched_compression = torch.stack(
        [c.to(token_embeddings.dtype).to(device) for c in compression_token_embeddings],
        dim=0,
    )
    compression_attention_mask = torch.ones(
        (len(compression_token_embeddings), num_compression_tokens),
        dtype=attention_mask.dtype,
        device=device,
    )
    from compression_horizon.train.inputs import build_united_input  # deferred (circular)

    united_token_embeddings, united_attention_mask = build_united_input(
        batched_compression, compression_attention_mask, token_embeddings, attention_mask
    )

    with AttentionKnockoutContext(model, knockout_layers, num_compression_tokens):
        outputs = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask)

    ppls = []
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())
        sample_logits = outputs.logits[i : i + 1, num_compression_tokens : num_compression_tokens + seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]
        sample_attention = attention_mask[i : i + 1, :seq_len]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids, sample_attention)
        ppls.append(ppl if not math.isnan(ppl) else float("inf"))
    return ppls


# ---------------------------------------------------------------------------
# Reconstruction accuracy with knockout
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_reconstruction_accuracy_with_knockout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    compression_token_embeddings: torch.Tensor,
    context: str,
    knockout_layers: list[int],
    num_compression_tokens: int = 1,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> float:
    """Compute teacher-forced reconstruction accuracy of the compressed prefix under knockout."""
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    encoded = tokenizer(context, truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [1, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [1, seq_len]

    token_embeddings = model.get_input_embeddings()(input_ids)  # [1, seq_len, hidden]
    compression_attention_mask = torch.ones((1, num_compression_tokens), dtype=attention_mask.dtype, device=device)
    from compression_horizon.train.inputs import build_united_input  # deferred (circular)

    united_token_embeddings, united_attention_mask = build_united_input(
        compression_token_embeddings.unsqueeze(0), compression_attention_mask, token_embeddings, attention_mask
    )

    with AttentionKnockoutContext(model, knockout_layers, num_compression_tokens):
        outputs = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask)

    seq_len = int(attention_mask.sum().item())
    # logits[num_compression_tokens - 1] predicts the first context token; in general
    # logits[num_compression_tokens + j - 1] predicts context token j. We score
    # context tokens 0..seq_len-1 from logits[num_compression_tokens - 1..num_compression_tokens + seq_len - 2].
    pred_logits = outputs.logits[0, num_compression_tokens - 1 : num_compression_tokens + seq_len - 1]
    predicted_tokens = pred_logits.argmax(dim=-1)
    target_tokens = input_ids[0, :seq_len]

    return float((predicted_tokens == target_tokens).float().mean().item())


# ---------------------------------------------------------------------------
# High-level intervention evaluation
# ---------------------------------------------------------------------------


def evaluate_sample_interventions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    compression_embedding: torch.Tensor,
    context: str,
    endings: list[str],
    num_compression_tokens: int,
    num_model_layers: int,
    device: torch.device,
    add_special_tokens: bool = True,
    skip_per_layer: bool = False,
    skip_cumulative: bool = False,
    skip_reverse_cumulative: bool = False,
) -> dict:
    """Run all intervention evaluations for a single sample.

    Returns a dict with keys (present only if not skipped):
        - attention_mass: list[float] per layer
        - per_layer_knockout: dict[int, list[float]] layer -> PPLs
        - cumulative_knockout: dict[int, list[float]] layer -> PPLs
        - reverse_cumulative_knockout: dict[int, list[float]] layer -> PPLs
    """
    result: dict = {}
    num_endings = len(endings)
    contexts_for_sample = [context] * num_endings
    comp_embeds_for_sample = [compression_embedding] * num_endings

    # Attention mass
    try:
        attn_mass = compute_attention_mass_per_layer(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=compression_embedding,
            context=context,
            num_compression_tokens=num_compression_tokens,
            device=device,
            add_special_tokens=add_special_tokens,
        )
    except Exception as e:
        print(f"Error computing attention mass: {e}")
        attn_mass = [0.0] * num_model_layers
    result["attention_mass"] = attn_mass

    def _run_knockout(knockout_layers: list[int]) -> list[float]:
        return compute_ppl_with_compression_and_knockout_batch(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=comp_embeds_for_sample,
            contexts=contexts_for_sample,
            endings=endings,
            knockout_layers=knockout_layers,
            num_compression_tokens=num_compression_tokens,
            device=device,
            add_special_tokens=add_special_tokens,
        )

    def _run_reconstruction(knockout_layers: list[int]) -> float:
        return compute_reconstruction_accuracy_with_knockout(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=compression_embedding,
            context=context,
            knockout_layers=knockout_layers,
            num_compression_tokens=num_compression_tokens,
            device=device,
            add_special_tokens=add_special_tokens,
        )

    # Per-layer knockout
    if not skip_per_layer:
        per_layer = {}
        per_layer_recon = {}
        for li in range(num_model_layers):
            try:
                per_layer[li] = _run_knockout([li])
            except Exception as e:
                print(f"Error in per-layer KO (layer {li}): {e}")
                per_layer[li] = [float("inf")] * num_endings
            try:
                per_layer_recon[li] = _run_reconstruction([li])
            except Exception as e:
                print(f"Error in per-layer recon (layer {li}): {e}")
                per_layer_recon[li] = 0.0
        result["per_layer_knockout"] = per_layer
        result["per_layer_reconstruction"] = per_layer_recon

    # Cumulative knockout (layers 0..li)
    if not skip_cumulative:
        cumulative = {}
        cumulative_recon = {}
        for li in range(num_model_layers):
            try:
                cumulative[li] = _run_knockout(list(range(li + 1)))
            except Exception as e:
                print(f"Error in cumulative KO (layers 0..{li}): {e}")
                cumulative[li] = [float("inf")] * num_endings
            try:
                cumulative_recon[li] = _run_reconstruction(list(range(li + 1)))
            except Exception as e:
                print(f"Error in cumulative recon (layers 0..{li}): {e}")
                cumulative_recon[li] = 0.0
        result["cumulative_knockout"] = cumulative
        result["cumulative_reconstruction"] = cumulative_recon

    # Reverse cumulative knockout (layers li..L-1)
    if not skip_reverse_cumulative:
        reverse_cumulative = {}
        reverse_cumulative_recon = {}
        for li in range(num_model_layers):
            try:
                reverse_cumulative[li] = _run_knockout(list(range(li, num_model_layers)))
            except Exception as e:
                print(f"Error in reverse cumulative KO (layers {li}..{num_model_layers - 1}): {e}")
                reverse_cumulative[li] = [float("inf")] * num_endings
            try:
                reverse_cumulative_recon[li] = _run_reconstruction(list(range(li, num_model_layers)))
            except Exception as e:
                print(f"Error in reverse cumulative recon (layers {li}..{num_model_layers - 1}): {e}")
                reverse_cumulative_recon[li] = 0.0
        result["reverse_cumulative_knockout"] = reverse_cumulative
        result["reverse_cumulative_reconstruction"] = reverse_cumulative_recon

    return result


def build_knockout_result_entry(ko_ppls: list[float], label: int) -> dict:
    """Build a single knockout result dict with PPLs, predicted label and correctness."""
    pred = int(torch.tensor(ko_ppls).argmin().item())
    return {
        "ppls": ko_ppls,
        "predicted_label": pred,
        "is_correct": pred == label,
    }


def build_intervention_result(
    intervention_data: dict,
    label: int,
    num_model_layers: int,
) -> dict:
    """Convert raw intervention data into result entries for JSON output.

    Returns a dict with keys matching the script's existing output format:
        attention_mass_per_layer, per_layer_knockout, cumulative_knockout, reverse_cumulative_knockout
    """
    result: dict = {}

    if "attention_mass" in intervention_data:
        result["attention_mass_per_layer"] = intervention_data["attention_mass"]

    for key in ("per_layer_knockout", "cumulative_knockout", "reverse_cumulative_knockout"):
        if key in intervention_data:
            result[key] = {
                str(li): build_knockout_result_entry(ko_ppls, label) for li, ko_ppls in intervention_data[key].items()
            }

    # Reconstruction accuracy per layer (teacher-forced prefix reconstruction)
    for key in ("per_layer_reconstruction", "cumulative_reconstruction", "reverse_cumulative_reconstruction"):
        if key in intervention_data:
            result[key] = {str(li): acc for li, acc in intervention_data[key].items()}

    return result


def build_intervention_summary(
    results: list[dict],
    num_model_layers: int,
    skip_per_layer: bool = False,
    skip_cumulative: bool = False,
    skip_reverse_cumulative: bool = False,
) -> dict:
    """Aggregate per-sample intervention results into a summary.

    Args:
        results: list of result dicts (each sample), as stored in the output JSON.
        num_model_layers: total number of decoder layers.

    Returns:
        Summary dict with per_layer_knockout, cumulative_knockout,
        reverse_cumulative_knockout accuracy tables, and avg_attention_mass_per_layer.
    """
    summary: dict = {}

    knockout_keys = []
    if not skip_per_layer:
        knockout_keys.append("per_layer_knockout")
    if not skip_cumulative:
        knockout_keys.append("cumulative_knockout")
    if not skip_reverse_cumulative:
        knockout_keys.append("reverse_cumulative_knockout")

    for key in knockout_keys:
        correct = {li: 0 for li in range(num_model_layers)}
        total = {li: 0 for li in range(num_model_layers)}
        for r in results:
            if key not in r:
                continue
            for li_str, entry in r[key].items():
                li = int(li_str)
                total[li] += 1
                if entry["is_correct"]:
                    correct[li] += 1
        layer_summary = {}
        for li in range(num_model_layers):
            layer_summary[str(li)] = {
                "accuracy": correct[li] / total[li] if total[li] > 0 else 0.0,
                "correct": correct[li],
                "total": total[li],
            }
        summary[key] = layer_summary

    # Average reconstruction accuracy per layer
    recon_keys = []
    if not skip_per_layer:
        recon_keys.append("per_layer_reconstruction")
    if not skip_cumulative:
        recon_keys.append("cumulative_reconstruction")
    if not skip_reverse_cumulative:
        recon_keys.append("reverse_cumulative_reconstruction")

    for key in recon_keys:
        acc_sums = {li: 0.0 for li in range(num_model_layers)}
        acc_counts = {li: 0 for li in range(num_model_layers)}
        for r in results:
            if key not in r:
                continue
            for li_str, acc in r[key].items():
                li = int(li_str)
                acc_sums[li] += acc
                acc_counts[li] += 1
        layer_recon = {}
        for li in range(num_model_layers):
            layer_recon[str(li)] = {
                "avg_accuracy": acc_sums[li] / acc_counts[li] if acc_counts[li] > 0 else 0.0,
                "total": acc_counts[li],
            }
        summary[key] = layer_recon

    # Average attention mass
    all_attn_mass = [r["attention_mass_per_layer"] for r in results if "attention_mass_per_layer" in r]
    if all_attn_mass:
        avg_attn_mass = [sum(s[li] for s in all_attn_mass) / len(all_attn_mass) for li in range(num_model_layers)]
        summary["avg_attention_mass_per_layer"] = avg_attn_mass

    return summary


def print_intervention_summary(
    summary: dict,
    num_model_layers: int,
    baseline_accuracy: float,
) -> None:
    """Print a human-readable intervention summary to stdout."""
    if "per_layer_knockout" in summary:
        print("\nPer-layer Knockout:")
        best_layer = max(
            summary["per_layer_knockout"],
            key=lambda li: summary["per_layer_knockout"][li]["accuracy"],
        )
        worst_layer = min(
            summary["per_layer_knockout"],
            key=lambda li: summary["per_layer_knockout"][li]["accuracy"],
        )
        print(
            f"  Best single-layer KO: layer {best_layer} (accuracy={summary['per_layer_knockout'][best_layer]['accuracy']:.4f})"
        )
        print(
            f"  Worst single-layer KO: layer {worst_layer} (accuracy={summary['per_layer_knockout'][worst_layer]['accuracy']:.4f})"
        )

    if "per_layer_reconstruction" in summary:
        best_recon = max(
            summary["per_layer_reconstruction"],
            key=lambda li: summary["per_layer_reconstruction"][li]["avg_accuracy"],
        )
        worst_recon = min(
            summary["per_layer_reconstruction"],
            key=lambda li: summary["per_layer_reconstruction"][li]["avg_accuracy"],
        )
        print(
            f"  Best recon layer: {best_recon} (avg_accuracy={summary['per_layer_reconstruction'][best_recon]['avg_accuracy']:.4f})"
        )
        print(
            f"  Worst recon layer: {worst_recon} (avg_accuracy={summary['per_layer_reconstruction'][worst_recon]['avg_accuracy']:.4f})"
        )

    if "cumulative_knockout" in summary:
        print("\nCumulative Knockout:")
        last_layer = str(num_model_layers - 1)
        full_ko_acc = summary["cumulative_knockout"][last_layer]["accuracy"]
        print(f"  Full knockout (all layers) accuracy: {full_ko_acc:.4f}")
        print(f"  Base accuracy: {baseline_accuracy:.4f}")
        print(f"  Sanity check delta (should be ~0): {full_ko_acc - baseline_accuracy:+.4f}")

    if "reverse_cumulative_knockout" in summary:
        print("\nReverse Cumulative Knockout:")
        full_rev_ko_acc = summary["reverse_cumulative_knockout"]["0"]["accuracy"]
        print(f"  Full reverse knockout (all layers) accuracy: {full_rev_ko_acc:.4f}")
        print(f"  Base accuracy: {baseline_accuracy:.4f}")
        print(f"  Sanity check delta (should be ~0): {full_rev_ko_acc - baseline_accuracy:+.4f}")
