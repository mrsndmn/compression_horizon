"""Tests for compression_horizon.analysis.attention_intervention module.

Tests all intervention functions across the three architecture families
covered by MODEL_CONFIGS: Llama/SmolLM2, Pythia (GPT-NeoX), and Gemma3.
"""

import os
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.analysis.attention_intervention import (
    AttentionKnockoutContext,
    EagerAttentionContext,
    build_intervention_result,
    build_intervention_summary,
    compute_attention_mass_per_layer,
    compute_ppl_with_compression_and_knockout_batch,
    evaluate_sample_interventions,
    get_decoder_layers,
    print_intervention_summary,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Small model representatives for each architecture family in MODEL_CONFIGS:
#   unsloth/Meta-Llama-3.1-8B, HuggingFaceTB/SmolLM2-1.7B -> SmolLM2-135M (model.model.layers)
#   EleutherAI/pythia-1.4b -> pythia-160m (model.gpt_neox.layers)
#   unsloth/gemma-3-4b-pt -> gemma-3-270m (model.model.language_model.layers)
MODEL_CONFIGS = [
    {
        "checkpoint": "HuggingFaceTB/SmolLM2-135M",
        "arch_family": "llama",
        "add_special_tokens": True,
    },
    {
        "checkpoint": "EleutherAI/pythia-160m",
        "arch_family": "pythia",
        "add_special_tokens": True,
    },
    {
        "checkpoint": "unsloth/gemma-3-270m",
        "arch_family": "gemma3",
        "add_special_tokens": True,
    },
]

SAMPLE_CONTEXT = "A man is sitting on a roof. He starts"
SAMPLE_ENDINGS = [
    "playing the guitar.",
    "eating a sandwich.",
    "flying into space.",
    "reading a newspaper.",
]


def _model_id(cfg):
    return cfg["checkpoint"].split("/")[-1]


@pytest.fixture(params=MODEL_CONFIGS, ids=[_model_id(c) for c in MODEL_CONFIGS], scope="module")
def model_bundle(request):
    """Load model, tokenizer, and config for each architecture."""
    cfg = request.param
    checkpoint = cfg["checkpoint"]
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": cfg,
    }


@pytest.fixture(scope="module")
def compression_embedding(model_bundle):
    """Create a random compression embedding matching the model's hidden size."""
    model = model_bundle["model"]
    hidden_size = model.config.hidden_size
    emb = torch.randn(1, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.02
    return emb


# ---------------------------------------------------------------------------
# get_decoder_layers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGetDecoderLayers:
    def test_returns_module_list(self, model_bundle):
        layers = get_decoder_layers(model_bundle["model"])
        assert isinstance(layers, torch.nn.ModuleList)
        assert len(layers) > 0

    def test_layer_count_matches_config(self, model_bundle):
        model = model_bundle["model"]
        layers = get_decoder_layers(model)
        expected = model.config.num_hidden_layers
        assert len(layers) == expected


# ---------------------------------------------------------------------------
# EagerAttentionContext
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestEagerAttentionContext:
    def test_switches_to_eager_and_restores(self, model_bundle):
        model = model_bundle["model"]
        original_impl = getattr(model.config, "_attn_implementation", None)
        with EagerAttentionContext(model):
            assert model.config._attn_implementation == "eager"
        restored = getattr(model.config, "_attn_implementation", None)
        assert restored == original_impl


# ---------------------------------------------------------------------------
# AttentionKnockoutContext
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAttentionKnockoutContext:
    def test_hooks_registered_and_removed(self, model_bundle):
        model = model_bundle["model"]
        num_layers = len(get_decoder_layers(model))
        knockout_layers = [0, num_layers - 1]
        ctx = AttentionKnockoutContext(model, knockout_layers, num_compression_tokens=1)
        with ctx:
            assert len(ctx.hooks) == len(knockout_layers)
        assert len(ctx.hooks) == 0

    def test_forward_pass_works_with_knockout(self, model_bundle, compression_embedding):
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        num_layers = len(get_decoder_layers(model))

        encoded = tokenizer(SAMPLE_CONTEXT, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"].to("cuda")
        token_embs = model.get_input_embeddings()(input_ids)
        comp_emb = compression_embedding.unsqueeze(0)  # [1, 1, hidden]
        united = torch.cat([comp_emb, token_embs], dim=1)
        attn_mask = torch.ones(1, united.shape[1], dtype=torch.long, device="cuda")

        with AttentionKnockoutContext(model, list(range(num_layers)), num_compression_tokens=1):
            outputs = model(inputs_embeds=united, attention_mask=attn_mask)

        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[1] == united.shape[1]


# ---------------------------------------------------------------------------
# compute_attention_mass_per_layer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestComputeAttentionMassPerLayer:
    def test_returns_correct_length(self, model_bundle, compression_embedding):
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        num_layers = len(get_decoder_layers(model))
        cfg = model_bundle["config"]

        mass = compute_attention_mass_per_layer(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=compression_embedding,
            context=SAMPLE_CONTEXT,
            num_compression_tokens=1,
            device=torch.device("cuda"),
            add_special_tokens=cfg["add_special_tokens"],
        )
        assert len(mass) == num_layers

    def test_values_in_valid_range(self, model_bundle, compression_embedding):
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        cfg = model_bundle["config"]

        mass = compute_attention_mass_per_layer(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=compression_embedding,
            context=SAMPLE_CONTEXT,
            num_compression_tokens=1,
            device=torch.device("cuda"),
            add_special_tokens=cfg["add_special_tokens"],
        )
        for val in mass:
            assert 0.0 <= val <= 100.0, f"Attention mass {val} out of range [0, 100]"


# ---------------------------------------------------------------------------
# compute_ppl_with_compression_and_knockout_batch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestComputePPLWithKnockout:
    def test_returns_finite_ppls(self, model_bundle, compression_embedding):
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        cfg = model_bundle["config"]

        contexts = [SAMPLE_CONTEXT] * 2
        endings = SAMPLE_ENDINGS[:2]
        comp_embeds = [compression_embedding] * 2

        ppls = compute_ppl_with_compression_and_knockout_batch(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=comp_embeds,
            contexts=contexts,
            endings=endings,
            knockout_layers=[0],
            num_compression_tokens=1,
            device=torch.device("cuda"),
            add_special_tokens=cfg["add_special_tokens"],
        )
        assert len(ppls) == 2
        for ppl in ppls:
            assert ppl > 0, "PPL should be positive"

    def test_empty_input(self, model_bundle):
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]

        ppls = compute_ppl_with_compression_and_knockout_batch(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=[],
            contexts=[],
            endings=[],
            knockout_layers=[0],
            num_compression_tokens=1,
            device=torch.device("cuda"),
        )
        assert ppls == []

    def test_full_knockout_differs_from_no_knockout(self, model_bundle, compression_embedding):
        """PPL with all layers knocked out should generally differ from single-layer knockout."""
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        cfg = model_bundle["config"]
        num_layers = len(get_decoder_layers(model))

        contexts = [SAMPLE_CONTEXT]
        endings = [SAMPLE_ENDINGS[0]]
        comp_embeds = [compression_embedding]

        ppls_single = compute_ppl_with_compression_and_knockout_batch(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=comp_embeds,
            contexts=contexts,
            endings=endings,
            knockout_layers=[0],
            num_compression_tokens=1,
            device=torch.device("cuda"),
            add_special_tokens=cfg["add_special_tokens"],
        )
        ppls_all = compute_ppl_with_compression_and_knockout_batch(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=comp_embeds,
            contexts=contexts,
            endings=endings,
            knockout_layers=list(range(num_layers)),
            num_compression_tokens=1,
            device=torch.device("cuda"),
            add_special_tokens=cfg["add_special_tokens"],
        )
        # With all layers knocked out vs one layer, PPLs should differ
        assert ppls_single[0] != ppls_all[0]


# ---------------------------------------------------------------------------
# evaluate_sample_interventions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestEvaluateSampleInterventions:
    def test_full_evaluation(self, model_bundle, compression_embedding):
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        cfg = model_bundle["config"]
        num_layers = len(get_decoder_layers(model))

        result = evaluate_sample_interventions(
            model=model,
            tokenizer=tokenizer,
            compression_embedding=compression_embedding,
            context=SAMPLE_CONTEXT,
            endings=SAMPLE_ENDINGS,
            num_compression_tokens=1,
            num_model_layers=num_layers,
            device=torch.device("cuda"),
            add_special_tokens=cfg["add_special_tokens"],
            skip_per_layer=False,
            skip_cumulative=False,
            skip_reverse_cumulative=False,
        )

        # Check all keys present
        assert "attention_mass" in result
        assert "per_layer_knockout" in result
        assert "cumulative_knockout" in result
        assert "reverse_cumulative_knockout" in result

        # Check shapes
        assert len(result["attention_mass"]) == num_layers
        assert len(result["per_layer_knockout"]) == num_layers
        assert len(result["cumulative_knockout"]) == num_layers
        assert len(result["reverse_cumulative_knockout"]) == num_layers

        # Each knockout entry should have PPLs for all endings
        for li in range(num_layers):
            assert len(result["per_layer_knockout"][li]) == len(SAMPLE_ENDINGS)
            assert len(result["cumulative_knockout"][li]) == len(SAMPLE_ENDINGS)
            assert len(result["reverse_cumulative_knockout"][li]) == len(SAMPLE_ENDINGS)

    def test_skip_all_knockouts(self, model_bundle, compression_embedding):
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        cfg = model_bundle["config"]
        num_layers = len(get_decoder_layers(model))

        result = evaluate_sample_interventions(
            model=model,
            tokenizer=tokenizer,
            compression_embedding=compression_embedding,
            context=SAMPLE_CONTEXT,
            endings=SAMPLE_ENDINGS,
            num_compression_tokens=1,
            num_model_layers=num_layers,
            device=torch.device("cuda"),
            add_special_tokens=cfg["add_special_tokens"],
            skip_per_layer=True,
            skip_cumulative=True,
            skip_reverse_cumulative=True,
        )

        # Only attention mass should be present
        assert "attention_mass" in result
        assert "per_layer_knockout" not in result
        assert "cumulative_knockout" not in result
        assert "reverse_cumulative_knockout" not in result


# ---------------------------------------------------------------------------
# build_intervention_result
# ---------------------------------------------------------------------------


class TestBuildInterventionResult:
    def test_formats_knockout_ppls(self):
        intervention_data = {
            "attention_mass": [5.0, 10.0, 15.0],
            "per_layer_knockout": {
                0: [2.0, 3.0, 1.0, 4.0],
                1: [3.0, 2.0, 4.0, 1.0],
                2: [1.0, 4.0, 3.0, 2.0],
            },
        }
        result = build_intervention_result(intervention_data, label=2, num_model_layers=3)

        assert result["attention_mass_per_layer"] == [5.0, 10.0, 15.0]
        assert "per_layer_knockout" in result

        # Layer 0: min PPL at index 2 -> predicted_label=2, is_correct=True (label=2)
        assert result["per_layer_knockout"]["0"]["predicted_label"] == 2
        assert result["per_layer_knockout"]["0"]["is_correct"] is True

        # Layer 1: min PPL at index 3 -> predicted_label=3, is_correct=False
        assert result["per_layer_knockout"]["1"]["predicted_label"] == 3
        assert result["per_layer_knockout"]["1"]["is_correct"] is False

    def test_missing_keys_handled(self):
        intervention_data = {"attention_mass": [1.0, 2.0]}
        result = build_intervention_result(intervention_data, label=0, num_model_layers=2)
        assert "attention_mass_per_layer" in result
        assert "per_layer_knockout" not in result


# ---------------------------------------------------------------------------
# build_intervention_summary
# ---------------------------------------------------------------------------


class TestBuildInterventionSummary:
    def _make_sample_results(self):
        """Create two sample results with known knockout outcomes."""
        return [
            {
                "attention_mass_per_layer": [10.0, 20.0],
                "per_layer_knockout": {
                    "0": {"ppls": [1.0, 2.0], "predicted_label": 0, "is_correct": True},
                    "1": {"ppls": [2.0, 1.0], "predicted_label": 1, "is_correct": False},
                },
                "cumulative_knockout": {
                    "0": {"ppls": [1.0, 2.0], "predicted_label": 0, "is_correct": True},
                    "1": {"ppls": [2.0, 1.0], "predicted_label": 1, "is_correct": True},
                },
            },
            {
                "attention_mass_per_layer": [30.0, 40.0],
                "per_layer_knockout": {
                    "0": {"ppls": [2.0, 1.0], "predicted_label": 1, "is_correct": True},
                    "1": {"ppls": [1.0, 2.0], "predicted_label": 0, "is_correct": True},
                },
                "cumulative_knockout": {
                    "0": {"ppls": [2.0, 1.0], "predicted_label": 1, "is_correct": False},
                    "1": {"ppls": [1.0, 2.0], "predicted_label": 0, "is_correct": False},
                },
            },
        ]

    def test_per_layer_accuracy(self):
        results = self._make_sample_results()
        summary = build_intervention_summary(results, num_model_layers=2, skip_reverse_cumulative=True)

        # Layer 0: 2/2 correct
        assert summary["per_layer_knockout"]["0"]["accuracy"] == 1.0
        assert summary["per_layer_knockout"]["0"]["correct"] == 2
        assert summary["per_layer_knockout"]["0"]["total"] == 2

        # Layer 1: 1/2 correct
        assert summary["per_layer_knockout"]["1"]["accuracy"] == 0.5
        assert summary["per_layer_knockout"]["1"]["correct"] == 1

    def test_cumulative_accuracy(self):
        results = self._make_sample_results()
        summary = build_intervention_summary(results, num_model_layers=2, skip_reverse_cumulative=True)

        # Layer 0 cumulative: 1/2 correct
        assert summary["cumulative_knockout"]["0"]["accuracy"] == 0.5
        # Layer 1 cumulative: 1/2 correct
        assert summary["cumulative_knockout"]["1"]["accuracy"] == 0.5

    def test_avg_attention_mass(self):
        results = self._make_sample_results()
        summary = build_intervention_summary(results, num_model_layers=2, skip_reverse_cumulative=True)

        assert "avg_attention_mass_per_layer" in summary
        # (10+30)/2 = 20, (20+40)/2 = 30
        assert summary["avg_attention_mass_per_layer"] == [20.0, 30.0]

    def test_skip_flags(self):
        results = self._make_sample_results()
        summary = build_intervention_summary(
            results, num_model_layers=2, skip_per_layer=True, skip_cumulative=True, skip_reverse_cumulative=True
        )
        assert "per_layer_knockout" not in summary
        assert "cumulative_knockout" not in summary
        assert "reverse_cumulative_knockout" not in summary
        # Attention mass is always included
        assert "avg_attention_mass_per_layer" in summary


# ---------------------------------------------------------------------------
# print_intervention_summary (smoke test)
# ---------------------------------------------------------------------------


class TestPrintInterventionSummary:
    def test_prints_without_error(self, capsys):
        summary = {
            "per_layer_knockout": {
                "0": {"accuracy": 0.8, "correct": 8, "total": 10},
                "1": {"accuracy": 0.6, "correct": 6, "total": 10},
            },
            "cumulative_knockout": {
                "0": {"accuracy": 0.7, "correct": 7, "total": 10},
                "1": {"accuracy": 0.5, "correct": 5, "total": 10},
            },
            "reverse_cumulative_knockout": {
                "0": {"accuracy": 0.4, "correct": 4, "total": 10},
                "1": {"accuracy": 0.9, "correct": 9, "total": 10},
            },
        }
        print_intervention_summary(summary, num_model_layers=2, baseline_accuracy=0.75)
        captured = capsys.readouterr()
        assert "Per-layer Knockout" in captured.out
        assert "Cumulative Knockout" in captured.out
        assert "Reverse Cumulative Knockout" in captured.out

    def test_partial_summary(self, capsys):
        summary = {
            "per_layer_knockout": {
                "0": {"accuracy": 0.8, "correct": 8, "total": 10},
            },
        }
        print_intervention_summary(summary, num_model_layers=1, baseline_accuracy=0.5)
        captured = capsys.readouterr()
        assert "Per-layer Knockout" in captured.out
        assert "Cumulative Knockout" not in captured.out
