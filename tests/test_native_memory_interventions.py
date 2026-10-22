"""Independent checks of route knockout and projected KV counterfactuals."""

import pytest
import torch
from tests.test_sps_causal_lm import tiny_args

from compression_horizon.analysis.native_memory_interventions import capture_memory_kv, patch_memory_kv, route_knockout
from compression_horizon.models.sps_causal_lm import SPSCausalLM, attention_visibility


@pytest.mark.parametrize("route", ["context", "predict"])
def test_sps_route_knockout_changes_exact_edges(route):
    model = SPSCausalLM(tiny_args(), sps=True).eval()
    observed = []
    with route_knockout(model, [0], route):
        handle = model.transformer.h[0].attn.register_forward_pre_hook(
            lambda m, args, kw: observed.append(kw["attn_block_mask"].clone()), with_kwargs=True
        )
        model(input_ids=torch.tensor([[1, 2, 3]]))
        handle.remove()
    expected = attention_visibility(torch.ones(1, 3), sps=True)
    selected = slice(2, None, 2) if route == "context" else slice(1, None, 2)
    expected[..., selected, 0] = False
    assert torch.equal(observed[0], expected)


@pytest.mark.parametrize("sps", [False, True])
@pytest.mark.parametrize("kind", ["key", "value", "both"])
def test_kv_patch_identity_cleanup_and_memory_residual(sps, kind):
    torch.manual_seed(5)
    model = SPSCausalLM(tiny_args(), sps=sps).eval()
    ids = torch.tensor([[1, 2, 3]])
    inputs = model.get_input_embeddings()(ids)
    clean = model(inputs_embeds=inputs).logits
    own = capture_memory_kv(model, inputs)
    with patch_memory_kv(model, range(2), own, kind):
        actual = model(inputs_embeds=inputs).logits
    torch.testing.assert_close(actual, clean, atol=1e-6, rtol=1e-5)
    donor = {i: {k: torch.randn_like(v) for k, v in kv.items()} for i, kv in own.items()}
    with pytest.raises(RuntimeError, match="intentional"):
        with patch_memory_kv(model, range(2), donor, kind):
            changed = model(inputs_embeds=inputs).logits
            if not sps:
                torch.testing.assert_close(changed[:, 0], clean[:, 0], atol=1e-6, rtol=1e-5)
            raise RuntimeError("intentional")
    torch.testing.assert_close(model(inputs_embeds=inputs).logits, clean)
    assert (changed[:, 1:] - clean[:, 1:]).abs().max() > 1e-5


def test_sps_first_prediction_query_is_content_independent_then_contextualizes():
    from compression_horizon.analysis.native_memory_interventions import capture_readout_queries

    model = SPSCausalLM(tiny_args(), sps=True).eval()
    a = model.get_input_embeddings()(torch.tensor([[1, 2, 3]]))
    b = model.get_input_embeddings()(torch.tensor([[1, 4, 5]]))
    qa = capture_readout_queries(model, a)
    qb = capture_readout_queries(model, b)
    torch.testing.assert_close(qa[0], qb[0], atol=0, rtol=0)
    assert (qa[1] - qb[1]).abs().max() > 1e-5
