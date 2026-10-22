"""Causal memory-edge interventions preserve all other attention edges."""

import pytest
import torch
from scripts.analyze_sps_memory_ablation import block_memory_attention
from tests.test_sps_causal_lm import tiny_args

from compression_horizon.models.sps_causal_lm import SPSCausalLM, attention_visibility


@pytest.mark.parametrize("sps", [False, True])
def test_mask_intervention_is_local_and_restored_even_on_error(sps):
    model = SPSCausalLM(tiny_args(), sps=sps).eval()
    original = attention_visibility(torch.ones(1, 4), sps=sps)
    observed = []

    def capture(module, args, kwargs):
        observed.append(kwargs["attn_block_mask"].clone())

    handles = []
    try:
        with pytest.raises(RuntimeError, match="deliberate"):
            with block_memory_attention(model, [0]):
                handles.extend(b.attn.register_forward_pre_hook(capture, with_kwargs=True) for b in model.transformer.h)
                model(input_ids=torch.tensor([[1, 2, 3, 4]]))
                raise RuntimeError("deliberate")
        expected = original.clone()
        expected[..., 1:, 0] = False
        assert torch.equal(observed[0], expected)
        assert torch.equal(observed[1], original)
        observed.clear()
        model(input_ids=torch.tensor([[1, 2, 3, 4]]))
        assert all(torch.equal(mask, original) for mask in observed)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("sps", [False, True])
def test_all_layer_ablation_removes_later_readout_memory_dependence(sps):
    torch.manual_seed(7)
    model = SPSCausalLM(tiny_args(), sps=sps).eval()
    ids = torch.tensor([[1, 2, 3]])
    memory = torch.randn(1, 1, 16, requires_grad=True)
    inputs = torch.cat((memory, model.get_input_embeddings()(ids)), dim=1)
    with block_memory_attention(model, range(2)):
        logits = model(inputs_embeds=inputs).logits
    # Later text readouts cannot depend on memory through any attention layer.
    grad = torch.autograd.grad(logits[:, 1:].square().sum(), memory, retain_graph=True)[0]
    assert torch.equal(grad, torch.zeros_like(grad))
    first_grad = torch.autograd.grad(logits[:, 0].square().sum(), memory)[0]
    if sps:
        assert torch.equal(first_grad, torch.zeros_like(first_grad))
    else:
        assert first_grad.norm() > 0  # Standard retains the memory readout residual.


def test_plot_writes_all_requested_figures(tmp_path, monkeypatch):
    import json

    import scripts.analyze_sps_memory_ablation as analysis

    sample = dict(
        sample_id=0,
        length=3,
        baseline=dict(nll=[0.1, 0.2, 0.3], correct=[True] * 3),
        interventions={f: [dict(nll=[0.1, 0.4, 0.5], correct=[True, False, True])] * 2 for f in analysis.FAMILIES},
    )
    model = dict(n_layers=2, arithmetic={a: [sample] for a in analysis.ARITHMETIC})
    (tmp_path / "probe.json").write_text(json.dumps(dict(models=dict(standard=model, sps_w0=model))))
    monkeypatch.setattr(analysis, "OUTPUT", tmp_path)
    analysis.plot()
    for name in ("layer_nll.png", "layer_accuracy.png", "position_heatmaps.png", "summary.json"):
        assert (tmp_path / name).stat().st_size > 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["standard/float32/single/nll"] == pytest.approx([0.2, 0.2])
