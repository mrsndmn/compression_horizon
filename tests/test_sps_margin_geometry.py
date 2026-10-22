import pytest
import torch
from scripts.analyze_sps_margin_geometry import margin_statistics, target_margins


def test_projected_control_separates_independent_and_conflicting_constraints():
    margins = torch.tensor([1.0, -1.0], dtype=torch.float64)
    independent = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    stat, direction = margin_statistics(margins, independent)
    assert stat["new_gradient_free_ratio"] == pytest.approx(1.0)
    assert (independent[:-1] @ direction).abs().max() < 1e-10
    assert independent[-1] @ direction > 0
    conflicting = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    stat, direction = margin_statistics(margins, conflicting)
    assert stat["new_gradient_free_ratio"] < 1e-10
    assert stat["negative_cosine_fraction"] == 1.0
    assert stat["required_linear_step_l2"] is None


def test_target_margin_uses_strongest_non_target_competitor():
    logits = torch.tensor([[[5.0, 4.0, 2.0], [3.0, 9.0, 6.0]]], requires_grad=True)
    ids = torch.tensor([[0, 2]])
    margin = target_margins(logits, ids)
    torch.testing.assert_close(margin, torch.tensor([1.0, -3.0]))
    margin.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[1.0, -1.0, 0.0], [0.0, -1.0, 1.0]]]))


@pytest.mark.parametrize("sps", [False, True])
def test_geometry_executes_native_forward_backward_and_local_trials(sps):
    from scripts.analyze_sps_margin_geometry import TRUST_RADII, geometry
    from tests.test_sps_causal_lm import tiny_args

    from compression_horizon.models.sps_causal_lm import SPSCausalLM

    model = SPSCausalLM(tiny_args(), sps=sps).eval().requires_grad_(False)
    row = dict(embedding=torch.randn(1, 16).tolist(), final_convergence=1.0, convergence_threshold=1.0)
    result = geometry(model, row, torch.tensor([[1, 2, 3]]))
    assert len(result["margins"]) == 3
    assert len(result["local_trials"]) == len(TRUST_RADII)
    assert result["max_protected_direction_derivative"] < 1e-5
    assert 0 <= result["new_gradient_free_ratio"] <= 1 + 1e-6
