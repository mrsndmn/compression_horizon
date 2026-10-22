"""Known constraint systems and end-to-end rescue with actual autograd."""

import pytest
import torch
from scripts.analyze_sps_margin_rescue import minimum_norm_step, rescue_memory


def test_qp_respects_slack_and_finds_known_minimum_norm():
    margins = torch.tensor([-1.0, 0.5], dtype=torch.float64)
    jac = torch.eye(2, dtype=torch.float64)
    step, info = minimum_norm_step(margins, jac)
    assert info["feasible"]
    torch.testing.assert_close(step, torch.tensor([1.05, 0.0], dtype=torch.float64), atol=1e-5, rtol=1e-5)
    margins = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    jac = torch.tensor([[1.0, 0.0], [-1.0, 1.0]], dtype=torch.float64)
    step, info = minimum_norm_step(margins, jac)
    assert info["feasible"]
    assert (margins + jac @ step).min() >= 0.05 - 1e-4
    torch.testing.assert_close(step, torch.tensor([1.05, 2.1], dtype=torch.float64), atol=1e-4, rtol=1e-4)


def test_qp_reports_inconsistent_linear_constraints():
    step, info = minimum_norm_step(torch.tensor([-1.0, -1.0]), torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    assert not info["feasible"]


@pytest.mark.parametrize("method", ["ce", "hinge", "trust"])
def test_rescue_tracks_best_memory_and_honors_gradient_budget(method):
    def forward(memory):
        scores = memory.flatten()[:2]
        return torch.stack((scores, torch.zeros_like(scores)), dim=-1)[None]

    ids = torch.zeros(1, 2, dtype=torch.long)
    initial = torch.tensor([[[-0.3, -0.2, 1.0]]])
    result = rescue_memory(forward, ids, initial, dict(method=method, lr=0.3), budget=100)
    assert 0 < result["gradient_evaluations"] <= 100
    assert result["best_min_margin"] > 0.049
    assert not torch.equal(initial, result["memory"])
    assert torch.equal(initial, torch.tensor([[[-0.3, -0.2, 1.0]]]))
