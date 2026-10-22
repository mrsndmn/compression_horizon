import pytest
import torch
from scripts.audit_sps_objective_gradients import gradient_decomposition


def test_decomposition_measures_opposing_correct_and_wrong_gradients():
    memory = torch.tensor([0.0], requires_grad=True)
    scores = torch.stack((memory[0] + 0.1, -memory[0] - 0.1))
    logits = torch.stack((scores, torch.zeros_like(scores)), -1)[None]
    report = gradient_decomposition(logits, memory, torch.tensor([[0, 0]]))
    assert report["wrong_tokens"] == 1
    assert report["correct_wrong_gradient_cosine"] == pytest.approx(-1.0)
    assert report["gradient_additivity_error"] < 1e-6
    assert report["cancellation_ratio"] < 0.1
    assert report["hinge_descent_worst_margin_cosine"] > 0
