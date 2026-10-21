"""Behavioral guards for research conclusions drawn from saved cramming stages."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("sps_diagnostics", Path(__file__).parents[1] / "scripts/analyze_sps_cramming.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_first_failed_stage_precedes_frozen_later_rows():
    rows = [
        dict(sample_id=0, stage_seq_len=n, final_convergence=acc, convergence_threshold=1.0, steps_taken=steps)
        for n, acc, steps in [(1, 1.0, 3), (2, 1.0, 5), (3, 2 / 3, 1005), (4, 0.5, 1005)]
    ]
    record = diag.summarize_rows(rows)[0]
    assert record["horizon"] == 2
    assert record["first_failed_length"] == 3
    assert record["steps_at_first_failure"] == 1005


@pytest.mark.parametrize("sps", [False, True])
def test_scoring_matches_native_trainer_including_first_target(sps):
    from tests.test_sps_causal_lm import tiny_args

    from compression_horizon.models.sps_causal_lm import SPSCausalLM
    from compression_horizon.train.loss import next_token_cross_entropy_loss_with_prefix

    torch.manual_seed(17)
    model = SPSCausalLM(tiny_args(), sps=sps).eval().requires_grad_(False)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    memory = torch.randn(2, 1, 16, requires_grad=True)
    losses, accuracy = diag.score_memory(model, memory, ids, mask)
    embeds = torch.cat((memory, model.get_input_embeddings()(ids)), dim=1)
    logits = model(inputs_embeds=embeds, attention_mask=torch.cat((torch.ones(2, 1, dtype=torch.long), mask), dim=1)).logits
    expected = next_token_cross_entropy_loss_with_prefix(logits, ids, mask, 1)
    torch.testing.assert_close((losses * mask.sum(1)).sum() / mask.sum(), expected)
    torch.testing.assert_close(accuracy, ((logits[:, :-1].argmax(-1) == ids) * mask).sum(1) / mask.sum(1))
    losses.sum().backward()
    assert memory.grad.isfinite().all() and memory.grad.norm() > 0


def test_refinement_freezes_already_exact_memory(monkeypatch):
    # One sample is exact from the start, the other needs a scalar parameter update.
    # The first sample must stay untouched despite Adam momentum/other samples.
    seen = []

    def score(model, memory, ids, mask):
        seen.append(memory.detach().clone())
        x = memory[:, 0, 0]
        return (x - 1).square(), torch.stack((torch.ones_like(x[0]), (x[1] > 0.1).float()))

    monkeypatch.setattr(diag, "score_memory", score)
    initial = torch.tensor([[[2.0]], [[0.0]]])
    result = diag.refine(None, initial, torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1), lr=0.2, max_steps=5)
    assert result["steps"][0] == 0
    assert result["best_accuracy"] == [1.0, 1.0]
    assert all(x[0, 0, 0] == 2.0 for x in seen)
