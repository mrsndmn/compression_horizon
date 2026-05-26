"""Smoke tests for ProgressiveCrammingTrainer."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from tests.trainer_helpers import TinyDataset, _collate_batch, _make_args
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.analysis.convergence import ProgressiveSampleStateMachine
from compression_horizon.train import ProgressiveCrammingTrainer
from compression_horizon.train.optimization import build_optimizer_and_scheduler
from compression_horizon.train.parametrization import PerSampleDirectParametrization

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_progressive_cramming_trainer_smoke():
    """Instantiate ProgressiveCrammingTrainer, run train() with tiny data, check no crash."""
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(
        progressive_train=True,
        progressive_min_seq_len=2,
        progressive_step=1,
        progressive_max_stages=2,
        max_optimization_steps_per_sample=1,
        max_optimization_steps_per_token=1,
        number_of_mem_tokens=1,
        logging_dir=None,
    )
    dataset = TinyDataset(num_samples=2, seq_len=8, vocab_size=16)

    trainer = ProgressiveCrammingTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=_collate_batch,
    )
    out = trainer.train()
    assert out is None or isinstance(out, str)


def _blank_trainer(args):
    """A ProgressiveCrammingTrainer without running __init__ (we only exercise pure methods)."""
    trainer = ProgressiveCrammingTrainer.__new__(ProgressiveCrammingTrainer)
    trainer.args = args
    trainer.writer = None
    return trainer


def test_geometric_snapshot_restore_roundtrip():
    """_snapshot/_restore_optimization_state restore embedding + Adam moments + LR position exactly (CPU)."""
    args = _make_args(lr_scheduler_type="cosine", learning_rate=0.1, max_optimization_steps_per_sample=20)
    trainer = _blank_trainer(args)

    param = PerSampleDirectParametrization(init_embedding=torch.zeros(1, 1, 4), device=torch.device("cpu"))
    opt, sched = build_optimizer_and_scheduler(args, [param.parameters[0]], num_training_steps=20)
    batch_ctx = SimpleNamespace(
        parametrization=param,
        per_sample_optimizers=[opt],
        per_sample_schedulers=[sched],
        shared_optimizer=None,
        shared_scheduler=None,
    )

    def _step():
        opt.zero_grad()
        (param.parameters[0] ** 2).sum().backward()
        opt.step()
        sched.step()

    for _ in range(3):  # build non-trivial Adam state + advance the LR schedule
        _step()

    snapshot = trainer._snapshot_optimization_state(batch_ctx)
    saved_emb = param.parameters[0].detach().clone()
    saved_exp_avg = opt.state[param.parameters[0]]["exp_avg"].detach().clone()
    saved_last_epoch, saved_lr = sched.last_epoch, sched.get_last_lr()[0]

    for _ in range(5):  # drift the live state away from the snapshot
        _step()
    with torch.no_grad():
        param.parameters[0].add_(123.0)
    assert not torch.allclose(param.parameters[0], saved_emb)
    assert sched.last_epoch != saved_last_epoch

    trainer._restore_optimization_state(batch_ctx, snapshot)

    assert torch.allclose(param.parameters[0], saved_emb), "embedding not restored"
    assert torch.allclose(opt.state[param.parameters[0]]["exp_avg"], saved_exp_avg), "Adam exp_avg not restored"
    assert sched.last_epoch == saved_last_epoch, "scheduler step not restored"
    assert abs(sched.get_last_lr()[0] - saved_lr) < 1e-12, "LR not restored"
    assert torch.allclose(snapshot["params"][0], saved_emb), "snapshot mutated by later live steps"


@pytest.mark.parametrize("backoff", ["bisect", "linear"])
@pytest.mark.parametrize("horizon", [2, 3, 5, 7, 13, 31, 50, 100, 200, 255])
def test_geometric_search_finds_horizon_and_restores_on_backoff(horizon, backoff):
    """Both back-off strategies pin the exact horizon and warm-restore from the converged anchor (CPU)."""
    args = _make_args(max_optimization_steps_per_token=1, max_optimization_steps_per_sample=10**9)
    trainer = _blank_trainer(args)
    max_len = 256

    ctx = SimpleNamespace(
        geometric_growth=True, geometric_backoff=backoff, threshold=1.0, min_len=2, step_increment=1, max_stages_cap=0
    )
    batch_ctx = SimpleNamespace(batch_size=1, max_len=max_len)
    state = ProgressiveSampleStateMachine(1, threshold=1.0)

    all_lens: list[int] = []
    restore_calls: list = []

    def fake_run_stage_loop(_bc, stage_ctx, _ctx, _state, _max_steps):
        all_lens.append(stage_ctx.seq_len)
        return torch.tensor(0.0), torch.tensor([1.0 if stage_ctx.seq_len <= horizon else 0.0])

    trainer._setup_stage = lambda _bc, seq_len, stage_index: SimpleNamespace(seq_len=seq_len, stage_index=stage_index)
    trainer._run_stage_loop = fake_run_stage_loop
    trainer._collect_stage_rows = lambda *a, **k: []
    trainer._snapshot_optimization_state = lambda _bc: {"sentinel": True}
    trainer._restore_optimization_state = lambda _bc, snap: restore_calls.append(snap)

    trainer._run_geometric_stages(batch_ctx, ctx, state, sample_id_counter=0)

    expected = min(horizon, max_len)
    converged = [length for length in all_lens if length <= horizon]
    assert max(converged) == expected, f"horizon not pinned: got {max(converged)}, want {expected}"

    fail_idxs = [i for i, length in enumerate(all_lens) if length > horizon]
    if backoff == "bisect":
        # Every probe after the first failure is a bisection probe and must be warm-restored first.
        n_bisect_probes = (len(all_lens) - (fail_idxs[0] + 1)) if fail_idxs else 0
        assert len(restore_calls) == n_bisect_probes, f"restores {len(restore_calls)} != bisect probes {n_bisect_probes}"
        # The +1 linear walk should also strictly contain the horizon between consecutive probes.
    else:
        # Linear back-off restores the converged checkpoint exactly once (before the +1 walk),
        # then grows one token per stage; consecutive probed lengths differ by 1.
        assert len(restore_calls) == (1 if fail_idxs else 0), f"linear should restore once, got {len(restore_calls)}"
        walk = all_lens[fail_idxs[0] + 1 :] if fail_idxs else []
        assert all(b - a == 1 for a, b in zip(walk, walk[1:])), f"linear walk not +1 contiguous: {walk}"
