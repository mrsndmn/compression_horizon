"""Config-level tests for the information-gain budget-rebalancing arms.

Covers the two config surfaces the deep dive adds, without touching the training/eval scripts:

* ``MyTrainingArguments`` exposes the ``budget_rebalance_*`` fields with the documented defaults
  (``mode='none'`` keeps every existing run byte-identical).
* ``run_jobs_progressive.render_job`` turns a ``budget_rebalance_mode`` config into the right CLI
  flags + output-dir suffix (``_cm_{eps}_brl_{mode}``), and leaves margin-only / plain configs
  free of any ``brl`` flag or suffix.
"""

from __future__ import annotations

import importlib.util
import os

from compression_horizon.train.arguments import MyTrainingArguments

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_run_jobs():
    path = os.path.join(_HERE, "scripts", "jobs", "run_jobs_progressive.py")
    spec = importlib.util.spec_from_file_location("run_jobs_progressive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arguments_defaults():
    args = MyTrainingArguments(output_dir="/tmp/x")
    # Disabled by default: no behaviour change for any pre-existing experiment.
    assert args.budget_rebalance_mode == "none"
    assert args.budget_rebalance_weight == 1.0
    assert args.budget_rebalance_dual_lr == 0.05
    assert args.budget_rebalance_softcount_tau == 0.5


def test_arguments_accept_modes():
    for mode in ("cap", "dual"):
        args = MyTrainingArguments(output_dir="/tmp/x", budget_rebalance_mode=mode, convergence_margin=1.0)
        assert args.budget_rebalance_mode == mode
        assert args.convergence_margin == 1.0


def test_budget_rebalance_experiment_matrix():
    m = _load_run_jobs()
    # 2 base x 3 epsilons x 2 modes.
    assert len(m.BUDGET_REBALANCE_EXPERIMENTS) == 12
    assert {e["budget_rebalance_mode"] for e in m.BUDGET_REBALANCE_EXPERIMENTS} == {"cap", "dual"}
    assert all(e["model_checkpoint"] == "EleutherAI/pythia-1.4b" for e in m.BUDGET_REBALANCE_EXPERIMENTS)


def test_render_job_cap_flags_and_suffix():
    m = _load_run_jobs()
    exp = {
        "model_checkpoint": "EleutherAI/pythia-1.4b",
        "learning_rate": 0.5,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
        "convergence_margin": 1.0,
        "budget_rebalance_mode": "cap",
    }
    cmd_args, exp_suffix, out_dir = m.render_job(exp)
    joined = " ".join(cmd_args)
    assert "--convergence_margin 1.0" in joined
    assert "--budget_rebalance_mode cap" in joined
    assert exp_suffix.endswith("_cm_1.0_brl_cap")
    assert out_dir.endswith("_cm_1.0_brl_cap")


def test_render_job_baseline_has_no_brl():
    m = _load_run_jobs()
    exp = {
        "model_checkpoint": "EleutherAI/pythia-1.4b",
        "learning_rate": 0.5,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
        "convergence_margin": 1.0,
    }
    cmd_args, exp_suffix, _ = m.render_job(exp)
    joined = " ".join(cmd_args)
    assert "budget_rebalance" not in joined
    assert "brl" not in exp_suffix
