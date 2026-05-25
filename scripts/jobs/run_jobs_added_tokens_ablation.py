"""Launch progressive-cramming jobs for the tokens-per-stage (``progressive_step``) ablation.

Progressive cramming grows the target prefix one token at a time: each time a
stage converges, the trainer appends ``--progressive_step`` (Δ, default 1) tokens
to the prefix and continues (see ``progressive_cramming.py`` line ~417,
``seq_len = min(seq_len + ctx.step_increment, ...)``). This ablation varies Δ —
the number of tokens *added to the sequence and re-compressed* each time the
current stage converges — over Δ ∈ {1, 2, 4, 8, 16, 32, 64, 128} on
SmolLM2-1.7B, holding everything else equal to the main run.

The Δ=1 run is the existing baseline
``sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1`` (already computed); it is the
reference row, so only Δ ∈ {2, 4, 8, 16, 32, 64, 128} are submitted here. For
each Δ we also set ``--progressive_min_seq_len Δ`` so the prefix is always a clean
multiple of Δ (Δ=1 reduces exactly to the baseline schedule), and add a unique
``_step_Δ`` suffix so output dirs do not collide with the baseline.

No trainer change is needed — ``--progressive_step`` already implements the
requested "add Δ tokens when the current stage converged" behaviour.

Exposes the uniform launcher interface (EXPERIMENTS, REFERENCE_OUT_DIR,
TABLE_NAME, APPENDIX_MARKER, ROW_LABELS, render_job, job_desc_for, make_client,
submit_experiment) consumed by ``scripts/jobs/watch_ablation.py``.

Usage:
    python scripts/jobs/run_jobs_added_tokens_ablation.py --dry
    python scripts/jobs/run_jobs_added_tokens_ablation.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile  # noqa: E402
from run_jobs_progressive import (  # noqa: E402
    MAX_OPTIMIZATION_STEPS_PER_SAMPLE,
    MAX_OPTIMIZATION_STEPS_PER_TOKEN,
)
from run_jobs_progressive import render_job as _base_render_job  # noqa: E402

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
JOB_DESC_PREFIX = "CH: added-tokens-ablation"

MODEL_CHECKPOINT = "HuggingFaceTB/SmolLM2-1.7B"

# Δ values to *submit*. Δ=1 is the pre-existing baseline (reference row), so it is
# intentionally omitted here.
ADDED_TOKENS_STEPS = [2, 4, 8, 16, 32, 64, 128]


def _experiment(step: int) -> dict:
    """Baseline SmolLM2-1.7B progressive variant, parameterized by the step Δ."""
    return {
        "model_checkpoint": MODEL_CHECKPOINT,
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
        "progressive_step": step,
    }


EXPERIMENTS = [_experiment(step) for step in ADDED_TOKENS_STEPS]

# Full SmolLM2-1.7B baseline (Δ=1): reference row (waited on, not retried, by the watcher).
REFERENCE_OUT_DIR = "artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1"

# Wiring for the watcher's table-regeneration + paper-text step.
TABLE_NAME = "tab:added_tokens_ablation"
APPENDIX_MARKER = "added-tokens-ablation-trend"
ROW_LABELS = [
    # Must match the table's names_mapping labels exactly (used to parse the
    # generated .tex for the appendix trend sentence). Δ=1 first = the baseline row.
    "1 token/stage",
    "2 tokens/stage",
    "4 tokens/stage",
    "8 tokens/stage",
    "16 tokens/stage",
    "32 tokens/stage",
    "64 tokens/stage",
    "128 tokens/stage",
]


def render_job(experiment):
    """Like ``run_jobs_progressive.render_job`` but injects ``--progressive_step Δ``.

    For Δ != 1 we also pass ``--progressive_min_seq_len Δ`` (so the prefix grows in
    clean multiples of Δ) and append a ``_step_Δ`` suffix to the exp name / output
    dir so the run does not collide with the Δ=1 baseline.
    """
    cmd_args, exp_suffix, out_dir_name = _base_render_job(experiment)
    step = int(experiment.get("progressive_step", 1))
    if step == 1:
        return cmd_args, exp_suffix, out_dir_name

    # A step Δ>1 makes each stage reconstruct Δ extra tokens at once -- a harder
    # per-stage optimization -- so scale the per-stage budget with Δ, capped at the
    # cumulative per-sample budget: min(Δ·1000, 10000).
    steps_per_token = min(step * MAX_OPTIMIZATION_STEPS_PER_TOKEN, MAX_OPTIMIZATION_STEPS_PER_SAMPLE)
    cmd_args = [
        f"--max_optimization_steps_per_token {steps_per_token}" if a.startswith("--max_optimization_steps_per_token ") else a
        for a in cmd_args
    ]

    # The base renderer appends ``--output_dir <dir>`` as the final arg.
    assert cmd_args[-1].startswith("--output_dir "), cmd_args[-1]
    new_suffix = f"{exp_suffix}_step_{step}"
    new_out_dir = f"artifacts/experiments_progressive/{new_suffix}"
    cmd_args = cmd_args[:-1] + [
        f"--progressive_step {step}",
        f"--progressive_min_seq_len {step}",
        f"--output_dir {new_out_dir}",
    ]
    return cmd_args, new_suffix, new_out_dir


def job_desc_for(exp_suffix: str) -> str:
    return f"{JOB_DESC_PREFIX} {exp_suffix} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"


def build_payload(experiment: dict, extra_options: dict, workdir: str | None = None):
    workdir = workdir or os.getcwd()
    cmd_args, exp_suffix, out_dir_name = render_job(experiment)
    script = f" cd {workdir} && {PYTHON_PATH} scripts/activation_distillation.py  {' '.join(cmd_args)}"
    payload = {
        "script": script,
        "job_desc": job_desc_for(exp_suffix),
        "env_variables": {
            "PYTHONPATH": "./src",
            "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
        },
        "instance_type": "a100.1gpu",
        "region": extra_options["region"],
        "type": "binary_exp",
        "shm_size_class": "medium",
        "base_image": BASE_IMAGE,
        "n_workers": 1,
        "processes_per_worker": 1,
    }
    return payload, exp_suffix, out_dir_name


def make_client():
    return training_job_api_from_profile("default")


def submit_experiment(experiment, client, extra_options, in_progress_descs=None, dry=False, force=False):
    """Submit one experiment; return the ``run_job`` result dict or ``None`` if skipped."""
    payload, exp_suffix, out_dir_name = build_payload(experiment, extra_options)

    # Only guard against missing *local* checkpoints; the baseline config uses a
    # HuggingFace hub model id, which is not a local directory.
    ckpt = experiment["model_checkpoint"]
    if ckpt.startswith(("artifacts/", "./", "/")) and not os.path.isdir(ckpt):
        print(f"\033[31mMissing checkpoint:\033[0m {ckpt}")
        return None

    if not force and os.path.exists(out_dir_name):
        print("Experiment", out_dir_name, "exists, skip.")
        return None

    if in_progress_descs and payload["job_desc"] in in_progress_descs:
        print(f"\033[33mSkipping: job already in queue:\033[0m {payload['job_desc']}")
        return None

    print(f"\033[32m Would launch:\033[0m {payload['job_desc']}")
    print(f"\033[90m     Output dir: {out_dir_name}\033[0m")

    if dry:
        return None

    result = client.run_job(payload=payload)
    print(out_dir_name, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Launch progressive cramming jobs for the tokens-per-stage ablation.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    args = parser.parse_args()

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for experiment in EXPERIMENTS:
        submit_experiment(experiment, client, extra_options, in_progress_descs, dry=args.dry)


if __name__ == "__main__":
    main()
