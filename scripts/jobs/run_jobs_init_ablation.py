"""Launch baseline progressive-cramming jobs for the initialization ablation.

For each initialization-ablated SmolLM2-1.7B checkpoint (random transformer
layers / random LM head / random input embeddings, created by
``scripts/checkpoints/make_random_init_ckpt.py``) we submit one baseline
progressive run, reusing :func:`run_jobs_progressive.render_job` so output dir
names and payloads match the main experiment matrix.

The pretrained full model run ``sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1``
is the reference row (same dataset / sample count / lr produced here).

Exposes a uniform interface (EXPERIMENTS, REFERENCE_OUT_DIR, TABLE_NAME,
APPENDIX_MARKER, ROW_LABELS, render_job, job_desc_for, make_client,
submit_experiment) consumed by ``scripts/jobs/watch_ablation.py``.

Usage:
    python scripts/jobs/run_jobs_init_ablation.py --dry
    python scripts/jobs/run_jobs_init_ablation.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile  # noqa: E402
from run_jobs_progressive import render_job  # noqa: E402  (re-exported for the watcher)

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
JOB_DESC_PREFIX = "CH: init-ablation"

# Initialization-ablated checkpoints (one component randomized each).
INIT_ABLATION_CHECKPOINTS = [
    "artifacts/checkpoints/SmolLM2-1.7B-randinit-layers",
    "artifacts/checkpoints/SmolLM2-1.7B-randinit-lmhead",
    "artifacts/checkpoints/SmolLM2-1.7B-randinit-embeddings",
]

# Baseline progressive variant only (matches the SmolLM2-1.7B baseline).
EXPERIMENTS = [
    {
        "model_checkpoint": ck,
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
    }
    for ck in INIT_ABLATION_CHECKPOINTS
]

# Full pretrained model: reference row (waited on, not retried, by the watcher).
REFERENCE_OUT_DIR = "artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1"

# Wiring for the watcher's table-regeneration + paper-text step.
TABLE_NAME = "tab:init_ablation"
APPENDIX_MARKER = "init-ablation-trend"
ROW_LABELS = [
    # Must match the table's names_mapping labels exactly (used to parse the
    # generated .tex for the appendix trend sentence).
    "Pretrained",
    "Random transformer layers",
    "Random LM head",
    "Random input embeddings",
]


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

    ckpt = experiment["model_checkpoint"]
    if not os.path.isdir(ckpt):
        print(f"\033[31mMissing checkpoint, run make_random_init_ckpt.py first:\033[0m {ckpt}")
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
    parser = argparse.ArgumentParser(description="Launch progressive cramming jobs for the initialization ablation.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    args = parser.parse_args()

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for experiment in EXPERIMENTS:
        submit_experiment(experiment, client, extra_options, in_progress_descs, dry=args.dry)


if __name__ == "__main__":
    main()
