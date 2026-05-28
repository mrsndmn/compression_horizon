"""Launch baseline progressive-cramming jobs for the FINETUNED Llama-3.1-8B depth checkpoints.

Llama-3.1-8B companion to ``scripts/jobs/run_jobs_layer_ablation_ft.py`` (SmolLM2-1.7B).
After ``scripts/jobs/run_jobs_finetune_truncated_llama.py`` finetunes each truncated
``Meta-Llama-3.1-8B`` checkpoint (``…-firstlast{N}-ftw``), this launcher submits the
same baseline progressive-cramming run used everywhere else in the matrix, reusing
:func:`run_jobs_progressive.render_job` so output-dir names and payloads stay
byte-identical (``embedding_init_method=random0.02``, lr 0.1, cross_entropy,
num_alignment_layers=1, no low-dim -- directly comparable to ``tab:layer_ablation``).

This is the **finetuned arm only** (no un-finetuned/raw-truncated rows -- the user
chose finetuned-only for the Llama extension). The finetuned runs land in
``…-firstlast{N}-ftw_ds_pg19_1k_limit_50_lr_0.1``. The full-depth (32-layer)
``sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1`` run already exists and is
the reference row (waited on, not retried, by the watcher).

``tab:layer_ablation`` is EXTENDED with these Llama rows as a deliberate follow-up
after the evals finish; the watcher only notifies, it does not regenerate the table.

Usage:
    python scripts/jobs/run_jobs_layer_ablation_ft_llama.py --dry
    python scripts/jobs/run_jobs_layer_ablation_ft_llama.py
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
JOB_DESC_PREFIX = "CH: layer-ablation-llama-ft"

# Finetuned depth-ablated Llama-3.1-8B checkpoints (produced by
# run_jobs_finetune_truncated_llama.py; -ftw suffix).
LAYER_ABLATION_FT_CHECKPOINTS = [
    "artifacts/checkpoints/Meta-Llama-3.1-8B-firstlast1-ftw",
    "artifacts/checkpoints/Meta-Llama-3.1-8B-firstlast2-ftw",
    "artifacts/checkpoints/Meta-Llama-3.1-8B-firstlast4-ftw",
]

# Baseline progressive variant only (matches the SmolLM2 depth ablation + main matrix).
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
    for ck in LAYER_ABLATION_FT_CHECKPOINTS
]

# Full pretrained Llama-3.1-8B: reference row (waited on, not retried, by a watcher).
# Already has data from the main matrix -- no new job needed.
REFERENCE_OUT_DIR = "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1"

# Wiring for the follow-up table step (extend tab:layer_ablation with the Llama rows;
# done by hand after evals finish, not by the watcher). Llama is firstlast{1,2,4} ->
# 2/4/8 of 32 decoder layers, plus the 32-layer full reference.
TABLE_NAME = "tab:layer_ablation"
APPENDIX_MARKER = "layer-ablation-trend"
ROW_LABELS = [
    "Llama-3.1-8B, 2 layers (finetuned)",
    "Llama-3.1-8B, 4 layers (finetuned)",
    "Llama-3.1-8B, 8 layers (finetuned)",
    "Llama-3.1-8B, 32 layers (full)",
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
        print(f"\033[31mMissing finetuned checkpoint, run run_jobs_finetune_truncated_llama.py first:\033[0m {ckpt}")
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
    parser = argparse.ArgumentParser(
        description="Launch progressive cramming jobs for the finetuned Llama-3.1-8B transformer-depth ablation."
    )
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    args = parser.parse_args()

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for experiment in EXPERIMENTS:
        submit_experiment(experiment, client, extra_options, in_progress_descs, dry=args.dry)


if __name__ == "__main__":
    main()
