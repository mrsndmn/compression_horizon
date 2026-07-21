"""Launch baseline progressive-cramming jobs for the FINETUNED last-only depth checkpoints.

Companion to run_jobs_layer_ablation_first_ft.py / _ft.py / _ft_llama.py. After
run_jobs_finetune_last.py finetunes each last-only checkpoint (``…-last{N}-ftw``, for
SmolLM2-1.7B and Meta-Llama-3.1-8B, N in {1,2,4,8}), this launcher submits the same
baseline progressive-cramming run used everywhere else in the matrix, reusing
:func:`run_jobs_progressive.render_job` so output-dir names and payloads stay
byte-identical (``embedding_init_method=random0.02``, lr 0.1, cross_entropy,
num_alignment_layers=1, no low-dim).

Finetuned arm only. The finetuned runs land in
``sl_4096_<model>-last{N}-ftw_ds_pg19_1k_limit_50_lr_0.1``. The full-depth reference
rows already exist for both models (24-layer SmolLM2 and 32-layer Llama) and are
waited on (not retried) by the watcher. These evals are a100.1gpu (model frozen);
only the finetunes use 4 GPUs.

``tab:layer_ablation`` is EXTENDED with these last-only rows as a deliberate follow-up
after the evals finish; the watcher only notifies, it does not regenerate the table.

Usage:
    python scripts/jobs/run_jobs_layer_ablation_last_ft.py --dry
    python scripts/jobs/run_jobs_layer_ablation_last_ft.py
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
JOB_DESC_PREFIX = "CH: layer-ablation-last-ft"

# Finetuned last-only checkpoints (produced by run_jobs_finetune_last.py; -ftw suffix).
LAYER_ABLATION_LAST_FT_CHECKPOINTS = [f"artifacts/checkpoints/SmolLM2-1.7B-last{n}-ftw" for n in (1, 2, 4, 8)] + [
    f"artifacts/checkpoints/Meta-Llama-3.1-8B-last{n}-ftw" for n in (1, 2, 4, 8)
]

# Baseline progressive variant only (matches the rest of the depth ablation + main matrix).
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
    for ck in LAYER_ABLATION_LAST_FT_CHECKPOINTS
]

# Full pretrained models: reference rows (waited on, not retried, by the watcher).
# Both already have data from the main matrix -- no new jobs needed.
REFERENCE_OUT_DIRS = [
    "artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1",
    "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1",
]

# Wiring for the follow-up table step (extend tab:layer_ablation with the last-only rows).
TABLE_NAME = "tab:layer_ablation"
APPENDIX_MARKER = "layer-ablation-trend"
ROW_LABELS = [
    "SmolLM2-1.7B, last 1 layer (finetuned)",
    "SmolLM2-1.7B, last 2 layers (finetuned)",
    "SmolLM2-1.7B, last 4 layers (finetuned)",
    "SmolLM2-1.7B, last 8 layers (finetuned)",
    "Llama-3.1-8B, last 1 layer (finetuned)",
    "Llama-3.1-8B, last 2 layers (finetuned)",
    "Llama-3.1-8B, last 4 layers (finetuned)",
    "Llama-3.1-8B, last 8 layers (finetuned)",
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
        "queue_name": "fusionbrainlab-job",
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
        print(f"\033[31mMissing finetuned checkpoint, run run_jobs_finetune_last.py first:\033[0m {ckpt}")
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
        description="Launch progressive cramming jobs for the finetuned last-only transformer-depth ablation."
    )
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    args = parser.parse_args()

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for experiment in EXPERIMENTS:
        submit_experiment(experiment, client, extra_options, in_progress_descs, dry=args.dry)


if __name__ == "__main__":
    main()
