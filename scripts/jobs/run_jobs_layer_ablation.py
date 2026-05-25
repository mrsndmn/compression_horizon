"""Launch baseline progressive-cramming jobs for the transformer-depth ablation.

For each depth-ablated SmolLM2-1.7B checkpoint (first N + last N layers, created
by ``scripts/checkpoints/make_first_last_layers_ckpt.py``) we submit one baseline
progressive run, reusing :func:`run_jobs_progressive.render_job` so the output
directory names, command construction, and cluster payload stay byte-identical
to the main experiment matrix.

The full-depth (24-layer) reference run is the existing
``sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1`` job -- same dataset, sample
count, and learning rate produced by ``render_job`` here, so the resulting rows
are directly comparable.

The submit helpers (:func:`make_client`, :func:`submit_experiment`) are factored
out so ``scripts/jobs/watch_layer_ablation.py`` can reuse them to resubmit failed
jobs.

Usage:
    python scripts/jobs/run_jobs_layer_ablation.py --dry
    python scripts/jobs/run_jobs_layer_ablation.py
"""

import argparse
import os
import sys

# Reuse the shared experiment-rendering logic and cluster helpers from the main
# progressive launcher (importing it does not trigger its __main__ block).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile  # noqa: E402
from run_jobs_progressive import render_job  # noqa: E402  (re-exported for the watcher)

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"

# Depth-ablated checkpoints: first N + last N layers, N in {1, 2, 4, 8}.
LAYER_ABLATION_CHECKPOINTS = [
    "artifacts/checkpoints/SmolLM2-1.7B-firstlast1",
    "artifacts/checkpoints/SmolLM2-1.7B-firstlast2",
    "artifacts/checkpoints/SmolLM2-1.7B-firstlast4",
    "artifacts/checkpoints/SmolLM2-1.7B-firstlast8",
]

# Baseline progressive variant only (cross_entropy, single alignment layer, no
# low-dim projection) -- matches the SmolLM2-1.7B baseline in run_jobs_progressive.
LAYER_ABLATION_EXPERIMENTS = [
    {
        "model_checkpoint": ck,
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
    }
    for ck in LAYER_ABLATION_CHECKPOINTS
]

# Full-depth (24-layer) reference run: not owned by this launcher, but the table
# depends on it. The watcher waits on it (without retrying).
REFERENCE_OUT_DIR = "artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1"


def job_desc_for(exp_suffix: str) -> str:
    """Cluster job description (also used to rediscover a job by name later)."""
    return f"CH: layer-ablation {exp_suffix} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"


def build_payload(experiment: dict, extra_options: dict, workdir: str | None = None):
    """Return ``(payload, exp_suffix, out_dir_name)`` for one experiment."""
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
    """Build the training-job API client for the default profile."""
    return training_job_api_from_profile("default")


def submit_experiment(
    experiment: dict,
    client,
    extra_options: dict,
    in_progress_descs: set[str] | None = None,
    dry: bool = False,
    force: bool = False,
):
    """Submit one experiment. Returns the ``run_job`` result dict, or ``None`` if skipped.

    ``force=True`` bypasses the "output dir exists, skip" guard (used by the
    watcher after it removes a failed partial run directory before resubmitting).
    """
    payload, exp_suffix, out_dir_name = build_payload(experiment, extra_options)

    ckpt = experiment["model_checkpoint"]
    if not os.path.isdir(ckpt):
        print(f"\033[31mMissing checkpoint, run make_first_last_layers_ckpt.py first:\033[0m {ckpt}")
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
    parser = argparse.ArgumentParser(description="Launch progressive cramming jobs for the transformer-depth ablation.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    args = parser.parse_args()

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for experiment in LAYER_ABLATION_EXPERIMENTS:
        submit_experiment(experiment, client, extra_options, in_progress_descs, dry=args.dry)


if __name__ == "__main__":
    main()
