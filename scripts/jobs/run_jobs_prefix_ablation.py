"""Launch progressive-cramming jobs for the fixed-prefix ablation.

For each prefix length ``P`` we submit one SmolLM2-1.7B progressive run in which the model attends to
a fixed, uncompressed prefix of ``P`` real tokens (visible context, never crammed) and progressively
crams only the continuation that follows. We reuse :func:`run_jobs_progressive.render_job` so the
command construction, output-directory names, and cluster payload stay byte-identical to the main
experiment matrix; ``render_job`` appends ``--progressive_prefix_len P`` and a ``_prefix_{P}`` suffix
when the experiment carries a ``progressive_prefix_len`` key.

The no-prefix (P=0) reference run is the existing
``sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1`` job -- same dataset, sample count, and learning
rate produced by ``render_job`` here, so the resulting rows are directly comparable.

This launcher exposes the attribute surface consumed by the generic watcher
(``scripts/jobs/watch_ablation.py``): EXPERIMENTS, REFERENCE_OUT_DIR, TABLE_NAME, APPENDIX_MARKER,
ROW_LABELS, render_job, job_desc_for, make_client, submit_experiment.

Usage:
    python scripts/jobs/run_jobs_prefix_ablation.py --dry
    python scripts/jobs/run_jobs_prefix_ablation.py
    python scripts/jobs/watch_ablation.py --launcher run_jobs_prefix_ablation --plan
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

MODEL_CHECKPOINT = "HuggingFaceTB/SmolLM2-1.7B"

# Prefix lengths to sweep. 0 (no prefix) is the reference run, handled separately below.
PREFIX_LENGTHS = [128, 256, 512, 1024]

# Baseline progressive recipe (cross_entropy, single alignment layer, no low-dim projection) --
# identical to the SmolLM2-1.7B baseline in run_jobs_progressive -- plus the fixed prefix length.
PREFIX_ABLATION_EXPERIMENTS = [
    {
        "model_checkpoint": MODEL_CHECKPOINT,
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
        "progressive_prefix_len": prefix_len,
    }
    for prefix_len in PREFIX_LENGTHS
]

# Watcher contract -----------------------------------------------------------
EXPERIMENTS = PREFIX_ABLATION_EXPERIMENTS
TABLE_NAME = "tab:prefix_ablation"
APPENDIX_MARKER = "prefix-ablation-trend"
# Row labels must match the non-reference entries of the table's names_mapping
# ("No prefix,P=128,P=256,P=512,P=1024") so the watcher can parse the per-row
# compressed-token counts for the appendix trend sentence.
ROW_LABELS = [f"P={p}" for p in PREFIX_LENGTHS]

# No-prefix (P=0) reference run: not owned by this launcher, but the table depends on it. The watcher
# waits on it (without retrying).
REFERENCE_OUT_DIR = "artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1"


def job_desc_for(exp_suffix: str) -> str:
    """Cluster job description (also used to rediscover a job by name later)."""
    return f"CH: prefix-ablation {exp_suffix} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"


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

    ``force=True`` bypasses the "output dir exists, skip" guard (used by the watcher after it removes
    a failed partial run directory before resubmitting). The model is a HuggingFace hub id (not a
    local checkpoint), so there is no checkpoint-directory existence check here.
    """
    payload, exp_suffix, out_dir_name = build_payload(experiment, extra_options)

    if not force and os.path.exists(out_dir_name):
        print("Experiment", out_dir_name, "exists, skip.")
        return None

    if in_progress_descs and payload["job_desc"] in in_progress_descs:
        print(f"\033[33mSkipping: job already in queue:\033[0m {payload['job_desc']}")
        return None

    print(f"\033[32m Would launch:\033[0m {payload['job_desc']}")
    print(f"\033[90m     Command: {payload['script']}\033[0m")
    print(f"\033[90m     Output dir: {out_dir_name}\033[0m")

    if dry:
        return None

    result = client.run_job(payload=payload)
    print(out_dir_name, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Launch progressive cramming jobs for the fixed-prefix ablation.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    args = parser.parse_args()

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for experiment in PREFIX_ABLATION_EXPERIMENTS:
        submit_experiment(experiment, client, extra_options, in_progress_descs, dry=args.dry)


if __name__ == "__main__":
    main()
