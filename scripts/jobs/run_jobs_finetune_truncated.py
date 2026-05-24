"""Launch 8-GPU causal-LM finetuning jobs for the depth-truncated checkpoints.

For each depth-ablated SmolLM2-1.7B checkpoint (first N + last N layers, built by
``scripts/checkpoints/make_first_last_layers_ckpt.py``) we submit one multi-GPU
finetuning job that runs ``scripts/finetune_causal_lm.py`` under
``scripts/jobs/multigpu.sh`` (``accelerate launch`` across all GPUs of the node).
The finetuned checkpoint is written next to the source as
``<checkpoint>-ft`` and is later re-evaluated on progressive cramming by
``scripts/jobs/run_jobs_layer_ablation_ft.py``.

Batch sizing follows the compression-head launcher convention:
``gradient_accumulation_steps = total_batch_size // (num_gpus * per_device_bs)``,
where ``total_batch_size`` counts *sequences*. With the defaults
(total 256 seqs x 1024 tokens) each optimizer step sees ~256k tokens.

The submit helpers (:func:`make_client`, :func:`submit_experiment`) are factored
out so a watcher can resubmit a failed job.

Usage:
    python scripts/jobs/run_jobs_finetune_truncated.py --dry
    python scripts/jobs/run_jobs_finetune_truncated.py
"""

import argparse
import os

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
JOB_DESC_PREFIX = "CH: ft-truncated"

# Depth-ablated checkpoints to finetune (first N + last N layers, N in {1,2,4,8}).
FINETUNE_CHECKPOINTS = [
    "artifacts/checkpoints/SmolLM2-1.7B-firstlast1",
    "artifacts/checkpoints/SmolLM2-1.7B-firstlast2",
    "artifacts/checkpoints/SmolLM2-1.7B-firstlast4",
    "artifacts/checkpoints/SmolLM2-1.7B-firstlast8",
]

# All truncated checkpoints share the SmolLM2-1.7B tokenizer/vocab, so the packed
# fineweb-edu dataset is identical across jobs -- pin one cache namespace.
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CACHE_KEY = "SmolLM2-1.7B"

# Finetuning hyperparameters (see .omc/specs/deep-interview-finetune-truncated-layers.md).
DEFAULTS = {
    "num_gpus": 8,
    "max_steps": 10000,
    "max_sequence_length": 1024,
    "per_device_train_batch_size": 8,
    "total_batch_size": 256,  # sequences/step -> 256 * 1024 ~= 256k tokens/step
    "learning_rate": 3e-4,
    "warmup_steps": 500,
    "weight_decay": 0.1,
    "lr_scheduler_type": "cosine",
    "limit_dataset_items": 400000,
    "dtype": "bf16",
}


def finetuned_dir(checkpoint: str) -> str:
    """Output path of the finetuned checkpoint for a given source checkpoint."""
    return checkpoint.rstrip("/") + "-ft"


def job_desc_for(model_short: str) -> str:
    return f"{JOB_DESC_PREFIX} {model_short} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"


def build_payload(checkpoint: str, extra_options: dict, opts: dict, workdir: str | None = None):
    """Return ``(payload, model_short, out_dir)`` for one finetuning job."""
    workdir = workdir or os.getcwd()
    model_short = os.path.basename(checkpoint.rstrip("/"))
    out_dir = finetuned_dir(checkpoint)

    num_gpus = opts["num_gpus"]
    per_device = opts["per_device_train_batch_size"]
    total = opts["total_batch_size"]
    denom = num_gpus * per_device
    if total % denom != 0:
        raise ValueError(f"total_batch_size ({total}) must be divisible by num_gpus*per_device ({denom}).")
    grad_accum = total // denom

    cmd_args = [
        f"--model_checkpoint {checkpoint}",
        f"--output_dir {out_dir}",
        f"--dataset_name {DATASET_NAME}",
        f"--dataset_cache_key {DATASET_CACHE_KEY}",
        f"--max_sequence_length {opts['max_sequence_length']}",
        f"--limit_dataset_items {opts['limit_dataset_items']}",
        f"--max_steps {opts['max_steps']}",
        f"--per_device_train_batch_size {per_device}",
        f"--gradient_accumulation_steps {grad_accum}",
        f"--learning_rate {opts['learning_rate']}",
        f"--warmup_steps {opts['warmup_steps']}",
        f"--weight_decay {opts['weight_decay']}",
        f"--lr_scheduler_type {opts['lr_scheduler_type']}",
        f"--dtype {opts['dtype']}",
    ]
    script = f"bash {workdir}/scripts/jobs/multigpu.sh scripts/finetune_causal_lm.py  {' '.join(cmd_args)}"
    payload = {
        "script": script,
        "job_desc": job_desc_for(model_short),
        "env_variables": {
            "PYTHONPATH": "./src",
            "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
        },
        "instance_type": f"a100.{num_gpus}gpu",
        "region": extra_options["region"],
        "type": "binary_exp",
        "shm_size_class": "medium",
        "base_image": BASE_IMAGE,
        "n_workers": 1,
        "processes_per_worker": 1,
    }
    return payload, model_short, out_dir


def make_client():
    return training_job_api_from_profile("default")


def submit_experiment(checkpoint, client, extra_options, opts, in_progress_descs=None, dry=False, force=False):
    """Submit one finetuning job; return the ``run_job`` result dict or ``None`` if skipped."""
    payload, model_short, out_dir = build_payload(checkpoint, extra_options, opts)

    if not os.path.isdir(checkpoint):
        print(f"\033[31mMissing checkpoint, run make_first_last_layers_ckpt.py first:\033[0m {checkpoint}")
        return None

    if not force and os.path.isdir(out_dir):
        print("Finetuned checkpoint", out_dir, "exists, skip.")
        return None

    if in_progress_descs and payload["job_desc"] in in_progress_descs:
        print(f"\033[33mSkipping: job already in queue:\033[0m {payload['job_desc']}")
        return None

    print(f"\033[32m Would launch:\033[0m {payload['job_desc']}")
    print(f"\033[90m     Command: {payload['script']}\033[0m")
    print(f"\033[90m     Output dir: {out_dir}\033[0m")

    if dry:
        return None

    result = client.run_job(payload=payload)
    print(out_dir, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Launch 8-GPU causal-LM finetuning for truncated checkpoints.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument("--force", action="store_true", help="Resubmit even if the -ft checkpoint dir exists.")
    parser.add_argument("--num_gpus", type=int, default=DEFAULTS["num_gpus"])
    parser.add_argument("--max_steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Restrict to specific checkpoint path(s); repeatable. Default: all four firstlast checkpoints.",
    )
    args = parser.parse_args()

    opts = dict(DEFAULTS)
    opts["num_gpus"] = args.num_gpus
    opts["max_steps"] = args.max_steps

    checkpoints = args.checkpoint or FINETUNE_CHECKPOINTS

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for checkpoint in checkpoints:
        submit_experiment(checkpoint, client, extra_options, opts, in_progress_descs, dry=args.dry, force=args.force)


if __name__ == "__main__":
    main()
