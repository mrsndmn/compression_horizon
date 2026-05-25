"""Launch 8-GPU causal-LM finetuning jobs for the depth-truncated Llama-3.1-8B checkpoints.

Llama-3.1-8B companion to ``scripts/jobs/run_jobs_finetune_truncated.py`` (which
does the same for SmolLM2-1.7B). For each depth-ablated ``Meta-Llama-3.1-8B``
checkpoint (first N + last N layers, N in {1,2,4}, built by
``scripts/checkpoints/make_first_last_layers_ckpt.py`` from ``unsloth/Meta-Llama-3.1-8B``)
we submit one multi-GPU finetuning job running ``scripts/finetune_causal_lm.py``
under ``scripts/jobs/multigpu.sh`` (``accelerate launch`` across all GPUs). The
finetuned checkpoint is written next to the source as ``<checkpoint>-ftw`` and is
later re-evaluated on progressive cramming by
``scripts/jobs/run_jobs_layer_ablation_ft_llama.py``.

Hyperparameters are the **same width/compression-head recipe** used by the
SmolLM2-1.7B depth retrain (lr 1e-3, wd 0.01, ``cosine_with_min_lr`` min_lr 1e-5,
warmup 500, seq 1024, bf16, 5k steps) so the Llama-3.1-8B rows are directly
comparable with the SmolLM2-1.7B rows in ``tab:layer_ablation``.

8B-specific differences from the SmolLM2 launcher (the cluster uses plain DDP --
``configs/accelerate.yaml`` is ``distributed_type: MULTI_GPU``, no sharding -- so
every GPU holds the full model + AdamW state):

  * ``per_device_train_batch_size=2`` with ``gradient_accumulation_steps=16`` keeps
    the effective batch at 256 sequences (~256k tokens/step, recipe-identical) while
    fitting an 8B-derived model in 80GB.
  * ``gradient_checkpointing`` is on, and ``torch.compile`` is OFF
    (``--no_torch_compile``) -- conservative defaults so the (much larger) 8B job
    does not OOM or hit a compile+checkpoint edge case on its first attempt. Both
    are pure memory/throughput knobs; the optimization is unchanged.

The packed fineweb-edu dataset uses a DISTINCT cache key (``Meta-Llama-3.1-8B``)
because the Llama tokenizer/vocab (128k) differs from SmolLM2's. All three jobs
share that one cache, so the watcher pre-builds it once on the login node before
submitting (otherwise three jobs would race the same ``save_to_disk``).

Batch sizing follows the usual convention:
``gradient_accumulation_steps = total_batch_size // (num_gpus * per_device_bs)``,
where ``total_batch_size`` counts *sequences*.

The submit helpers (:func:`make_client`, :func:`submit_experiment`) are factored
out so ``scripts/jobs/watch_finetune_truncated_llama.py`` can resubmit a failed job.

Usage:
    python scripts/jobs/run_jobs_finetune_truncated_llama.py --dry
    python scripts/jobs/run_jobs_finetune_truncated_llama.py
"""

import argparse
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
JOB_DESC_PREFIX = "CH: ft-truncated-llama"

# Pretrained base the truncated checkpoints were sliced from (tokenizer source for
# the shared packed-dataset prebuild).
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B"

# Depth-ablated Llama-3.1-8B checkpoints to finetune (first N + last N layers,
# N in {1,2,4} -> 2/4/8 of 32 decoder layers). Built from unsloth/Meta-Llama-3.1-8B
# so the model_short ("Meta-Llama-3.1-8B-...") matches the full-depth reference row
# already present at sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1.
FINETUNE_CHECKPOINTS = [
    "artifacts/checkpoints/Meta-Llama-3.1-8B-firstlast1",
    "artifacts/checkpoints/Meta-Llama-3.1-8B-firstlast2",
    "artifacts/checkpoints/Meta-Llama-3.1-8B-firstlast4",
]

# The Llama tokenizer/vocab differs from SmolLM2, so the packed fineweb-edu dataset
# is NOT the one used by the SmolLM2 launchers -- pin a distinct cache namespace.
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CACHE_KEY = "Meta-Llama-3.1-8B"

# Finetuning hyperparameters: the same width/compression-head recipe used by the
# SmolLM2 depth retrain (run_jobs_finetune_truncated.py) so depth ablations across
# model families share one finetune recipe. The batch/memory knobs below are sized
# for an 8B-derived model under plain DDP (see module docstring).
DEFAULTS = {
    "num_gpus": 8,
    "max_steps": 5000,
    "max_sequence_length": 1024,
    "per_device_train_batch_size": 2,
    "total_batch_size": 256,  # sequences/step -> 256 * 1024 ~= 256k tokens/step
    "learning_rate": 0.001,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine_with_min_lr",
    "lr_scheduler_kwargs": "min_lr=0.00001",
    "limit_dataset_items": 3000000,
    "dtype": "bf16",
    "gradient_checkpointing": True,
    "no_torch_compile": True,
}


def finetuned_dir(checkpoint: str) -> str:
    """Output path of the finetuned checkpoint (``-ftw``, the shared depth/width recipe)."""
    return checkpoint.rstrip("/") + "-ftw"


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
        f"--lr_scheduler_kwargs '{opts['lr_scheduler_kwargs']}'",
        f"--dtype {opts['dtype']}",
    ]
    if opts.get("gradient_checkpointing"):
        cmd_args.append("--gradient_checkpointing")
    if opts.get("no_torch_compile"):
        cmd_args.append("--no_torch_compile")
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


def prebuild_dataset_cache(opts: dict | None = None) -> None:
    """Pre-build the shared packed fineweb-edu cache on the login node (tokenizer-only, CPU).

    All three Llama finetune jobs share one packed dataset (identical tokenizer,
    dataset, seq_len and doc limit), so building it once here avoids three cluster
    jobs racing the same ``save_to_disk``. No-op if the cache already exists. Needs
    ``PYTHONPATH=./src`` (for ``compression_horizon.utils.launch``), which the watcher
    sets; run manually as ``PYTHONPATH=./src python ... --prebuild-only``.
    """
    opts = opts or DEFAULTS
    # finetune_causal_lm lives in scripts/ (parent of scripts/jobs/).
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from finetune_causal_lm import DEFAULT_CACHE_DIR, build_packed_dataset
    from transformers import AutoTokenizer

    print(f"Prebuilding packed dataset cache (cache_key={DATASET_CACHE_KEY}, base={BASE_MODEL}) ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    build_packed_dataset(
        dataset_name=DATASET_NAME,
        split="train",
        tokenizer=tok,
        seq_len=opts["max_sequence_length"],
        limit_docs=opts["limit_dataset_items"],
        cache_dir=DEFAULT_CACHE_DIR,
        cache_key=DATASET_CACHE_KEY,
        num_proc=8,
    )


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
    parser = argparse.ArgumentParser(description="Launch 8-GPU causal-LM finetuning for truncated Llama-3.1-8B checkpoints.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument("--force", action="store_true", help="Resubmit even if the -ftw checkpoint dir exists.")
    parser.add_argument("--num_gpus", type=int, default=DEFAULTS["num_gpus"])
    parser.add_argument("--max_steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument(
        "--prebuild-only",
        action="store_true",
        help="Only build the shared packed-dataset cache on the login node, then exit (no job submission).",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Restrict to specific checkpoint path(s); repeatable. Default: all three firstlast checkpoints.",
    )
    args = parser.parse_args()

    opts = dict(DEFAULTS)
    opts["num_gpus"] = args.num_gpus
    opts["max_steps"] = args.max_steps

    if args.prebuild_only:
        prebuild_dataset_cache(opts)
        return

    checkpoints = args.checkpoint or FINETUNE_CHECKPOINTS

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for checkpoint in checkpoints:
        submit_experiment(checkpoint, client, extra_options, opts, in_progress_descs, dry=args.dry, force=args.force)


if __name__ == "__main__":
    main()
