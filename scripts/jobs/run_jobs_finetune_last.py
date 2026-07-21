"""Launch 4-GPU causal-LM finetuning jobs for the LAST-ONLY depth-ablated checkpoints.

Companion to run_jobs_finetune_first.py (first-only) and run_jobs_finetune_truncated*.py
(first-N+last-N). This one keeps only the **last N** decoder layers (N in {1,2,4,8} ->
N total layers, built by ``make_first_last_layers_ckpt.py --keep_mode last``) for BOTH
model families:

  * SmolLM2-1.7B-last{1,2,4,8}      (of 24 layers)
  * Meta-Llama-3.1-8B-last{1,2,4,8} (of 32 layers)

Each ``…-last{N}`` checkpoint is finetuned with the shared width/compression-head
``-ftw`` recipe (lr 1e-3, wd 0.01, cosine_with_min_lr min_lr 1e-5, warmup 500, seq
1024, bf16, 5k steps) so last-only rows are directly comparable with the first-only
and firstlast rows in ``tab:layer_ablation``. The finetuned checkpoint is written as
``<checkpoint>-ftw`` and re-evaluated by run_jobs_layer_ablation_last_ft.py.

These jobs run on **a100.4gpu** (NOT 8gpu) to avoid the 8-GPU queue contention, while
keeping the GLOBAL batch identical to the rest of the matrix (256 sequences/step,
~256k tokens): ``gradient_accumulation_steps = total_batch_size // (num_gpus *
per_device_bs)`` with num_gpus=4 just doubles grad_accum vs the 8-GPU runs (per-GPU
memory, hence per_device_bs, is unchanged on the same A100).

Per-model knobs (one ``FINETUNE_SPECS`` entry per checkpoint):
  * cache_key -- the packed fineweb-edu dataset differs by tokenizer; each reuses the
    cache already built for the other ablations.
  * 8B-derived Llama runs under plain DDP, so per_device 2 + gradient_checkpointing +
    no torch.compile (conservative, avoids OOM); SmolLM2-1.7B uses per_device 8.

Usage:
    python scripts/jobs/run_jobs_finetune_last.py --dry
    python scripts/jobs/run_jobs_finetune_last.py
"""

import argparse
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
JOB_DESC_PREFIX = "CH: ft-last"

DATASET_NAME = "HuggingFaceFW/fineweb-edu"

# Pretrained base per cache key (tokenizer source for the shared packed-dataset prebuild).
CACHE_BASE = {
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
    "Meta-Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B",
}

# Per-model knob bundles merged over DEFAULTS for each checkpoint.
_SMOL = {
    "cache_key": "SmolLM2-1.7B",
    "per_device_train_batch_size": 8,
    "gradient_checkpointing": False,
    "no_torch_compile": False,
}
_LLAMA = {
    "cache_key": "Meta-Llama-3.1-8B",
    "per_device_train_batch_size": 2,
    "gradient_checkpointing": True,
    "no_torch_compile": True,
    # Gentler LR + longer warmup than SmolLM2: the shared peak LR 1e-3 spiked during
    # warmup on the 8B truncations (degrading some checkpoints), so Llama uses 3e-4
    # with a 1000-step warmup. Schedule (cosine-with-min-lr) and the 256-seq global
    # batch are unchanged.
    "learning_rate": 0.0003,
    "warmup_steps": 1000,
}

# Last-only depth-ablated checkpoints to finetune (keep last N layers; N total).
FINETUNE_SPECS = [{"checkpoint": f"artifacts/checkpoints/SmolLM2-1.7B-last{n}", **_SMOL} for n in (1, 2, 4, 8)] + [
    {"checkpoint": f"artifacts/checkpoints/Meta-Llama-3.1-8B-last{n}", **_LLAMA} for n in (1, 2, 4, 8)
]

# Shared -ftw recipe; per-checkpoint specs override cache_key + the memory knobs.
# num_gpus=4 (a100.4gpu); total_batch_size keeps the global batch at 256 seqs/step.
DEFAULTS = {
    "num_gpus": 4,
    "max_steps": 5000,
    "max_sequence_length": 1024,
    "total_batch_size": 256,  # sequences/step -> 256 * 1024 ~= 256k tokens/step (matches 8-GPU runs)
    "learning_rate": 0.001,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine_with_min_lr",
    "lr_scheduler_kwargs": "min_lr=0.00001",
    "limit_dataset_items": 3000000,
    "dtype": "bf16",
}


def finetuned_dir(checkpoint: str) -> str:
    """Output path of the finetuned checkpoint (``-ftw``, the shared depth/width recipe)."""
    return checkpoint.rstrip("/") + "-ftw"


def job_desc_for(model_short: str) -> str:
    return f"{JOB_DESC_PREFIX} {model_short} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"


def opts_for(spec: dict, num_gpus: int | None = None, max_steps: int | None = None) -> dict:
    """Merge the shared recipe (DEFAULTS) with one checkpoint's per-model knobs."""
    o = dict(DEFAULTS)
    o.update(spec)
    if num_gpus is not None:
        o["num_gpus"] = num_gpus
    if max_steps is not None:
        o["max_steps"] = max_steps
    return o


def build_payload(spec: dict, extra_options: dict, num_gpus: int, max_steps: int, workdir: str | None = None):
    """Return ``(payload, model_short, out_dir)`` for one finetuning job."""
    workdir = workdir or os.getcwd()
    opts = opts_for(spec, num_gpus=num_gpus, max_steps=max_steps)
    checkpoint = spec["checkpoint"]
    model_short = os.path.basename(checkpoint.rstrip("/"))
    out_dir = finetuned_dir(checkpoint)

    per_device = opts["per_device_train_batch_size"]
    total = opts["total_batch_size"]
    denom = opts["num_gpus"] * per_device
    if total % denom != 0:
        raise ValueError(f"total_batch_size ({total}) must be divisible by num_gpus*per_device ({denom}).")
    grad_accum = total // denom

    cmd_args = [
        f"--model_checkpoint {checkpoint}",
        f"--output_dir {out_dir}",
        f"--dataset_name {DATASET_NAME}",
        f"--dataset_cache_key {opts['cache_key']}",
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
        "instance_type": f"a100.{opts['num_gpus']}gpu",
        "queue_name": "fusionbrainlab-job",
        "region": extra_options["region"],
        "type": "binary_exp",
        "shm_size_class": "medium",
        "base_image": BASE_IMAGE,
        "n_workers": 1,
        "processes_per_worker": 1,
    }
    return payload, model_short, out_dir


def prebuild_dataset_caches(opts: dict | None = None) -> None:
    """Pre-build each distinct packed fineweb-edu cache on the login node (tokenizer-only, CPU).

    No-op for caches that already exist (the first-only/firstlast runs built them).
    Needs ``PYTHONPATH=./src`` (set by the watcher; pass it manually for ``--prebuild-only``).
    """
    opts = opts or DEFAULTS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from finetune_causal_lm import DEFAULT_CACHE_DIR, build_packed_dataset
    from transformers import AutoTokenizer

    for cache_key in sorted({s["cache_key"] for s in FINETUNE_SPECS}):
        base = CACHE_BASE[cache_key]
        print(f"Prebuilding packed dataset cache (cache_key={cache_key}, base={base}) ...")
        tok = AutoTokenizer.from_pretrained(base)
        build_packed_dataset(
            dataset_name=DATASET_NAME,
            split="train",
            tokenizer=tok,
            seq_len=opts["max_sequence_length"],
            limit_docs=opts["limit_dataset_items"],
            cache_dir=DEFAULT_CACHE_DIR,
            cache_key=cache_key,
            num_proc=8,
        )


def make_client():
    return training_job_api_from_profile("default")


def submit_experiment(spec, client, extra_options, num_gpus, max_steps, in_progress_descs=None, dry=False, force=False):
    """Submit one finetuning job; return the ``run_job`` result dict or ``None`` if skipped."""
    payload, model_short, out_dir = build_payload(spec, extra_options, num_gpus, max_steps)

    checkpoint = spec["checkpoint"]
    if not os.path.isdir(checkpoint):
        print(f"\033[31mMissing checkpoint, run make_first_last_layers_ckpt.py --keep_mode last:\033[0m {checkpoint}")
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
    parser = argparse.ArgumentParser(description="Launch 4-GPU causal-LM finetuning for last-only truncated checkpoints.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument("--force", action="store_true", help="Resubmit even if the -ftw checkpoint dir exists.")
    parser.add_argument("--num_gpus", type=int, default=DEFAULTS["num_gpus"])
    parser.add_argument("--max_steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument(
        "--prebuild-only",
        action="store_true",
        help="Only build the shared packed-dataset caches on the login node, then exit (no job submission).",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Restrict to specific checkpoint path(s); repeatable. Default: all eight last-only checkpoints.",
    )
    args = parser.parse_args()

    if args.prebuild_only:
        prebuild_dataset_caches()
        return

    specs = FINETUNE_SPECS
    if args.checkpoint:
        wanted = set(args.checkpoint)
        specs = [s for s in FINETUNE_SPECS if s["checkpoint"] in wanted]

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for spec in specs:
        submit_experiment(
            spec, client, extra_options, args.num_gpus, args.max_steps, in_progress_descs, dry=args.dry, force=args.force
        )


if __name__ == "__main__":
    main()
