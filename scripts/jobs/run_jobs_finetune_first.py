"""Launch 8-GPU causal-LM finetuning jobs for the FIRST-ONLY depth-ablated checkpoints.

Companion to the first-N+last-N depth ablations (run_jobs_finetune_truncated.py /
_llama.py). This one keeps only the **first N** decoder layers (N in {1,2,4,8} ->
N total layers, built by ``make_first_last_layers_ckpt.py --keep_mode first``) for
BOTH model families:

  * SmolLM2-1.7B-first{1,2,4,8}      (of 24 layers)
  * Meta-Llama-3.1-8B-first{1,2,4,8} (of 32 layers)

Each ``…-first{N}`` checkpoint is finetuned with the shared width/compression-head
``-ftw`` recipe (lr 1e-3, wd 0.01, cosine_with_min_lr min_lr 1e-5, warmup 500, seq
1024, bf16, 5k steps, ~256k tokens/step) so first-only rows are directly comparable
with the firstlast rows in ``tab:layer_ablation``. The finetuned checkpoint is
written as ``<checkpoint>-ftw`` and re-evaluated by run_jobs_layer_ablation_first_ft.py.

Llama exception (scaled recovery): the 8B truncations recover a much smaller fraction
of their cramming ceiling than SmolLM2 under the shared 5k-step/3M-doc recipe, so the
Llama specs override the budget to 15k steps / 9M docs at lr 3e-4 / warmup 1000 (3x
steps + 3x data), still at the same 256-seq (~256k-token) global batch. SmolLM2 keeps
the 5k/3M recipe. The previous 5k/3M Llama -ftw checkpoints + their PG19 evals were
archived under ``artifacts/**/_archive_ftw_5k3M/`` so the rerun gets a clean ``-ftw``.

Per-model knobs (one ``FINETUNE_SPECS`` entry per checkpoint):
  * cache_key -- the packed fineweb-edu dataset differs by tokenizer (SmolLM2 vs
    Llama 128k vocab); the packed-cache hash also includes limit_docs, so the scaled
    Llama 9M cache is a fresh dir (the 3M caches are untouched).
  * The 8B-derived Llama models run under plain DDP (configs/accelerate.yaml is
    MULTI_GPU, no sharding) on a100.8gpu, so they use per_device 2 x grad_accum 16
    (=256 seqs/step, global batch unchanged), gradient_checkpointing, and no
    torch.compile -- conservative so the larger jobs don't OOM. SmolLM2-1.7B keeps the
    standard per_device 8 on a100.4gpu.

The submit helpers are factored out so watch_finetune_first.py can resubmit a
failed job. Both packed caches already exist; ``--prebuild-only`` rebuilds any
missing one on the login node (no-op otherwise) to avoid jobs racing save_to_disk.

Usage:
    python scripts/jobs/run_jobs_finetune_first.py --dry
    python scripts/jobs/run_jobs_finetune_first.py
"""

import argparse
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
JOB_DESC_PREFIX = "CH: ft-first"

DATASET_NAME = "HuggingFaceFW/fineweb-edu"

# Pretrained base per cache key (tokenizer source for the shared packed-dataset prebuild).
CACHE_BASE = {
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
    "Meta-Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B",
    "SmolLM3-3B": "HuggingFaceTB/SmolLM3-3B",
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
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
    # Scaled recovery budget (Llama only). The 8B truncations recover a much smaller
    # fraction of their cramming ceiling than SmolLM2 under the shared 5k-step/3M-doc
    # recipe, so we give them 3x the steps and 3x the data and run on a100.8gpu
    # (per_device 2 x grad_accum 16) while KEEPING the 256-seq (~256k-token) global
    # batch. SmolLM2 stays on DEFAULTS (5k/3M/4gpu). Outputs still land at
    # ``<ckpt>-ftw``; the previous 5k/3M Llama -ftw checkpoints+evals were archived
    # under ``artifacts/**/_archive_ftw_5k3M/`` to free the name.
    "num_gpus": 8,
    "max_steps": 15000,
    "limit_dataset_items": 9000000,
}

# Model-family size sweep added at the standard 5k-step budget: SmolLM3-3B, Qwen3-4B,
# Qwen3-8B (all 36-layer decoders). They use the gentler lr 3e-4 / warmup 1000 recipe
# (the one the 8B Llama needed) applied uniformly so the larger members don't hit the
# 1e-3 warmup spike, on a100.4gpu (per_device 4 x grad_accum 16 = 256-seq global batch,
# gradient_checkpointing, no torch.compile -- conservative for the bigger-vocab models and
# keeps them OFF the 8-GPU slot so they run parallel to the Llama rescale). Steps/data come
# from DEFAULTS (5k / 3M docs); each family's tokenizer gets its own packed cache.
_NEW_5K = {
    "per_device_train_batch_size": 4,
    "gradient_checkpointing": True,
    "no_torch_compile": True,
    "learning_rate": 0.0003,
    "warmup_steps": 1000,
    "num_gpus": 4,
}
_SMOL3 = {**_NEW_5K, "cache_key": "SmolLM3-3B"}
_QWEN3_4B = {**_NEW_5K, "cache_key": "Qwen3-4B"}
_QWEN3_8B = {**_NEW_5K, "cache_key": "Qwen3-8B"}

# First-only depth-ablated checkpoints to finetune (keep first N layers; N total).
FINETUNE_SPECS = (
    [{"checkpoint": f"artifacts/checkpoints/SmolLM2-1.7B-first{n}", **_SMOL} for n in (1, 2, 4, 8)]
    + [{"checkpoint": f"artifacts/checkpoints/Meta-Llama-3.1-8B-first{n}", **_LLAMA} for n in (1, 2, 4, 8)]
    + [{"checkpoint": f"artifacts/checkpoints/SmolLM3-3B-first{n}", **_SMOL3} for n in (1, 2, 4, 8)]
    + [{"checkpoint": f"artifacts/checkpoints/Qwen3-4B-first{n}", **_QWEN3_4B} for n in (1, 2, 4, 8)]
    + [{"checkpoint": f"artifacts/checkpoints/Qwen3-8B-first{n}", **_QWEN3_8B} for n in (1, 2, 4, 8)]
)

# Shared -ftw recipe; per-checkpoint specs override cache_key + the memory knobs.
DEFAULTS = {
    "num_gpus": 4,  # a100.4gpu; grad_accum auto-scales to hold the 256-seq global batch
    "max_steps": 5000,
    "max_sequence_length": 1024,
    "total_batch_size": 256,  # sequences/step -> 256 * 1024 ~= 256k tokens/step
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
        "region": extra_options["region"],
        "type": "binary_exp",
        "shm_size_class": "medium",
        "base_image": BASE_IMAGE,
        "n_workers": 1,
        "processes_per_worker": 1,
    }
    return payload, model_short, out_dir


def prebuild_dataset_caches(specs: list[dict] | None = None, num_proc: int = 8) -> None:
    """Pre-build each DISTINCT packed fineweb-edu cache used by ``specs`` (login node, CPU).

    Distinct = ``(cache_key, limit_docs, seq_len)``. The packed-cache hash includes
    ``limit_docs``, so SmolLM2 (3M docs) and the scaled Llama recovery (9M docs) resolve to
    separate ``packed_<hash>`` dirs and never collide. No-op for caches that already exist.
    Needs ``PYTHONPATH=./src`` (set by the watcher; pass it manually for ``--prebuild-only``).
    """
    specs = specs if specs is not None else FINETUNE_SPECS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from finetune_causal_lm import DEFAULT_CACHE_DIR, build_packed_dataset
    from transformers import AutoTokenizer

    seen: set[tuple] = set()
    for spec in specs:
        o = opts_for(spec)
        key = (o["cache_key"], o["limit_dataset_items"], o["max_sequence_length"])
        if key in seen:
            continue
        seen.add(key)
        base = CACHE_BASE[o["cache_key"]]
        print(
            f"Prebuilding packed dataset cache (cache_key={o['cache_key']}, base={base}, "
            f"limit_docs={o['limit_dataset_items']}, seq_len={o['max_sequence_length']}) ..."
        )
        tok = AutoTokenizer.from_pretrained(base)
        build_packed_dataset(
            dataset_name=DATASET_NAME,
            split="train",
            tokenizer=tok,
            seq_len=o["max_sequence_length"],
            limit_docs=o["limit_dataset_items"],
            cache_dir=DEFAULT_CACHE_DIR,
            cache_key=o["cache_key"],
            num_proc=num_proc,
        )


def make_client():
    return training_job_api_from_profile("default")


def submit_experiment(spec, client, extra_options, num_gpus, max_steps, in_progress_descs=None, dry=False, force=False):
    """Submit one finetuning job; return the ``run_job`` result dict or ``None`` if skipped."""
    payload, model_short, out_dir = build_payload(spec, extra_options, num_gpus, max_steps)

    checkpoint = spec["checkpoint"]
    if not os.path.isdir(checkpoint):
        print(f"\033[31mMissing checkpoint, run make_first_last_layers_ckpt.py --keep_mode first:\033[0m {checkpoint}")
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
    parser = argparse.ArgumentParser(description="Launch 8-GPU causal-LM finetuning for first-only truncated checkpoints.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument("--force", action="store_true", help="Resubmit even if the -ftw checkpoint dir exists.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Override num_gpus for ALL jobs (default: per-spec, e.g. Llama=8, else DEFAULTS=4).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max_steps for ALL jobs (default: per-spec, e.g. Llama=15000, else DEFAULTS=5000).",
    )
    parser.add_argument(
        "--prebuild-only",
        action="store_true",
        help="Only build the shared packed-dataset caches on the login node, then exit (no job submission).",
    )
    parser.add_argument(
        "--prebuild-num-proc",
        type=int,
        default=8,
        help="Worker processes for the packed-dataset tokenize/pack map during --prebuild-only (default 8).",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Restrict to specific checkpoint path(s); repeatable. Default: all eight first-only checkpoints.",
    )
    args = parser.parse_args()

    specs = FINETUNE_SPECS
    if args.checkpoint:
        wanted = set(args.checkpoint)
        specs = [s for s in FINETUNE_SPECS if s["checkpoint"] in wanted]

    if args.prebuild_only:
        prebuild_dataset_caches(specs, num_proc=args.prebuild_num_proc)
        return

    client, extra_options = make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for spec in specs:
        submit_experiment(
            spec, client, extra_options, args.num_gpus, args.max_steps, in_progress_descs, dry=args.dry, force=args.force
        )


if __name__ == "__main__":
    main()
