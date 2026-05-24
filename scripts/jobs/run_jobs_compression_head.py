"""Launch compression-head training jobs (and their progressive-cramming evaluations).

Experiments are described as code (like ``scripts/jobs/run_jobs_progressive.py``) rather than driven
by per-run CLI overrides. There are two stages:

* ``--stage train`` (default): train each compression head on 8 GPUs, keeping the global batch at
  ``TARGET_GLOBAL_TOKENS`` tokens (gradient accumulation is derived from per-device batch size, GPU
  count, and sequence length).
* ``--stage eval``: after the training jobs finish, launch one **1-GPU** progressive-cramming job per
  produced checkpoint. Each eval seeds per-sample compression embeddings from the trained head via
  ``--embedding_init_method compression_head_forward`` and then runs progressive cramming. Eval jobs are
  skipped while their compression-head output directory does not yet exist.

The three experiment families are:
  1. simple-MLP compression head on the full SmolLM2-1.7B,
  2. Q-Former compression head (num_queries=1, num_layers=3, num_heads=8) on the full SmolLM2-1.7B,
  3. a truncated-layer ablation: the Q-Former head on the first-last{1,2,4,8} SmolLM2-1.7B checkpoints
     produced by ``scripts/checkpoints/make_first_last_layers_ckpt.py``.
"""

import argparse
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

# --- Compression-head training constants (legacy argument set, baked in here). -----------------
DATASET_NAME = "HuggingFaceFW/fineweb-edu"  # the 10BT sample is selected automatically by the loader
# Sized for exactly 10k optimizer steps in a single epoch (no data repetition):
# N = 10_000 * (per_device 4 * grad_accum 8 * num_gpus 8) = 10_000 * 256 = 2_560_000 sequences.
# Well under the ~9.67M docs in the fineweb-edu 10BT sample, so each sample is seen once.
LIMIT_DATASET_ITEMS = 2_560_000
MAX_SEQ_LEN = 1024
LEARNING_RATE = 0.001
DISTILL_ALPHA = 1.0
DISTILL_BETA = 0.0
DATALOADER_NUM_WORKERS = 4
DTYPE = "bf16"
NUM_TRAIN_EPOCHS = 1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 0
LOGGING_STEPS = 50
LR_SCHEDULER_TYPE = "cosine_with_min_lr"
LR_SCHEDULER_KWARGS = "min_lr=0.00001"
FREEZE_BASE_MODEL = False

# 8-GPU training; keep the global batch at 256k tokens. gradient_accumulation_steps is derived so that
# per_device_batch * grad_accum * num_gpus * seq_len == TARGET_GLOBAL_TOKENS.
# At seq_len=1024, per_device=4 keeps per-device memory equal to the original seq_len=64/per_device=64
# config (4*1024 == 64*64 == 4096 tokens/device) and yields grad_accum=8.
NUM_GPUS = 8
PER_DEVICE_TRAIN_BATCH_SIZE = 4
TARGET_GLOBAL_TOKENS = 256 * 1024  # 262144

# --- Progressive-cramming evaluation constants (mirrors run_jobs_progressive.py, 1 GPU per job). ---
EMBEDDING_INIT_METHOD = "compression_head_forward"
PROG_MAX_SEQ_LEN = MAX_SEQ_LEN  # evaluate at the sequence length the compression head was trained on
PROG_LIMIT_DATASET_ITEMS = 50  # per-sample optimization is expensive; small eval set like the progressive matrix
PROG_LEARNING_RATE = 0.01
PROG_LOSS_TYPE = "cross_entropy"
PROG_NUM_ALIGNMENT_LAYERS = 1
PROG_MAX_OPTIMIZATION_STEPS_PER_SAMPLE = 10_000
PROG_MAX_OPTIMIZATION_STEPS_PER_TOKEN = 1_000
PROG_WARMUP_STEPS = 100

SMOLLM2 = "HuggingFaceTB/SmolLM2-1.7B"
TRUNCATED_CHECKPOINT_ROOT = "artifacts/checkpoints"
TRUNCATED_KEEP = [1, 2, 4, 8]  # first-last N layers (built by make_first_last_layers_ckpt.py)

# Q-Former configuration shared by the Q-Former experiments.
QFORMER = {"num_queries": 1, "num_layers": 3, "num_heads": 8}

# Experiment matrix (described as code). Each entry is one compression-head training run.
EXPERIMENTS: list[dict] = [
    # 1. Simple MLP compression head on the full model.
    {"model_checkpoint": SMOLLM2, "head_kind": "mlp"},
    # 2. Q-Former compression head on the full model.
    {"model_checkpoint": SMOLLM2, "head_kind": "qformer", **QFORMER},
]
# 3. Truncated-layer ablation: Q-Former head on the first-last{N} checkpoints.
for _n in TRUNCATED_KEEP:
    EXPERIMENTS.append(
        {
            "model_checkpoint": f"{TRUNCATED_CHECKPOINT_ROOT}/SmolLM2-1.7B-firstlast{_n}",
            "head_kind": "qformer",
            **QFORMER,
        }
    )


def compute_grad_accum(per_device_bs: int, num_gpus: int, seq_len: int, target_tokens: int) -> int:
    """Gradient-accumulation steps that hit ``target_tokens`` per optimizer step."""
    tokens_per_micro_step = per_device_bs * num_gpus * seq_len
    if tokens_per_micro_step <= 0 or target_tokens % tokens_per_micro_step != 0:
        raise ValueError(
            f"TARGET_GLOBAL_TOKENS={target_tokens} is not divisible by "
            f"per_device_bs*num_gpus*seq_len={tokens_per_micro_step} "
            f"(per_device_bs={per_device_bs}, num_gpus={num_gpus}, seq_len={seq_len})."
        )
    return target_tokens // tokens_per_micro_step


def _head_tag(experiment: dict) -> str:
    if experiment["head_kind"] == "qformer":
        return f"qformer_q{experiment['num_queries']}_l{experiment['num_layers']}_h{experiment['num_heads']}"
    return "mlp"


def render_ch_job(experiment: dict) -> tuple[list[str], str, str]:
    """Build (cmd_args, exp_suffix, out_dir_name) for one compression-head training run."""
    model_checkpoint = experiment["model_checkpoint"]
    model_short = model_checkpoint.split("/")[-1]
    head_kind = experiment["head_kind"]
    grad_accum = compute_grad_accum(PER_DEVICE_TRAIN_BATCH_SIZE, NUM_GPUS, MAX_SEQ_LEN, TARGET_GLOBAL_TOKENS)

    # The ``ch_head_`` prefix is required: scripts/activation_distillation.py loads the compression-head
    # model class when the checkpoint path contains "experiments_compression_head/ch_head_".
    exp_suffix = f"ch_head_{model_short}_{_head_tag(experiment)}_ds_fineweb-edu_seq_{MAX_SEQ_LEN}"
    exp_suffix = f"{exp_suffix}_lr_{LEARNING_RATE}_a_{DISTILL_ALPHA}_b_{DISTILL_BETA}"
    if not FREEZE_BASE_MODEL:
        exp_suffix = f"{exp_suffix}_unfrozen"
    out_dir_name = f"artifacts/experiments_compression_head/{exp_suffix}"

    cmd_args = [
        "--train_compression_head",
        f"--model_checkpoint {model_checkpoint}",
        f"--dataset_name {DATASET_NAME}",
        f"--limit_dataset_items {LIMIT_DATASET_ITEMS}",
        f"--per_device_train_batch_size {PER_DEVICE_TRAIN_BATCH_SIZE}",
        f"--gradient_accumulation_steps {grad_accum}",
        f"--max_sequence_length {MAX_SEQ_LEN}",
        f"--learning_rate {LEARNING_RATE}",
        f"--compression_head_distill_alpha {DISTILL_ALPHA}",
        f"--compression_head_distill_beta {DISTILL_BETA}",
        f"--compression_head_kind {head_kind}",
        f"--dataloader_num_workers {DATALOADER_NUM_WORKERS}",
        f"--dtype {DTYPE}",
        f"--num_train_epochs {NUM_TRAIN_EPOCHS}",
        f"--weight_decay {WEIGHT_DECAY}",
        f"--max_grad_norm {MAX_GRAD_NORM}",
        f"--warmup_steps {WARMUP_STEPS}",
        f"--logging_steps {LOGGING_STEPS}",
        f"--lr_scheduler_type {LR_SCHEDULER_TYPE}",
        f"--lr_scheduler_kwargs '{LR_SCHEDULER_KWARGS}'",
        f"--compression_head_freeze_base_model {FREEZE_BASE_MODEL}",
    ]
    if head_kind == "qformer":
        cmd_args += [
            f"--compression_head_num_queries {experiment['num_queries']}",
            f"--compression_head_num_layers {experiment['num_layers']}",
            f"--compression_head_num_heads {experiment['num_heads']}",
        ]
    cmd_args.append(f"--output_dir {out_dir_name}")
    cmd_args.append(f"--logging_dir {out_dir_name}/logs")
    return cmd_args, exp_suffix, out_dir_name


def render_eval_job(ch_out_dir_name: str, ch_exp_suffix: str) -> tuple[list[str], str, str]:
    """Build (cmd_args, exp_suffix, out_dir_name) for the progressive eval of one trained head."""
    exp_suffix = f"progeval_chfwd_{ch_exp_suffix}"
    out_dir_name = f"artifacts/experiments_progressive/{exp_suffix}"
    cmd_args = [
        "--remove_unused_columns False",
        f"--num_alignment_layers {PROG_NUM_ALIGNMENT_LAYERS}",
        f"--loss_type {PROG_LOSS_TYPE}",
        f"--max_sequence_length {PROG_MAX_SEQ_LEN}",
        f"--warmup_steps {PROG_WARMUP_STEPS}",
        f"--model_checkpoint {ch_out_dir_name}",
        "--per_device_train_batch_size 1",
        f"--max_optimization_steps_per_sample {PROG_MAX_OPTIMIZATION_STEPS_PER_SAMPLE}",
        f"--max_optimization_steps_per_token {PROG_MAX_OPTIMIZATION_STEPS_PER_TOKEN}",
        f"--learning_rate {PROG_LEARNING_RATE}",
        "--progressive_train 1",
        f"--embedding_init_method {EMBEDDING_INIT_METHOD}",
        f"--limit_dataset_items {PROG_LIMIT_DATASET_ITEMS}",
        f"--dataset_name {DATASET_NAME}",
        f"--output_dir {out_dir_name}",
    ]
    return cmd_args, exp_suffix, out_dir_name


def _filter_experiments(experiments: list[dict], model_filters: list[str] | None) -> list[dict]:
    if not model_filters:
        return experiments
    model_filters = [m.lower() for m in model_filters]
    filtered = []
    for experiment in experiments:
        checkpoint = experiment["model_checkpoint"].lower()
        model_name = checkpoint.split("/")[-1]
        if any(filt in checkpoint or filt in model_name for filt in model_filters):
            filtered.append(experiment)
    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument(
        "--stage",
        choices=["train", "eval"],
        default="train",
        help="train: launch compression-head training (8 GPU). "
        "eval: launch 1-GPU progressive-cramming evaluation of finished heads.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter experiments by model name (substring match against checkpoint path / model name).",
    )
    args = parser.parse_args()

    # Compute nodes mount the repo under /workspace-SR004.nfs2; normalize the dev-shell mount
    # (e.g. /mnt/...-nfs2) to that path so the emitted job paths resolve on the cluster. Works
    # from a worktree too: the suffix after /d.tarasov/ (e.g. compression_horizon/worktrees/...) is kept.
    _cwd = os.getcwd()
    _marker = "/d.tarasov/compression_horizon"
    workdir = "/workspace-SR004.nfs2" + _cwd[_cwd.index(_marker) :] if _marker in _cwd else _cwd
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    client, extra_options = training_job_api_from_profile("default")
    author_name = "d.tarasov"
    in_progress_jobs = get_in_progress_jobs()
    in_progress_job_descs = {job.get("job_desc", "") for job in in_progress_jobs}

    experiments = _filter_experiments(EXPERIMENTS, args.model)
    if not experiments:
        print(f"\033[33mNo models matched the filter: {args.model}\033[0m")
        sys.exit(0)

    for experiment in experiments:
        ch_cmd_args, ch_exp_suffix, ch_out_dir_name = render_ch_job(experiment)

        if args.stage == "train":
            cmd_args, exp_suffix, out_dir_name = ch_cmd_args, ch_exp_suffix, ch_out_dir_name
            if os.path.exists(out_dir_name):
                print("Experiment", out_dir_name, "exists, skip.")
                continue
            script = f"bash {workdir}/scripts/jobs/multigpu.sh scripts/activation_distillation.py  {' '.join(cmd_args)}"
            job_desc = f"CH: compression_head {exp_suffix} #{author_name} #multimodal #notify_completed @mrsndmn"
            instance_type = f"a100.{NUM_GPUS}gpu"
        else:  # eval
            # The trained head must exist before we can evaluate it.
            if not os.path.exists(ch_out_dir_name):
                print(f"\033[33mCompression head not trained yet, skip eval:\033[0m {ch_out_dir_name}")
                continue
            cmd_args, exp_suffix, out_dir_name = render_eval_job(ch_out_dir_name, ch_exp_suffix)
            if os.path.exists(out_dir_name):
                print("Eval", out_dir_name, "exists, skip.")
                continue
            script = f" cd {workdir} && {python_path} scripts/activation_distillation.py  {' '.join(cmd_args)}"
            job_desc = f"CH: progressive_eval {exp_suffix} #{author_name} #multimodal #notify_completed @mrsndmn"
            instance_type = "a100.1gpu"

        if job_desc in in_progress_job_descs:
            print(f"\033[33mSkipping: job already in queue with description:\033[0m {job_desc}")
            continue

        payload = {
            "script": script,
            "job_desc": job_desc,
            "env_variables": {
                "PYTHONPATH": "./src",
                "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
            },
            "instance_type": instance_type,
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41",
            "n_workers": 1,
            "processes_per_worker": 1,
        }

        print(f"\033[32m Would launch with description:\033[0m {job_desc}")
        print(f"\033[90m     Command: {script}\033[0m")
        print(f"\033[90m     Output dir: {out_dir_name}\033[0m")

        if args.dry:
            continue

        result = client.run_job(payload=payload)
        print(out_dir_name, result)
