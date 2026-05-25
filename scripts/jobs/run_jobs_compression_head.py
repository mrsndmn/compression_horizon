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
# Sized for exactly 5k optimizer steps in a single epoch (no data repetition):
# N = 5_000 * (per_device 4 * grad_accum 8 * num_gpus 8) = 5_000 * 256 = 1_280_000 sequences.
# Well under the ~9.67M docs in the fineweb-edu 10BT sample, so each sample is seen once.
LIMIT_DATASET_ITEMS = 1_280_000
MAX_SEQ_LEN = 1024
LEARNING_RATE = 0.001
DISTILL_ALPHA = 1.0
DISTILL_BETA = 0.0
DATALOADER_NUM_WORKERS = 4
DTYPE = "bf16"
NUM_TRAIN_EPOCHS = 1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 500
LOGGING_STEPS = 50
LR_SCHEDULER_TYPE = "cosine_with_min_lr"
LR_SCHEDULER_KWARGS = "min_lr=0.00001"
FREEZE_BASE_MODEL = False
# Bump to give a fresh output dir and avoid colliding with earlier (e.g. failed) runs of the same config.
RUN_TAG = "v3"

# 8-GPU training; keep the global batch at 256k tokens. gradient_accumulation_steps is derived so that
# per_device_batch * grad_accum * num_gpus * seq_len == TARGET_GLOBAL_TOKENS.
# At seq_len=1024, per_device=4 keeps per-device memory equal to the original seq_len=64/per_device=64
# config (4*1024 == 64*64 == 4096 tokens/device) and yields grad_accum=8.
NUM_GPUS = 8
PER_DEVICE_TRAIN_BATCH_SIZE = 4
TARGET_GLOBAL_TOKENS = 256 * 1024  # 262144

# --- Progressive-cramming evaluation constants (mirrors run_jobs_progressive.py, 1 GPU per job). ---
EMBEDDING_INIT_METHOD = "compression_head_forward"
# Evaluate on the canonical progressive-cramming benchmark (pg19_1k / sl_4096 / lr_0.1), identical to
# run_jobs_progressive.py, so the compression_head_forward-init rows are directly comparable to
# tab:layer_ablation and every other progressive table. NOTE: this eval benchmark is intentionally
# unrelated to the head's *training* dataset/seq-len (fineweb-edu / 1024) -- the head just provides the
# per-sample init; the reconstruction is measured on the standard pg19 benchmark.
PROG_DATASET_NAME = "LarryLovestein/pg19_1k"
PROG_MAX_SEQ_LEN = 4096
PROG_LIMIT_DATASET_ITEMS = 50  # per-sample optimization is expensive; small eval set like the progressive matrix
PROG_LEARNING_RATE = 0.1
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


def render_ch_job(experiment: dict, separate_reconstructor_model: bool | None = None) -> tuple[list[str], str, str]:
    """Build (cmd_args, exp_suffix, out_dir_name) for one compression-head training run.

    ``separate_reconstructor_model`` (dual-model ablation) is resolved from the explicit argument
    when given, otherwise from the experiment dict's ``separate_reconstructor_model`` key. Encoding
    it on the experiment lets watchers/launchers that call ``render_ch_job(exp)`` without the flag
    still produce the correct ``_dualmodel`` output dir.
    """
    if separate_reconstructor_model is None:
        separate_reconstructor_model = bool(experiment.get("separate_reconstructor_model", False))
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
    if separate_reconstructor_model:
        exp_suffix = f"{exp_suffix}_dualmodel"
    exp_suffix = f"{exp_suffix}_{RUN_TAG}"
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
    if separate_reconstructor_model:
        cmd_args.append("--separate_reconstructor_model True")
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
        f"--dataset_name {PROG_DATASET_NAME}",
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


AUTHOR_NAME = "d.tarasov"
PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"


def cluster_workdir() -> str:
    """Normalize the dev-shell NFS mount to the compute-node mount (/workspace-SR004.nfs2),
    preserving the worktree suffix so emitted job paths resolve on the cluster."""
    cwd = os.getcwd()
    marker = "/d.tarasov/compression_horizon"
    return "/workspace-SR004.nfs2" + cwd[cwd.index(marker) :] if marker in cwd else cwd


def make_client():
    """Return (client, extra_options) for the default training-job profile."""
    return training_job_api_from_profile("default")


def _single_ckpt_ready(out_dir: str) -> bool:
    """One saved HF model is present: ``config.json`` plus a ``.safetensors`` weights file."""
    if not os.path.isfile(os.path.join(out_dir, "config.json")):
        return False
    try:
        return any(f.endswith(".safetensors") for f in os.listdir(out_dir))
    except OSError:
        return False


def checkpoint_ready(out_dir: str) -> bool:
    """A compression-head run is finished only once its output dir holds a saved model.

    The training job creates ``<out_dir>/logs`` at startup, so plain directory existence is not a
    completion signal; require ``config.json`` plus a ``.safetensors`` weights file (written by
    ``save_pretrained`` at the very end of training). In dual-model mode the two models are saved
    under ``<out_dir>/compressor`` and ``<out_dir>/reconstructor``, so both must be present.
    """
    compressor = os.path.join(out_dir, "compressor")
    reconstructor = os.path.join(out_dir, "reconstructor")
    if os.path.isdir(compressor) and os.path.isdir(reconstructor):
        return _single_ckpt_ready(compressor) and _single_ckpt_ready(reconstructor)
    return _single_ckpt_ready(out_dir)


def ch_job_desc(exp_suffix: str) -> str:
    return f"CH: compression_head {exp_suffix} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"


def eval_job_desc(exp_suffix: str) -> str:
    return f"CH: progressive_eval {exp_suffix} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"


def _payload(script: str, job_desc: str, instance_type: str, region: str) -> dict:
    return {
        "script": script,
        "job_desc": job_desc,
        "env_variables": {"PYTHONPATH": "./src", "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface"},
        "instance_type": instance_type,
        "region": region,
        "type": "binary_exp",
        "shm_size_class": "medium",
        "base_image": "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41",
        "n_workers": 1,
        "processes_per_worker": 1,
    }


def build_job(
    experiment: dict, stage: str, separate_reconstructor_model: bool | None = None
) -> tuple[str, str, str, str] | None:
    """Return (script, job_desc, out_dir_name, instance_type) for ``stage``.

    Returns ``None`` for an eval whose compression-head checkpoint directory does not exist yet.
    ``separate_reconstructor_model`` defaults to ``None`` so ``render_ch_job`` resolves it from the
    experiment dict (see ``render_ch_job``); pass an explicit bool to force it for all experiments.
    """
    _, ch_exp_suffix, ch_out_dir_name = render_ch_job(experiment, separate_reconstructor_model)
    workdir = cluster_workdir()
    if stage == "train":
        ch_cmd_args = render_ch_job(experiment, separate_reconstructor_model)[0]
        script = f"bash {workdir}/scripts/jobs/multigpu.sh scripts/activation_distillation.py  {' '.join(ch_cmd_args)}"
        return script, ch_job_desc(ch_exp_suffix), ch_out_dir_name, f"a100.{NUM_GPUS}gpu"
    if not checkpoint_ready(ch_out_dir_name):
        return None
    cmd_args, exp_suffix, out_dir_name = render_eval_job(ch_out_dir_name, ch_exp_suffix)
    script = f" cd {workdir} && {PYTHON_PATH} scripts/activation_distillation.py  {' '.join(cmd_args)}"
    return script, eval_job_desc(exp_suffix), out_dir_name, "a100.1gpu"


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
    parser.add_argument(
        "--separate_reconstructor_model",
        action="store_true",
        help="Dual-model ablation: train a separate reconstructor copy (saved under compressor/ + "
        "reconstructor/) and tag the run with _dualmodel. The eval stage auto-detects the dual layout.",
    )
    args = parser.parse_args()

    client, extra_options = make_client()
    in_progress_job_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    experiments = _filter_experiments(EXPERIMENTS, args.model)
    if not experiments:
        print(f"\033[33mNo models matched the filter: {args.model}\033[0m")
        sys.exit(0)

    # CLI flag forces dual on for ALL experiments; when absent (None) each experiment's own
    # ``separate_reconstructor_model`` key decides (see render_ch_job).
    srm_override = True if args.separate_reconstructor_model else None

    for experiment in experiments:
        built = build_job(experiment, args.stage, srm_override)
        if built is None:
            print(
                "\033[33mCompression head not trained yet, skip eval:\033[0m " f"{render_ch_job(experiment, srm_override)[2]}"
            )
            continue
        script, job_desc, out_dir_name, instance_type = built

        # Train: skip only if a real checkpoint was saved (a half-built dir of logs/cmd files from a
        # crashed/cancelled run must not block a restart). Eval: dir existence means it already ran.
        already_done = checkpoint_ready(out_dir_name) if args.stage == "train" else os.path.exists(out_dir_name)
        if already_done:
            print(f"{'Experiment' if args.stage == 'train' else 'Eval'} {out_dir_name} exists, skip.")
            continue
        if job_desc in in_progress_job_descs:
            print(f"\033[33mSkipping: job already in queue with description:\033[0m {job_desc}")
            continue

        payload = _payload(script, job_desc, instance_type, extra_options["region"])
        print(f"\033[32m Would launch with description:\033[0m {job_desc}")
        print(f"\033[90m     Command: {script}\033[0m")
        print(f"\033[90m     Output dir: {out_dir_name}\033[0m")

        if args.dry:
            continue

        result = client.run_job(payload=payload)
        print(out_dir_name, result)
