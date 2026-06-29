import argparse
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

# Shared constants for every progressive experiment (ported from
# scripts/progressive_experiments.sh, which baked these into each invocation).
DATASET_NAME = "LarryLovestein/pg19_1k"
LIMIT_DATASET_ITEMS = 50
MAX_SEQ_LEN = 4096
MAX_OPTIMIZATION_STEPS_PER_SAMPLE = 10_000
MAX_OPTIMIZATION_STEPS_PER_TOKEN = 1_000
EMBEDDING_INIT_METHOD = "random0.02"
WARMUP_STEPS = 100

# Experiments are grouped per model (ported 1:1 from
# scripts/progressive_experiments.sh). Each group lists the four variants:
# baseline, low-dim, hybrid alpha, and hybrid alpha + low-projection.
LLAMA_31_8B_EXPERIMENTS = [
    {
        "model_checkpoint": "unsloth/Meta-Llama-3.1-8B",
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
    {
        "model_checkpoint": "unsloth/Meta-Llama-3.1-8B",
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": True,
        "low_dim_size": 256,
    },
    {
        # NOTE: anomaly preserved from the shell script — Llama-3.1-8B's plain
        # hybrid variant used 4 alignment layers while every other model used 8.
        "model_checkpoint": "unsloth/Meta-Llama-3.1-8B",
        "learning_rate": 0.1,
        "loss_type": "cosine",
        "num_alignment_layers": 4,
        "hybrid_alpha": 1.0,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
    {
        "model_checkpoint": "unsloth/Meta-Llama-3.1-8B",
        "learning_rate": 0.1,
        "loss_type": "cosine",
        "num_alignment_layers": 8,
        "hybrid_alpha": 1.0,
        "low_dim_projection": True,
        "low_dim_size": 256,
    },
]

PYTHIA_14B_EXPERIMENTS = [
    {
        "model_checkpoint": "EleutherAI/pythia-1.4b",
        "learning_rate": 0.5,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
    {
        "model_checkpoint": "EleutherAI/pythia-1.4b",
        "learning_rate": 0.5,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": True,
        "low_dim_size": 256,
    },
    {
        "model_checkpoint": "EleutherAI/pythia-1.4b",
        "learning_rate": 0.5,
        "loss_type": "cosine",
        "num_alignment_layers": 8,
        "hybrid_alpha": 1.0,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
    {
        "model_checkpoint": "EleutherAI/pythia-1.4b",
        "learning_rate": 0.5,
        "loss_type": "cosine",
        "num_alignment_layers": 8,
        "hybrid_alpha": 1.0,
        "low_dim_projection": True,
        "low_dim_size": 256,
    },
]

SMOLLM2_17B_EXPERIMENTS = [
    {
        "model_checkpoint": "HuggingFaceTB/SmolLM2-1.7B",
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
    {
        "model_checkpoint": "HuggingFaceTB/SmolLM2-1.7B",
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": True,
        "low_dim_size": 256,
    },
    {
        "model_checkpoint": "HuggingFaceTB/SmolLM2-1.7B",
        "learning_rate": 0.1,
        "loss_type": "cosine",
        "num_alignment_layers": 8,
        "hybrid_alpha": 1.0,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
    {
        "model_checkpoint": "HuggingFaceTB/SmolLM2-1.7B",
        "learning_rate": 0.1,
        "loss_type": "cosine",
        "num_alignment_layers": 8,
        "hybrid_alpha": 1.0,
        "low_dim_projection": True,
        "low_dim_size": 256,
    },
]

GEMMA_3_4B_EXPERIMENTS = [
    {
        "model_checkpoint": "unsloth/gemma-3-4b-pt",
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
    {
        "model_checkpoint": "unsloth/gemma-3-4b-pt",
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": True,
        "low_dim_size": 32,
    },
    {
        "model_checkpoint": "unsloth/gemma-3-4b-pt",
        "learning_rate": 0.1,
        "loss_type": "cosine",
        "num_alignment_layers": 8,
        "hybrid_alpha": 1.0,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
    {
        "model_checkpoint": "unsloth/gemma-3-4b-pt",
        "learning_rate": 0.1,
        "loss_type": "cosine",
        "num_alignment_layers": 8,
        "hybrid_alpha": 1.0,
        "low_dim_projection": True,
        "low_dim_size": 32,
    },
]

# Full-depth Qwen3 reference rows for the width/size sweep (baseline variant only --
# byte-identical config to the other full-depth reference rows, so the full-model row is
# comparable to the truncated first-N Qwen3 rows). No low-dim / hybrid variants requested.
QWEN3_EXPERIMENTS = [
    {
        "model_checkpoint": "Qwen/Qwen3-4B",
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
    {
        "model_checkpoint": "Qwen/Qwen3-8B",
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
    },
]

# --- Convergence-margin reproduction of tab:progressive_modifications --------------------
# The 8 rows of paper/tables/progressive_modifications.tex are the cross_entropy baseline +
# low-dim variants (first two entries of each model group above). Reproduce all 8 under the
# decode-robustness variants: for each margin epsilon in CONVERGENCE_MARGINS, a plain-CE arm
# (convergence_margin=eps) and a loss-margin arm (convergence_margin=eps and loss_margin=eps).
# 8 base x len(CONVERGENCE_MARGINS) x 2 experiments (= 32 for eps in {0.5, 1.0}).
# convergence_margin requires every token to clear an epsilon logit margin (decode-robust,
# honest greedy reconstruction); loss_margin additionally reweights CE toward the deficient
# tokens. (model_checkpoint, learning_rate, low_dim_size) match the table rows 1:1; low_dim_size
# None => baseline row.
PROGRESSIVE_MODIFICATIONS_BASE = [
    ("unsloth/Meta-Llama-3.1-8B", 0.1, None),
    ("unsloth/Meta-Llama-3.1-8B", 0.1, 256),
    ("EleutherAI/pythia-1.4b", 0.5, None),
    ("EleutherAI/pythia-1.4b", 0.5, 256),
    ("HuggingFaceTB/SmolLM2-1.7B", 0.1, None),
    ("HuggingFaceTB/SmolLM2-1.7B", 0.1, 256),
    ("unsloth/gemma-3-4b-pt", 0.1, None),
    ("unsloth/gemma-3-4b-pt", 0.1, 32),
]

CONVERGENCE_MARGINS = [0.5, 1.0]

MARGIN_EXPERIMENTS = [
    {
        "model_checkpoint": model_checkpoint,
        "learning_rate": learning_rate,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": low_dim_size is not None,
        "low_dim_size": low_dim_size,
        "convergence_margin": convergence_margin,
        # None for the plain-CE arm (no flag), == convergence_margin for the reweighted arm.
        "loss_margin": convergence_margin if with_loss_margin else None,
    }
    for (model_checkpoint, learning_rate, low_dim_size) in PROGRESSIVE_MODIFICATIONS_BASE
    for convergence_margin in CONVERGENCE_MARGINS
    for with_loss_margin in (False, True)
]

EXPERIMENTS = [
    *LLAMA_31_8B_EXPERIMENTS,
    *PYTHIA_14B_EXPERIMENTS,
    *SMOLLM2_17B_EXPERIMENTS,
    *GEMMA_3_4B_EXPERIMENTS,
    *QWEN3_EXPERIMENTS,
    *MARGIN_EXPERIMENTS,
]


def render_job(experiment):
    """Build (cmd_args, exp_suffix, out_dir_name) for one experiment.

    The exp_suffix construction order mirrors the previous CLI-driven version so
    output directory names stay byte-identical to prior runs (the "exists, skip"
    idempotency check depends on this).
    """
    model_checkpoint = experiment["model_checkpoint"]
    model_short = model_checkpoint.split("/")[-1]
    exp_suffix = f"sl_{MAX_SEQ_LEN}_{model_short}"

    cmd_args = [
        "--remove_unused_columns False",
        f"--num_alignment_layers {experiment['num_alignment_layers']}",
        f"--loss_type {experiment['loss_type']}",
        f"--max_sequence_length {MAX_SEQ_LEN}",
        f"--warmup_steps {WARMUP_STEPS}",
        f"--model_checkpoint {model_checkpoint}",
        "--per_device_train_batch_size 1",
        f"--max_optimization_steps_per_sample {MAX_OPTIMIZATION_STEPS_PER_SAMPLE}",
        f"--max_optimization_steps_per_token {MAX_OPTIMIZATION_STEPS_PER_TOKEN}",
        f"--learning_rate {experiment['learning_rate']}",
        "--progressive_train 1",
        f"--embedding_init_method {EMBEDDING_INIT_METHOD}",
        f"--limit_dataset_items {LIMIT_DATASET_ITEMS}",
    ]

    # Hybrid alpha (cmd flag only; suffix added later to preserve ordering).
    if experiment["hybrid_alpha"] is not None:
        cmd_args.append(f"--hybrid_alpha {experiment['hybrid_alpha']}")

    # Dataset name (always set for these experiments).
    cmd_args.append(f"--dataset_name {DATASET_NAME}")
    dataset_suffix = DATASET_NAME.split("/")[-1] if "/" in DATASET_NAME else DATASET_NAME
    exp_suffix = f"{exp_suffix}_ds_{dataset_suffix}"

    # limit_dataset_items (only added to suffix when non-default vs legacy 10).
    if LIMIT_DATASET_ITEMS != 10:
        exp_suffix = f"{exp_suffix}_limit_{LIMIT_DATASET_ITEMS}"

    # Low dimension projection.
    if experiment["low_dim_size"] is not None:
        cmd_args.append(f"--low_dim_size {experiment['low_dim_size']}")
        exp_suffix = f"{exp_suffix}_lowdim_{experiment['low_dim_size']}"
    if experiment["low_dim_projection"]:
        cmd_args.append("--low_dim_projection")
        exp_suffix = f"{exp_suffix}_lowproj"

    # learning_rate (only added to suffix when non-default vs legacy 0.01).
    if experiment["learning_rate"] != 0.01:
        exp_suffix = f"{exp_suffix}_lr_{experiment['learning_rate']}"

    # loss_type (only added to suffix when non-default vs legacy cross_entropy).
    if experiment["loss_type"] != "cross_entropy":
        exp_suffix = f"{exp_suffix}_loss_{experiment['loss_type']}"

    # hybrid_alpha suffix.
    if experiment["hybrid_alpha"] is not None:
        exp_suffix = f"{exp_suffix}_hybrid_{experiment['hybrid_alpha']}"

    # num_alignment_layers (only added to suffix when non-default vs legacy 1).
    if experiment["num_alignment_layers"] != 1:
        exp_suffix = f"{exp_suffix}_align_{experiment['num_alignment_layers']}"

    # Convergence/loss margin (decode-robustness variants). ``.get`` keeps every existing experiment
    # byte-identical (no key => no flag, no suffix). convergence_margin forces an epsilon logit
    # margin on the convergence criterion; loss_margin additionally reweights CE toward deficient
    # tokens. See src/compression_horizon/train/{arguments,loss}.py.
    convergence_margin = experiment.get("convergence_margin")
    if convergence_margin is not None:
        cmd_args.append(f"--convergence_margin {convergence_margin}")
        exp_suffix = f"{exp_suffix}_cm_{convergence_margin}"
    loss_margin = experiment.get("loss_margin")
    if loss_margin is not None:
        cmd_args.append(f"--loss_margin {loss_margin}")
        exp_suffix = f"{exp_suffix}_lm_{loss_margin}"

    # Fixed uncompressed prefix (progressive cramming). ``.get`` keeps every existing experiment
    # byte-identical (no key => no flag, no suffix); set it to enable the prefix-length ablation.
    prefix_len = experiment.get("progressive_prefix_len")
    if prefix_len:
        cmd_args.append(f"--progressive_prefix_len {prefix_len}")
        exp_suffix = f"{exp_suffix}_prefix_{prefix_len}"

    # Initialization from a prior experiment's artifacts at a specific stage.
    init_artifact = experiment.get("progressive_init_from_artifact")
    init_stage = experiment.get("progressive_init_from_stage")
    init_sample_ids = experiment.get("progressive_init_sample_ids")
    if init_artifact and init_stage is not None:
        cmd_args.append(f"--progressive_init_from_artifact {init_artifact}")
        cmd_args.append(f"--progressive_init_from_stage {init_stage}")
        # Derive a short suffix from the artifact directory basename.
        artifact_short = os.path.basename(init_artifact.rstrip("/"))
        exp_suffix = f"{exp_suffix}_initfrom_{artifact_short}_stage{init_stage}"
        if init_sample_ids is not None:
            cmd_args.append(f"--progressive_init_sample_ids {init_sample_ids}")
            exp_suffix = f"{exp_suffix}_s{init_sample_ids}"

    out_dir_name = f"artifacts/experiments_progressive/{exp_suffix}"
    cmd_args.append(f"--output_dir {out_dir_name}")
    return cmd_args, exp_suffix, out_dir_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch progressive compression_horizon training jobs "
        "(experiment matrix ported from scripts/progressive_experiments.sh)."
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated scripts, do not launch jobs.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter experiments by model name (substring match). Can specify multiple models. "
        "Matches against the full checkpoint path or the model name.",
    )
    args = parser.parse_args()
    workdir = os.getcwd()
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    client, extra_options = training_job_api_from_profile("default")

    author_name = "d.tarasov"

    # Get in-progress jobs once at the start.
    in_progress_jobs = get_in_progress_jobs()
    in_progress_job_descs = {job.get("job_desc", "") for job in in_progress_jobs}

    experiments = EXPERIMENTS

    # Filter experiments by --model flag if provided.
    if args.model:
        model_filters = [m.lower() for m in args.model]
        filtered_experiments = []
        for experiment in experiments:
            checkpoint = experiment["model_checkpoint"]
            checkpoint_lower = checkpoint.lower()
            model_name = checkpoint.split("/")[-1].lower() if "/" in checkpoint else checkpoint_lower
            if any(filt in checkpoint_lower or filt in model_name for filt in model_filters):
                filtered_experiments.append(experiment)
        experiments = filtered_experiments
        if not experiments:
            print(f"\033[33mNo models matched the filter: {args.model}\033[0m")
            sys.exit(0)

    for experiment in experiments:
        cmd_args, exp_suffix, out_dir_name = render_job(experiment)

        if os.path.exists(out_dir_name):
            print("Experiment", out_dir_name, "exists, skip.")
            continue

        script = f" cd {workdir} && {python_path} scripts/activation_distillation.py  {' '.join(cmd_args)}"
        job_desc = f"CH: progressive {exp_suffix} #{author_name} #multimodal #notify_completed @mrsndmn"

        # Check if job with same description already exists in queue.
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
            "instance_type": "a100.1gpu",
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41",
            "n_workers": 1,  # Количество воркеров.
            "processes_per_worker": 1,  # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
        }

        print(f"\033[32m Would launch with description:\033[0m {job_desc}")
        print(f"\033[90m     Command: {script}\033[0m")
        print(f"\033[90m     Output dir: {out_dir_name}\033[0m")

        if args.dry:
            continue

        result = client.run_job(payload=payload)
        print(out_dir_name, result)
