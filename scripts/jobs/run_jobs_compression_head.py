import argparse
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch compression head training jobs.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs per node to request for the job (affects instance_type). Default: 1",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes (workers) for multinode training. Default: 1",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter models by name (substring match). Can specify multiple models.",
    )
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        help="Explicit model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Explicit model checkpoint",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype to use: auto | float32/fp32 | bfloat16/bf16 | float16/fp16. Default: bf16",
    )
    parser.add_argument(
        "--limit_dataset_items",
        type=int,
        default=None,
        help="Limit the number of dataset items to use.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name to use for training (e.g., 'mrsndmn/pg19', 'HuggingFaceFW/fineweb-edu').",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for optimization. Default: 1e-4",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=None,
        help="Learning rate scheduler type. Default: cosine_with_min_lr",
    )
    parser.add_argument(
        "--lr_scheduler_kwargs",
        type=str,
        default=None,
        help="Learning rate scheduler kwargs as JSON string (e.g., '{\"min_lr\":1e-3}').",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Maximum sequence length. Default: 1024",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Batch size per device. Default: 4",
    )
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=None,
        help="Total (global) train batch size across all GPUs and gradient accumulation. Default: 128",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=None,
        help="Number of dataloader workers. Default: 4",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps. If not specified, uses num_train_epochs.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Number of training epochs. Default: 1",
    )
    parser.add_argument(
        "--compression_head_distill_alpha",
        type=float,
        default=None,
        help="Weight for distillation loss. Default: 1.0",
    )
    parser.add_argument(
        "--compression_head_distill_beta",
        type=float,
        default=None,
        help="Weight (beta) for base next-token loss in the compression-head objective. Default: 1.0",
    )
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="Enable torch.autograd.set_detect_anomaly(True) in the training script. Debug only - very slow.",
    )
    parser.add_argument(
        "--compression_head_freeze_base_model",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Freeze base model (default: True). Pass 'false' to unfreeze.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay. Default: 0.01",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Max gradient norm. Default: 1.0",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Number of warmup steps. Default: 0",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Logging steps. Default: 50",
    )
    parser.add_argument(
        "--torch_compile",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Enable torch.compile on the wrapped model after accelerator.prepare().",
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default=None,
        help="torch.compile mode: default | reduce-overhead | max-autotune | max-autotune-no-cudagraphs.",
    )
    parser.add_argument(
        "--separate_reconstructor_model",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Train a separate reconstructor model copy (dual-model ablation).",
    )
    parser.add_argument(
        "--compression_head_kind",
        type=str,
        default=None,
        help="Compression head type: 'mlp' (legacy) or 'qformer'.",
    )
    parser.add_argument(
        "--compression_head_num_queries",
        type=int,
        default=None,
        help="Number of compression tokens (queries) when --compression_head_kind=qformer.",
    )
    parser.add_argument(
        "--fineweb_edu_sample",
        type=str,
        default=None,
        help="HuggingFaceFW/fineweb-edu sample to use: '10BT' (~9M items) or '100BT' (~30M items).",
    )
    parser.add_argument(
        "--truncate_layers",
        type=str,
        default=None,
        help="Truncate LM backbone to subset of layers: csv ('0,1,2,28,29'), 'first_last:K', or 'even:N'.",
    )
    parser.add_argument(
        "--compression_head_num_heads",
        type=int,
        default=None,
        help="Number of attention heads in each Q-Former layer (default: 8).",
    )
    parser.add_argument(
        "--compression_head_num_layers",
        type=int,
        default=None,
        help="Number of cross-attention layers in the Q-Former (default: 1).",
    )
    parser.add_argument(
        "--compression_head_query_proj_factor",
        type=int,
        default=None,
        help=(
            "Width multiplier for the Q-Former learnable query parameter. "
            "When > 1, the trainable query is [N, factor*H] and a linear "
            "projects it down to [N, H] before cross-attention."
        ),
    )
    parser.add_argument(
        "--distill_teacher_checkpoint",
        type=str,
        default=None,
        help="Path to teacher dual checkpoint for KD.",
    )
    parser.add_argument(
        "--distill_alpha",
        type=float,
        default=None,
        help="KD α weight (loss = (1-α)·CE + α·KL).",
    )
    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=None,
        help="KD temperature.",
    )

    args = parser.parse_args()
    workdir = os.getcwd()
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    client, extra_options = training_job_api_from_profile("default")

    author_name = "d.tarasov"
    region = extra_options["region"]
    in_progress_jobs = get_in_progress_jobs()
    in_progress_job_descs = {job.get("job_desc", "") for job in in_progress_jobs}

    checkpoints = [
        "HuggingFaceTB/SmolLM2-1.7B",
        "unsloth/Llama-3.2-3B",
        "Qwen/Qwen3-4B",
        "unsloth/Meta-Llama-3.1-8B",
        "Qwen/Qwen3-8B",
        "allenai/OLMo-1B-hf",
        "allenai/Olmo-3-1025-7B",
        "unsloth/gemma-3-4b-pt",
        "unsloth/gemma-3-1b-pt",
    ]

    if args.model:
        model_filters = [m.lower() for m in args.model]
        filtered_checkpoints = []
        for checkpoint in checkpoints:
            checkpoint_lower = checkpoint.lower()
            model_name = checkpoint.split("/")[-1].lower() if "/" in checkpoint else checkpoint_lower
            if any(filt in checkpoint_lower or filt in model_name for filt in model_filters):
                filtered_checkpoints.append(checkpoint)
        checkpoints = filtered_checkpoints
        if not checkpoints:
            print(f"\033[33mNo models matched the filter: {args.model}\033[0m")
            sys.exit(0)
    elif args.model_checkpoint:
        checkpoints = [args.model_checkpoint]

    for model_checkpoint in checkpoints:
        model_name = model_checkpoint.split("/")[-1]
        if model_name == "":
            model_name = args.model_name
        exp_suffix = f"ch_head_{model_name}"

        # Default values
        limit_dataset_items = args.limit_dataset_items if args.limit_dataset_items is not None else 50000
        dataset_name = args.dataset_name if args.dataset_name is not None else "HuggingFaceFW/fineweb-edu"
        learning_rate = args.learning_rate if args.learning_rate is not None else 1e-4
        max_sequence_length = args.max_sequence_length if args.max_sequence_length is not None else 1024
        per_device_train_batch_size = args.per_device_train_batch_size if args.per_device_train_batch_size is not None else 4
        num_gpus = args.num_gpus if args.num_gpus is not None else 1
        if num_gpus < 1:
            raise ValueError(f"--num_gpus must be >= 1, got {num_gpus}")
        num_nodes = args.num_nodes if args.num_nodes is not None else 1
        if num_nodes < 1:
            raise ValueError(f"--num_nodes must be >= 1, got {num_nodes}")

        total_batch_size = args.total_batch_size if args.total_batch_size is not None else 128
        denom = num_nodes * num_gpus * per_device_train_batch_size
        if denom <= 0:
            raise ValueError(
                f"Invalid batch sizing: num_nodes={num_nodes}, num_gpus={num_gpus}, "
                f"per_device_train_batch_size={per_device_train_batch_size}"
            )
        if total_batch_size % denom != 0:
            raise ValueError(
                "total_batch_size must be divisible by (num_nodes * num_gpus * per_device_train_batch_size). "
                f"Got total_batch_size={total_batch_size}, num_nodes={num_nodes}, num_gpus={num_gpus}, "
                f"per_device_train_batch_size={per_device_train_batch_size}"
            )
        gradient_accumulation_steps = total_batch_size // denom
        if gradient_accumulation_steps < 1:
            raise ValueError(
                "Computed gradient_accumulation_steps < 1. "
                f"Got total_batch_size={total_batch_size}, num_nodes={num_nodes}, num_gpus={num_gpus}, "
                f"per_device_train_batch_size={per_device_train_batch_size}"
            )
        dataloader_num_workers = args.dataloader_num_workers if args.dataloader_num_workers is not None else 4
        compression_head_distill_alpha = (
            args.compression_head_distill_alpha if args.compression_head_distill_alpha is not None else 1.0
        )
        compression_head_distill_beta = (
            args.compression_head_distill_beta if args.compression_head_distill_beta is not None else 1.0
        )
        compression_head_freeze_base_model = (
            args.compression_head_freeze_base_model if args.compression_head_freeze_base_model is not None else True
        )
        dtype = args.dtype if args.dtype is not None else "bf16"
        num_train_epochs = args.num_train_epochs if args.num_train_epochs is not None else 1
        weight_decay = args.weight_decay if args.weight_decay is not None else 0.01
        max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else 1.0
        warmup_steps = args.warmup_steps if args.warmup_steps is not None else 0
        logging_steps = args.logging_steps if args.logging_steps is not None else 50
        lr_scheduler_type = args.lr_scheduler_type if args.lr_scheduler_type is not None else "cosine_with_min_lr"

        cmd_args = (
            [
                "--train_compression_head",
                f"--model_checkpoint {model_checkpoint}",
                f"--dataset_name {dataset_name}",
                f"--limit_dataset_items {limit_dataset_items}",
                f"--per_device_train_batch_size {per_device_train_batch_size}",
                f"--gradient_accumulation_steps {gradient_accumulation_steps}",
                f"--max_sequence_length {max_sequence_length}",
                f"--learning_rate {learning_rate}",
                f"--compression_head_distill_alpha {compression_head_distill_alpha}",
                f"--compression_head_distill_beta {compression_head_distill_beta}",
            ]
            + (["--detect_anomaly"] if args.detect_anomaly else [])
            + [
                f"--dataloader_num_workers {dataloader_num_workers}",
                f"--dtype {dtype}",
                f"--num_train_epochs {num_train_epochs}",
                f"--weight_decay {weight_decay}",
                f"--max_grad_norm {max_grad_norm}",
                f"--warmup_steps {warmup_steps}",
                f"--logging_steps {int(logging_steps)}",
                f"--lr_scheduler_type {lr_scheduler_type}",
            ]
        )

        if not compression_head_freeze_base_model:
            cmd_args.append("--compression_head_freeze_base_model False")
        else:
            exp_suffix = f"{exp_suffix}_freeze_llm"

        if args.torch_compile:
            cmd_args.append("--torch_compile True")
            if args.torch_compile_mode is not None:
                cmd_args.append(f"--torch_compile_mode {args.torch_compile_mode}")
            exp_suffix = f"{exp_suffix}_compile"

        if args.separate_reconstructor_model:
            cmd_args.append("--separate_reconstructor_model True")
            exp_suffix = f"{exp_suffix}_dualmodel"

        if args.compression_head_kind:
            cmd_args.append(f"--compression_head_kind {args.compression_head_kind}")
            if args.compression_head_num_queries is not None:
                cmd_args.append(f"--compression_head_num_queries {args.compression_head_num_queries}")
                exp_suffix = f"{exp_suffix}_{args.compression_head_kind}{args.compression_head_num_queries}"
            else:
                exp_suffix = f"{exp_suffix}_{args.compression_head_kind}"
            if args.compression_head_num_layers is not None:
                cmd_args.append(f"--compression_head_num_layers {args.compression_head_num_layers}")
                exp_suffix = f"{exp_suffix}L{args.compression_head_num_layers}"
            if args.compression_head_num_heads is not None:
                cmd_args.append(f"--compression_head_num_heads {args.compression_head_num_heads}")
                exp_suffix = f"{exp_suffix}H{args.compression_head_num_heads}"
            if args.compression_head_query_proj_factor is not None:
                cmd_args.append(f"--compression_head_query_proj_factor {args.compression_head_query_proj_factor}")
                exp_suffix = f"{exp_suffix}Q{args.compression_head_query_proj_factor}x"

        if args.truncate_layers:
            cmd_args.append(f"--truncate_layers {args.truncate_layers}")
            _trunc_token = args.truncate_layers.replace(":", "").replace(",", "_")
            exp_suffix = f"{exp_suffix}_TRUNC_{_trunc_token}"

        if args.distill_teacher_checkpoint:
            cmd_args.append(f"--distill_teacher_checkpoint {args.distill_teacher_checkpoint}")
            exp_suffix = f"{exp_suffix}_KD"
        if args.distill_alpha is not None:
            cmd_args.append(f"--distill_alpha {args.distill_alpha}")
            exp_suffix = f"{exp_suffix}a{args.distill_alpha}".replace(".", "p")
        if args.distill_temperature is not None:
            cmd_args.append(f"--distill_temperature {args.distill_temperature}")
            exp_suffix = f"{exp_suffix}T{args.distill_temperature}".replace(".", "p")

        if args.fineweb_edu_sample:
            cmd_args.append(f"--fineweb_edu_sample {args.fineweb_edu_sample}")
            if args.fineweb_edu_sample != "10BT":
                exp_suffix = f"{exp_suffix}_fwe{args.fineweb_edu_sample}"

        if args.max_steps is not None:
            cmd_args.append(f"--max_steps {args.max_steps}")
            exp_suffix = f"{exp_suffix}_steps_{args.max_steps}"
        else:
            exp_suffix = f"{exp_suffix}_epochs_{num_train_epochs}"

        if args.random_seed is not None and args.random_seed != 42:
            cmd_args.append(f"--random_seed {args.random_seed}")
            exp_suffix = f"{exp_suffix}_seed_{args.random_seed}"

        if args.lr_scheduler_kwargs is not None:
            cmd_args.append(f"--lr_scheduler_kwargs '{args.lr_scheduler_kwargs}'")
            exp_suffix = f"{exp_suffix}_schedkw_{args.lr_scheduler_kwargs}"

        # Add to exp_suffix if non-default values
        if args.dataset_name is not None:
            dataset_suffix = args.dataset_name.split("/")[-1] if "/" in args.dataset_name else args.dataset_name
            exp_suffix = f"{exp_suffix}_ds_{dataset_suffix}"

        if args.limit_dataset_items is not None and args.limit_dataset_items != 50000:
            exp_suffix = f"{exp_suffix}_limit_{args.limit_dataset_items}"

        if args.max_sequence_length is not None and args.max_sequence_length != 1024:
            exp_suffix = f"{exp_suffix}_seq_{args.max_sequence_length}"

        if args.per_device_train_batch_size is not None and args.per_device_train_batch_size != 4:
            exp_suffix = f"{exp_suffix}_bs_{args.per_device_train_batch_size}"

        if args.total_batch_size is not None and args.total_batch_size != 128:
            exp_suffix = f"{exp_suffix}_tbs_{args.total_batch_size}"

        if args.num_gpus is not None and args.num_gpus != 1:
            exp_suffix = f"{exp_suffix}_ngpu_{args.num_gpus}"

        if args.num_nodes is not None and args.num_nodes != 1:
            exp_suffix = f"{exp_suffix}_nnode_{args.num_nodes}"

        if args.learning_rate is not None and args.learning_rate != 1e-4:
            lr_str = str(args.learning_rate).replace(".", "p").replace("-", "m")
            exp_suffix = f"{exp_suffix}_lr_{lr_str}"

        if args.compression_head_distill_alpha is not None and args.compression_head_distill_alpha != 1.0:
            alpha_str = str(args.compression_head_distill_alpha).replace(".", "p")
            exp_suffix = f"{exp_suffix}_distill_{alpha_str}"

        if args.compression_head_distill_beta is not None and args.compression_head_distill_beta != 1.0:
            beta_str = str(args.compression_head_distill_beta).replace(".", "p")
            exp_suffix = f"{exp_suffix}_beta_{beta_str}"

        if args.detect_anomaly:
            exp_suffix = f"{exp_suffix}_debug_anomaly"

        if args.dtype is not None and args.dtype != "bf16":
            exp_suffix = f"{exp_suffix}_dtype_{args.dtype}"

        if args.lr_scheduler_type is not None and args.lr_scheduler_type != "cosine_with_min_lr":
            exp_suffix = f"{exp_suffix}_sched_{args.lr_scheduler_type}"

        if not compression_head_freeze_base_model:
            exp_suffix = f"{exp_suffix}_unfrozen"

        out_dir_name = f"artifacts/experiments_compression_head/{exp_suffix}"
        logging_dir = f"artifacts/experiments_compression_head/{exp_suffix}/logs"

        if os.path.exists(out_dir_name):
            print("Experiment", out_dir_name, "exists, skip.")
            continue

        cmd_args.append(f"--output_dir {out_dir_name}")
        cmd_args.append(f"--logging_dir {logging_dir}")
        script = f"bash {workdir}/scripts/jobs/multigpu.sh scripts/activation_distillation.py  {' '.join(cmd_args)}"
        job_desc = f"CH: compression_head {exp_suffix} #{author_name} #multimodal #notify_completed @mrsndmn"

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
            "instance_type": f"a100.{num_gpus}gpu",
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41",
            "n_workers": num_nodes,
            "processes_per_worker": 1,
        }

        print(f"\033[32m Would launch with description:\033[0m {job_desc}")
        print(f"\033[90m     Command: {script}\033[0m")
        print(f"\033[90m     Output dir: {out_dir_name}\033[0m")

        if args.dry:
            continue

        result = client.run_job(payload=payload)
        print(out_dir_name, result)
