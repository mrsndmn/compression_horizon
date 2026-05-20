import argparse
import json
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

if __name__ == "__main__":

    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if str(v).lower() in ("true", "1", "yes", "t"):
            return True
        elif str(v).lower() in ("false", "0", "no", "f"):
            return False
        else:
            raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

    parser = argparse.ArgumentParser(description="Launch low dim compression_horizon training jobs.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated scripts, do not launch jobs.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optimizer to use (e.g., 'adamw_torch', 'sgd'). Default: 'adamw_torch'.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=None,
        help="Adam beta1 parameter. Default: 0.9.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=None,
        help="Adam beta2 parameter. Default: 0.999.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter models by name (substring match). Can specify multiple models. Matches against model name or full checkpoint path.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype to use: auto | float32/fp32 | bfloat16/bf16 | float16/fp16. If not specified, dtype is not included in output dir.",
    )
    parser.add_argument(
        "--limit_dataset_items",
        type=int,
        default=None,
        help="Limit the number of dataset items to use. If not specified, defaults to 10 and is not included in output dir.",
    )
    parser.add_argument(
        "--offset_dataset_items",
        type=int,
        default=None,
        help="Offset for dataset items selection (applied before limit_dataset_items). If not specified, not included in output dir.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Limit the number of dataset items to use. If not specified, defaults to 10 and is not included in output dir.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name to use for training (e.g., 'mrsndmn/pg19'). If not specified, defaults to 'mrsndmn/pg19' and is not included in output dir.",
    )
    parser.add_argument(
        "--low_dim_size",
        type=int,
        default=None,
        help="Low dimension size for projection. If not specified, not included in output dir.",
    )
    parser.add_argument(
        "--low_dim_projection",
        action="store_true",
        default=False,
        help="Enable low dimension projection. If not specified, not included in output dir.",
    )
    parser.add_argument(
        "--low_dim_projection_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to load low-dimensional projection state from. If not specified, not included in output dir.",
    )
    parser.add_argument(
        "--low_dim_projection_train",
        type=str_to_bool,
        default=None,
        help="Whether to optimize the low-dimensional projection (True/1/yes to train, False/0/no to freeze). If not specified, defaults to True and is not included in output dir.",
    )
    parser.add_argument(
        "--no_low_dim_projection_train",
        action="store_const",
        const=False,
        dest="low_dim_projection_train",
        help="Disable optimization of the low-dimensional projection (freeze it). This is equivalent to --low_dim_projection_train False.",
    )
    parser.add_argument(
        "--embedding_init_path",
        type=str,
        default=None,
        help="Path to file containing initial compression embeddings (when embedding_init_method=load_from_disk). If not specified, not included in output dir.",
    )
    parser.add_argument(
        "--embedding_init_method",
        type=str,
        default=None,
        help="Initialization method for compression embeddings. If not specified, defaults to 'random0.02' and is not included in output dir.",
    )
    parser.add_argument(
        "--load_from_disk_embedding_init_method",
        type=str,
        default=None,
        help="Initialization method to use when generating embeddings for load_from_disk (when embedding_init_path is empty). If not specified, defaults to 'random'.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for optimization. If not specified, defaults to 0.01 and is not included in output dir.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not specified, defaults to 42 and is not included in output dir.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=None,
        help="Learning rate scheduler type. If not specified, defaults to 'cosine' and is not included in output dir.",
    )
    parser.add_argument(
        "--lr_scheduler_kwargs",
        type=str,
        default=None,
        help="Learning rate scheduler kwargs as JSON string (e.g., '{\"min_lr\":1e-3}'). If not specified, not included in output dir.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Maximum sequence length. If not specified, defaults to 2048 and is not included in output dir.",
    )
    args = parser.parse_args()
    workdir = os.getcwd()
    python_path = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"

    client, extra_options = training_job_api_from_profile("default")

    author_name = "d.tarasov"

    # Get in-progress jobs once at the start
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
    ]
    # checkpoints = []

    # Filter checkpoints by --model flag if provided
    if args.model:
        model_filters = [m.lower() for m in args.model]
        filtered_checkpoints = []
        for checkpoint in checkpoints:
            checkpoint_lower = checkpoint.lower()
            model_name = checkpoint.split("/")[-1].lower() if "/" in checkpoint else checkpoint_lower
            # Match if any filter is found in full checkpoint path or model name
            if any(filt in checkpoint_lower or filt in model_name for filt in model_filters):
                filtered_checkpoints.append(checkpoint)
        checkpoints = filtered_checkpoints
        if not checkpoints:
            print(f"\033[33mNo models matched the filter: {args.model}\033[0m")
            sys.exit(0)

    max_optimization_steps_per_sample = 1_000

    for model_checkpoint in checkpoints:
        # Build command arguments
        max_seq_len = args.max_sequence_length if args.max_sequence_length is not None else 128
        limit_dataset_items = args.limit_dataset_items if args.limit_dataset_items is not None else 100
        embedding_init_method = args.embedding_init_method if args.embedding_init_method is not None else "random0.02"
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.01
        per_device_train_batch_size = args.per_device_train_batch_size if args.per_device_train_batch_size else 100

        assert (
            limit_dataset_items >= per_device_train_batch_size
        ), f"limit_dataset_items > per_device_train_batch_size, {limit_dataset_items} >= {per_device_train_batch_size}"

        cmd_args = [
            "--remove_unused_columns False",
            "--loss_type cross_entropy",
            f"--max_sequence_length {max_seq_len}",
            "--warmup_steps 100",
            f"--model_checkpoint {model_checkpoint}",
            f"--per_device_train_batch_size {per_device_train_batch_size}",
            f"--max_optimization_steps_per_sample {max_optimization_steps_per_sample}",
            f"--learning_rate {learning_rate}",
            "--low_dim_train 1",
            f"--embedding_init_method {embedding_init_method}",
            f"--limit_dataset_items {limit_dataset_items}",
        ]

        # Add offset_dataset_items if specified
        if args.offset_dataset_items is not None:
            cmd_args.append(f"--offset_dataset_items {args.offset_dataset_items}")

        exp_suffix = f"sl_{max_seq_len}_{model_checkpoint.split('/')[1]}"

        # Add dataset_name if specified (non-default)
        if args.dataset_name is not None:
            cmd_args.append(f"--dataset_name {args.dataset_name}")
            # Extract dataset name for suffix (last part after /)
            dataset_suffix = args.dataset_name.split("/")[-1] if "/" in args.dataset_name else args.dataset_name
            exp_suffix = f"{exp_suffix}_ds_{dataset_suffix}"

        # Add dtype if specified
        if args.dtype:
            cmd_args.append(f"--dtype {args.dtype}")
            exp_suffix = f"{exp_suffix}_dtype_{args.dtype}"

        # Add limit_dataset_items to output dir if specified (non-default)
        if args.limit_dataset_items is not None and args.limit_dataset_items != 10:
            exp_suffix = f"{exp_suffix}_limit_{args.limit_dataset_items}"

        # Add offset_dataset_items to output dir if specified
        if args.offset_dataset_items is not None:
            exp_suffix = f"{exp_suffix}_offset_{args.offset_dataset_items}"

        # Add low_dim_size if specified
        if args.low_dim_size is not None:
            cmd_args.append(f"--low_dim_size {args.low_dim_size}")
            exp_suffix = f"{exp_suffix}_lowdim_{args.low_dim_size}"

        # Add low_dim_projection if specified
        if args.low_dim_projection:
            cmd_args.append("--low_dim_projection")
            exp_suffix = f"{exp_suffix}_lowproj"

        # Add low_dim_projection_checkpoint if specified
        if args.low_dim_projection_checkpoint is not None:
            cmd_args.append(f"--low_dim_projection_checkpoint {args.low_dim_projection_checkpoint}")
            # Extract checkpoint name for suffix (last part of path)
            checkpoint_name = os.path.basename(args.low_dim_projection_checkpoint).replace(".pt", "").replace(".pth", "")
            exp_suffix = f"{exp_suffix}_lowprojckpt_{checkpoint_name}"

        # Add low_dim_projection_train if specified (non-default)
        if args.low_dim_projection_train is not None:
            cmd_args.append(f"--low_dim_projection_train {args.low_dim_projection_train}")
            if not args.low_dim_projection_train:
                exp_suffix = f"{exp_suffix}_lowprojfrozen"

        # Add embedding_init_method to output dir if specified (non-default)
        if args.embedding_init_method is not None and args.embedding_init_method != "random0.02":
            exp_suffix = f"{exp_suffix}_embinit_{args.embedding_init_method}"

        # Add embedding_init_path if specified
        if args.embedding_init_path is not None:
            cmd_args.append(f"--embedding_init_path {args.embedding_init_path}")
            # Extract path name for suffix (last part of path, without extension)
            path_name = os.path.basename(args.embedding_init_path).replace(".pt", "").replace(".pth", "")
            exp_suffix = f"{exp_suffix}_embpath_{path_name}"

        # Add load_from_disk_embedding_init_method if specified (non-default)
        if args.load_from_disk_embedding_init_method is not None:
            cmd_args.append(f"--load_from_disk_embedding_init_method {args.load_from_disk_embedding_init_method}")
            exp_suffix = f"{exp_suffix}_embgen_{args.load_from_disk_embedding_init_method}"

        # Add optimizer parameters if specified (non-default)
        optim_params = []
        if args.optim is not None:
            cmd_args.append(f"--optim {args.optim}")
            optim_params.append(f"opt_{args.optim}")
        if args.adam_beta1 is not None:
            cmd_args.append(f"--adam_beta1 {args.adam_beta1}")
            optim_params.append(f"b1_{args.adam_beta1}")
        if args.adam_beta2 is not None:
            cmd_args.append(f"--adam_beta2 {args.adam_beta2}")
            optim_params.append(f"b2_{args.adam_beta2}")

        # Update exp_suffix if optimizer parameters are non-default
        if optim_params:
            optim_suffix = "_".join(optim_params)
            exp_suffix = f"{exp_suffix}_{optim_suffix}"

        # Add learning_rate to output dir if specified (non-default)
        if args.learning_rate is not None and args.learning_rate != 0.01:
            exp_suffix = f"{exp_suffix}_lr_{args.learning_rate}"

        # Add random_seed if specified (non-default)
        if args.random_seed is not None and args.random_seed != 42:
            cmd_args.append(f"--random_seed {args.random_seed}")
            exp_suffix = f"{exp_suffix}_seed_{args.random_seed}"

        # Add lr_scheduler_type if specified
        if args.lr_scheduler_type is not None:
            cmd_args.append(f"--lr_scheduler_type {args.lr_scheduler_type}")
            print("args.lr_scheduler_type", args.lr_scheduler_type is None, args.lr_scheduler_type)
            # Add to suffix only if non-default
            exp_suffix = f"{exp_suffix}_sched_{args.lr_scheduler_type}"

        # Add lr_scheduler_kwargs if specified
        if args.lr_scheduler_kwargs is not None:
            # Validate JSON format
            try:
                json.loads(args.lr_scheduler_kwargs)
                cmd_args.append(f"--lr_scheduler_kwargs '{args.lr_scheduler_kwargs}'")
                # Create a short identifier from kwargs for suffix
                kwargs_dict = json.loads(args.lr_scheduler_kwargs)
                kwargs_parts = [f"{k}_{v}" for k, v in sorted(kwargs_dict.items())]
                kwargs_suffix = "_".join(kwargs_parts).replace(".", "p").replace("-", "m")
                exp_suffix = f"{exp_suffix}_schedkw_{kwargs_suffix}"
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format for --lr_scheduler_kwargs: {args.lr_scheduler_kwargs}")

        out_dir_name = f"artifacts/experiments_low_dim/{exp_suffix}"
        if os.path.exists(out_dir_name):
            print("Experiment", out_dir_name, "exists, skip.")
            continue

        # Add output_dir to command
        cmd_args.append(f"--output_dir {out_dir_name}")
        script = f" cd {workdir} && {python_path} scripts/activation_distillation.py  {' '.join(cmd_args)}"
        job_desc = f"CH: low_dim {exp_suffix} #{author_name} #multimodal #notify_completed @mrsndmn"

        # Check if job with same description already exists in queue
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
