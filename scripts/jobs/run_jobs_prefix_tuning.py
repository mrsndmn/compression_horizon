import argparse
import json
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch prefix tuning compression_horizon training jobs.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optimizer to use (e.g., 'adamw_torch', 'sgd'). Default: 'adamw_torch'.",
    )
    parser.add_argument("--adam_beta1", type=float, default=None, help="Adam beta1 parameter. Default: 0.9.")
    parser.add_argument("--adam_beta2", type=float, default=None, help="Adam beta2 parameter. Default: 0.999.")
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter models by name (substring match). Can specify multiple models.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype to use: auto | float32/fp32 | bfloat16/bf16 | float16/fp16.",
    )
    parser.add_argument(
        "--limit_dataset_items",
        type=int,
        default=None,
        help="Limit the number of dataset items to use. If not specified, defaults to 10.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name to use for training (e.g., 'mrsndmn/pg19').",
    )
    parser.add_argument(
        "--embedding_init_method",
        type=str,
        default=None,
        help="Initialization method for compression embeddings.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for optimization.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=None,
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--lr_scheduler_kwargs",
        type=str,
        default=None,
        help="Learning rate scheduler kwargs as JSON string (e.g., '{\"min_lr\":1e-3}').",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help="Loss type for activation alignment: l2, l1, cosine, or cross_entropy.",
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=None,
        help="Multiplier in the loss function for hybrid loss.",
    )
    parser.add_argument(
        "--num_alignment_layers",
        type=int,
        default=None,
        help="Number of transformer layers to align (0 = all layers).",
    )
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Truncate dataset to sequence length")

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
        "HuggingFaceTB/SmolLM2-135M",
        "HuggingFaceTB/SmolLM2-360M",
        "unsloth/Llama-3.2-3B",
        "unsloth/Llama-3.2-1B",
        "Qwen/Qwen3-4B",
        "unsloth/Meta-Llama-3.1-8B",
        "Qwen/Qwen3-8B",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1.4b",
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

    max_seq_len = args.max_seq_len
    max_optimization_steps_per_sample = 10_000

    for model_checkpoint in checkpoints:
        exp_suffix = f"pt_sl_{max_seq_len}_{model_checkpoint.split('/')[1]}"

        limit_dataset_items = args.limit_dataset_items if args.limit_dataset_items is not None else 10
        embedding_init_method = args.embedding_init_method if args.embedding_init_method is not None else "random0.02"
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.01
        loss_type = args.loss_type if args.loss_type is not None else "cross_entropy"
        num_alignment_layers = args.num_alignment_layers if args.num_alignment_layers is not None else 1

        cmd_args = [
            "--remove_unused_columns False",
            "--train_prefix_tuning 1",
            f"--num_alignment_layers {num_alignment_layers}",
            f"--loss_type {loss_type}",
            f"--max_sequence_length {max_seq_len}",
            "--warmup_steps 100",
            f"--model_checkpoint {model_checkpoint}",
            "--per_device_train_batch_size 1",
            f"--max_optimization_steps_per_sample {max_optimization_steps_per_sample}",
            f"--learning_rate {learning_rate}",
            f"--embedding_init_method {embedding_init_method}",
            f"--limit_dataset_items {limit_dataset_items}",
        ]

        if args.hybrid_alpha is not None:
            cmd_args.append(f"--hybrid_alpha {args.hybrid_alpha}")

        if args.dataset_name is not None:
            cmd_args.append(f"--dataset_name {args.dataset_name}")
            dataset_suffix = args.dataset_name.split("/")[-1] if "/" in args.dataset_name else args.dataset_name
            exp_suffix = f"{exp_suffix}_ds_{dataset_suffix}"

        if args.dtype:
            cmd_args.append(f"--dtype {args.dtype}")
            exp_suffix = f"{exp_suffix}_dtype_{args.dtype}"

        if args.limit_dataset_items is not None and args.limit_dataset_items != 10:
            exp_suffix = f"{exp_suffix}_limit_{args.limit_dataset_items}"

        if args.embedding_init_method is not None and args.embedding_init_method != "random0.02":
            exp_suffix = f"{exp_suffix}_embinit_{args.embedding_init_method}"

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
        if optim_params:
            exp_suffix = f"{exp_suffix}_{'_'.join(optim_params)}"

        if args.learning_rate is not None and args.learning_rate != 0.01:
            exp_suffix = f"{exp_suffix}_lr_{args.learning_rate}"

        if args.random_seed is not None and args.random_seed != 42:
            cmd_args.append(f"--random_seed {args.random_seed}")
            exp_suffix = f"{exp_suffix}_seed_{args.random_seed}"

        if args.lr_scheduler_type is not None:
            cmd_args.append(f"--lr_scheduler_type {args.lr_scheduler_type}")
            if args.lr_scheduler_type != "cosine":
                exp_suffix = f"{exp_suffix}_sched_{args.lr_scheduler_type}"

        if args.lr_scheduler_kwargs is not None:
            try:
                json.loads(args.lr_scheduler_kwargs)
                cmd_args.append(f"--lr_scheduler_kwargs '{args.lr_scheduler_kwargs}'")
                kwargs_dict = json.loads(args.lr_scheduler_kwargs)
                kwargs_parts = [f"{k}_{v}" for k, v in sorted(kwargs_dict.items())]
                kwargs_suffix = "_".join(kwargs_parts).replace(".", "p").replace("-", "m")
                exp_suffix = f"{exp_suffix}_schedkw_{kwargs_suffix}"
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format for --lr_scheduler_kwargs: {args.lr_scheduler_kwargs}")

        if args.loss_type is not None and args.loss_type != "cross_entropy":
            exp_suffix = f"{exp_suffix}_loss_{args.loss_type}"

        if args.hybrid_alpha is not None:
            exp_suffix = f"{exp_suffix}_hybrid_{args.hybrid_alpha}"

        if args.num_alignment_layers is not None and args.num_alignment_layers != 1:
            exp_suffix = f"{exp_suffix}_align_{args.num_alignment_layers}"

        out_dir_name = f"artifacts/experiments_prefix_tuning/{exp_suffix}"
        if os.path.exists(out_dir_name):
            print("Experiment", out_dir_name, "exists, skip.")
            continue

        cmd_args.append(f"--output_dir {out_dir_name}")
        script = f" cd {workdir} && {python_path} scripts/activation_distillation.py  {' '.join(cmd_args)}"
        job_desc = f"CH: prefix_tuning {exp_suffix} #{author_name} #multimodal #notify_completed @mrsndmn"

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
            "queue_name": "fusionbrainlab-job",
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
