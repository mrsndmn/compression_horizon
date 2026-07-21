import argparse
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch HellaSwag compression head evaluation jobs.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated scripts, do not launch jobs.",
    )
    parser.add_argument(
        "--compression_head_checkpoint",
        type=str,
        required=True,
        help="Path to HF checkpoint directory (preferred) or legacy compression_head.pt file (relative to workdir or absolute).",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Base model checkpoint. If not specified, will try to infer from compression_head_checkpoint path or use default.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype to use: auto | float32/fp32 | bfloat16/bf16 | float16/fp16. If not specified, defaults to 'bf16' and is not included in output dir.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="Limit the number of samples to evaluate. If not specified, defaults to 100 and is not included in output dir.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation. If not specified, defaults to 4 and is not included in output dir.",
    )
    parser.add_argument(
        "--compress_prefix_ratio",
        type=float,
        default=None,
        help="Ratio of context tokens to compress (0.0-1.0). If not specified, defaults to 0.5 and is not included in output dir.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not specified, defaults to 42 and is not included in output dir.",
    )
    parser.add_argument(
        "--evaluate_baseline",
        action="store_true",
        default=True,
        help="Evaluate baseline (without compression). Default: True",
    )
    parser.add_argument(
        "--no_evaluate_baseline",
        action="store_false",
        dest="evaluate_baseline",
        help="Skip baseline evaluation.",
    )
    parser.add_argument(
        "--evaluate_compressed",
        action="store_true",
        default=True,
        help="Evaluate with compression. Default: True",
    )
    parser.add_argument(
        "--no_evaluate_compressed",
        action="store_false",
        dest="evaluate_compressed",
        help="Skip compressed evaluation.",
    )
    parser.add_argument(
        "--search_checkpoints",
        action="store_true",
        help="Search for compression-head checkpoints in artifacts/experiments_compression_head/ and evaluate all found.",
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

    # Collect checkpoints to evaluate
    checkpoints_to_evaluate = []

    if args.search_checkpoints:
        # Search for checkpoints in artifacts/experiments_compression_head/
        # - New format: HF checkpoint directory (config.json + model weights)
        # - Legacy format: compression_head.pt
        search_dir = "artifacts/experiments_compression_head"
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                if "compression_head.pt" in files:
                    checkpoints_to_evaluate.append(os.path.join(root, "compression_head.pt"))
                    continue

                if "config.json" in files:
                    has_weights = any(
                        fn.endswith(".safetensors")
                        or fn == "pytorch_model.bin"
                        or fn.endswith(".bin")
                        or fn.endswith(".bin.index.json")
                        or fn.startswith("pytorch_model")
                        for fn in files
                    )
                    if has_weights:
                        checkpoints_to_evaluate.append(root)
        if not checkpoints_to_evaluate:
            print(f"\033[33mNo compression-head checkpoints found in {search_dir}\033[0m")
            sys.exit(0)
        print(f"Found {len(checkpoints_to_evaluate)} checkpoints to evaluate")
    else:
        # Use the single checkpoint provided
        checkpoint_path = args.compression_head_checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(workdir, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            print(f"\033[31mCheckpoint not found: {checkpoint_path}\033[0m")
            sys.exit(1)
        checkpoints_to_evaluate = [checkpoint_path]

    for compression_head_checkpoint in checkpoints_to_evaluate:
        # Determine model checkpoint
        model_checkpoint = args.model_checkpoint
        if model_checkpoint is None:
            # Try to infer from checkpoint metadata
            if os.path.isdir(compression_head_checkpoint):
                # New format: training_args.json written by trainer (best-effort)
                try:
                    import json

                    args_path = os.path.join(compression_head_checkpoint, "training_args.json")
                    if os.path.exists(args_path):
                        with open(args_path, "r", encoding="utf-8") as f:
                            saved_args = json.load(f)
                        if isinstance(saved_args, dict) and "model_checkpoint" in saved_args:
                            model_checkpoint = saved_args["model_checkpoint"]
                except Exception:
                    pass

                # Fallback: config.json may contain _name_or_path
                if model_checkpoint is None:
                    try:
                        import json

                        cfg_path = os.path.join(compression_head_checkpoint, "config.json")
                        if os.path.exists(cfg_path):
                            with open(cfg_path, "r", encoding="utf-8") as f:
                                cfg = json.load(f)
                            if isinstance(cfg, dict) and "_name_or_path" in cfg and cfg["_name_or_path"]:
                                model_checkpoint = cfg["_name_or_path"]
                    except Exception:
                        pass
            else:
                # Legacy: torch checkpoint
                try:
                    import torch

                    checkpoint_data = torch.load(compression_head_checkpoint, map_location="cpu")
                    if "args" in checkpoint_data and isinstance(checkpoint_data["args"], dict):
                        if "model_checkpoint" in checkpoint_data["args"]:
                            model_checkpoint = checkpoint_data["args"]["model_checkpoint"]
                except Exception:
                    pass

            # If still not found, try to infer from directory structure
            if model_checkpoint is None:
                # Default fallback
                model_checkpoint = "meta-llama/Llama-2-7b-hf"
                print(
                    f"\033[33mWarning: Could not infer model_checkpoint from {compression_head_checkpoint}, using default: {model_checkpoint}\033[0m"
                )

        # Build experiment suffix from checkpoint path
        rel_checkpoint_path = os.path.relpath(compression_head_checkpoint, workdir)
        checkpoint_dir = (
            rel_checkpoint_path if os.path.isdir(compression_head_checkpoint) else os.path.dirname(rel_checkpoint_path)
        )
        exp_suffix = checkpoint_dir.replace("artifacts/experiments_compression_head/", "").replace("/", "_")
        if not exp_suffix:
            exp_suffix = "compression_head_eval"

        # Build command arguments with defaults
        limit_samples = args.limit_samples if args.limit_samples is not None else 100
        batch_size = args.batch_size if args.batch_size is not None else 4
        compress_prefix_ratio = args.compress_prefix_ratio if args.compress_prefix_ratio is not None else 0.5
        dtype = args.dtype if args.dtype is not None else "bf16"

        cmd_args = [
            f"--model_checkpoint {model_checkpoint}",
            f"--compression_head_checkpoint {rel_checkpoint_path}",
            f"--limit_samples {limit_samples}",
            f"--batch_size {batch_size}",
            f"--compress_prefix_ratio {compress_prefix_ratio}",
            f"--dtype {dtype}",
        ]

        # Note: --no_evaluate_baseline and --no_evaluate_compressed are handled via args
        # They default to True, so we only add flags if they should be False
        if not args.evaluate_baseline:
            cmd_args.append("--no_evaluate_baseline")
        if not args.evaluate_compressed:
            cmd_args.append("--no_evaluate_compressed")

        # Add random_seed if specified (non-default)
        if args.random_seed is not None and args.random_seed != 42:
            cmd_args.append(f"--random_seed {args.random_seed}")
            exp_suffix = f"{exp_suffix}_seed_{args.random_seed}"

        # Add limit_samples to output dir if specified (non-default)
        if args.limit_samples is not None and args.limit_samples != 100:
            exp_suffix = f"{exp_suffix}_samples_{args.limit_samples}"

        # Add batch_size to output dir if specified (non-default)
        if args.batch_size is not None and args.batch_size != 4:
            exp_suffix = f"{exp_suffix}_batch_{args.batch_size}"

        # Add compress_prefix_ratio to output dir if specified (non-default)
        if args.compress_prefix_ratio is not None and args.compress_prefix_ratio != 0.5:
            exp_suffix = f"{exp_suffix}_ratio_{args.compress_prefix_ratio}"

        # Add dtype to output dir if specified (non-default)
        if args.dtype is not None and args.dtype != "bf16":
            exp_suffix = f"{exp_suffix}_dtype_{args.dtype}"

        if not args.evaluate_baseline:
            exp_suffix = f"{exp_suffix}_no_baseline"
        if not args.evaluate_compressed:
            exp_suffix = f"{exp_suffix}_no_compressed"

        out_dir_name = f"artifacts/hellaswag_compression_head_evaluation/{exp_suffix}"
        if os.path.exists(out_dir_name):
            print("Experiment", out_dir_name, "exists, skip.")
            continue

        # Add output_dir to command
        cmd_args.append(f"--output_dir {out_dir_name}")
        script = f" cd {workdir} && {python_path} -m scripts.hellaswag_compression_head_evaluate  {' '.join(cmd_args)}"
        job_desc = f"CH: hellaswag compression_head eval {exp_suffix} #{author_name} #multimodal #notify_completed @mrsndmn"

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
            "queue_name": "fusionbrainlab-job",
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
