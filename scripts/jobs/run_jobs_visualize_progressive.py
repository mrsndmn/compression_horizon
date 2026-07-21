import argparse
import os
import sys

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch progressive embeddings visualization jobs.")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Only print generated scripts, do not launch jobs.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to progressive_prefixes dataset (or base directory to search for datasets).",
    )
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        type=int,
        default=None,
        help="List of sample IDs to process. If not specified, processes all samples found in dataset.",
    )
    parser.add_argument(
        "--perplexity_max_samples",
        type=int,
        default=128,
        help="Max rows to use for perplexity estimation. Default: 128.",
    )
    parser.add_argument(
        "--perplexity_model",
        type=str,
        default=None,
        help="HF model name to compute token-level perplexity. If not specified, will try to infer from dataset.",
    )
    parser.add_argument(
        "--process_samples",
        action="store_true",
        default=False,
        help="Enable per-sample processing. Default: False.",
    )
    parser.add_argument(
        "--draw-landscape",
        action="store_true",
        default=False,
        help="Draw loss landscape for PCA component pairs. Default: False.",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=2.0,
        help="Maximum radius for neighborhood loss computation in PCA space. Default: 2.0.",
    )
    parser.add_argument(
        "--draw-landscape-points-step",
        type=int,
        default=2,
        help="Compute landscape only for every Nth point. Default: 2.",
    )
    parser.add_argument(
        "--draw-landscape-points-limit",
        type=int,
        default=4,
        help="Limit number of points for GIF visualization. Default: 4.",
    )
    parser.add_argument(
        "--mesh_resolution",
        type=int,
        default=40,
        help="Resolution of the mesh grid for loss landscape computation. Default: 40.",
    )
    parser.add_argument(
        "--landscape_pairs_limit",
        type=int,
        default=2,
        help="Limit number of PCA component pairs to compute landscape for. Default: 2.",
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

    # Determine dataset paths to process
    dataset_paths = []
    if os.path.isdir(args.dataset_path) and os.path.exists(os.path.join(args.dataset_path, "progressive_prefixes")):
        # If it's a directory containing progressive_prefixes, use it
        dataset_paths.append(os.path.join(args.dataset_path, "progressive_prefixes"))
    elif os.path.exists(args.dataset_path):
        # If it's a direct path to progressive_prefixes, use it
        dataset_paths.append(args.dataset_path)
    else:
        # Try to find all progressive_prefixes directories under the given path
        if os.path.isdir(args.dataset_path):
            for root, dirs, files in os.walk(args.dataset_path):
                if "progressive_prefixes" in dirs:
                    dataset_paths.append(os.path.join(root, "progressive_prefixes"))
        if not dataset_paths:
            print(f"\033[33mNo progressive_prefixes datasets found at: {args.dataset_path}\033[0m")
            sys.exit(1)

    # Determine sample IDs to process
    sample_ids_to_process = args.sample_ids
    if sample_ids_to_process is None:
        # Try to infer from first dataset (assuming all datasets have same samples)
        # This is a simple heuristic - in practice you might want to scan all datasets
        try:
            from datasets import Dataset

            ds = Dataset.load_from_disk(dataset_paths[0])
            unique_sample_ids = sorted(set(int(r.get("sample_id", -1)) for r in ds if r.get("sample_id") is not None))
            sample_ids_to_process = unique_sample_ids
            print(f"Inferred {len(sample_ids_to_process)} sample IDs from dataset")
        except Exception as e:
            print(f"Could not infer sample IDs from dataset: {e}")
            print("Please specify --sample_ids explicitly")
            sys.exit(1)

    # Infer perplexity model from dataset if not provided
    perplexity_model = args.perplexity_model
    if perplexity_model is None:
        try:
            from datasets import Dataset

            ds = Dataset.load_from_disk(dataset_paths[0])
            # Try to get model_checkpoint from dataset metadata or first row
            model_names = [str(r.get("model_checkpoint", "")).strip() for r in ds if r.get("model_checkpoint")]
            if model_names:
                # Use most common model name
                from collections import Counter

                most_common = Counter(model_names).most_common(1)[0][0]
                perplexity_model = most_common
                print(f"Inferred perplexity_model: {perplexity_model}")
        except Exception as e:
            print(f"Could not infer perplexity_model from dataset: {e}")
            print("Please specify --perplexity_model explicitly")
            sys.exit(1)

    # Build job commands for each dataset_path and sample_id combination
    for dataset_path in dataset_paths:
        # Extract experiment name from path for job description
        # Path format: artifacts/experiments_progressive/<exp_name>/progressive_prefixes
        exp_name = os.path.basename(os.path.dirname(dataset_path))
        if not exp_name or exp_name == "progressive_prefixes":
            # Fallback: use parent directory name
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_path)))

        for sample_id in sample_ids_to_process:
            # Build command arguments
            cmd_args = [
                f"--dataset_path {dataset_path}",
                f"--sample_id {sample_id}",
                f"--perplexity_max_samples {args.perplexity_max_samples}",
                f"--perplexity_model {perplexity_model}",
            ]

            if args.process_samples:
                cmd_args.append("--process_samples")

            if args.draw_landscape:
                cmd_args.append("--draw-landscape")
                cmd_args.append(f"--max-radius {args.max_radius}")
                cmd_args.append(f"--draw-landscape-points-step {args.draw_landscape_points_step}")
                cmd_args.append(f"--draw-landscape-points-limit {args.draw_landscape_points_limit}")
                cmd_args.append(f"--mesh_resolution {args.mesh_resolution}")
                cmd_args.append(f"--landscape_pairs_limit {args.landscape_pairs_limit}")

            script = f" cd {workdir} && {python_path} scripts/visualize_progressive_embeddings.py {' '.join(cmd_args)}"
            job_desc = (
                f"CH: visualize progressive {exp_name} sample_{sample_id} #{author_name} #multimodal #notify_completed @mrsndmn"
            )

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
                "n_workers": 1,
                "processes_per_worker": 1,
            }

            print(f"\033[32m Would launch with description:\033[0m {job_desc}")
            print(f"\033[90m     Command: {script}\033[0m")
            print(f"\033[90m     Dataset: {dataset_path}\033[0m")
            print(f"\033[90m     Sample ID: {sample_id}\033[0m")

            if args.dry:
                continue

            result = client.run_job(payload=payload)
            print(f"Launched job: {job_desc}")
            print(f"Result: {result}")
