"""Submit 1-GPU MLS jobs to compute accuracy-basin areas for each model.

Each job runs plot_basin_area_vs_stage.py for all 10 samples of one model,
storing per-sample NPZ caches under the experiment's visualizations/ dir.

Usage:
    python scripts/jobs/run_jobs_basin_area.py --dry          # preview commands
    python scripts/jobs/run_jobs_basin_area.py                # submit all
"""

from __future__ import annotations

import os

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR = "d.tarasov"

MODELS = [
    {
        "name": "Llama-3.1-8B",
        "exp_dir": "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1",
    },
    {
        "name": "SmolLM2-1.7B",
        "exp_dir": "artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1",
    },
    {
        "name": "Pythia-1.4B",
        "exp_dir": "artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.1",
    },
    {
        "name": "SmolLM2-135M",
        "exp_dir": "artifacts/experiments_progressive/sl_4096_SmolLM2-135M_lr_0.1",
    },
]

SAMPLE_IDS = "0 1 2 3 4 5 6 7 8 9"
COMMON_ARGS = "--num_anchors 8 --mesh_resolution 60 --batch_size 32 --recompute"


def job_desc_for(model_name: str) -> str:
    return f"CH: basin_area {model_name} #{AUTHOR} #multimodal #notify_completed @mrsndmn"


def build_script(model: dict) -> str:
    exp_dir = model["exp_dir"]
    dataset_path = f"{exp_dir}/progressive_prefixes"
    output = f"{exp_dir}/visualizations/basin_area_vs_stage_normalised.png"
    return (
        f"cd {PROJ} && {PYTHON} scripts/paper/plot_basin_area_vs_stage.py"
        f" --dataset_path {dataset_path}"
        f" --sample_ids {SAMPLE_IDS}"
        f" {COMMON_ARGS}"
        f" --output {output}"
    )


def make_client():
    return training_job_api_from_profile("default")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry", action="store_true", help="Print commands without submitting")
    args = parser.parse_args()

    client, extra_options = make_client()
    in_progress = {j.get("job_desc", "") for j in get_in_progress_jobs()}

    submitted = []
    for model in MODELS:
        desc = job_desc_for(model["name"])
        script = build_script(model)

        print(f"\033[32m{model['name']}\033[0m: {desc}")
        print(f"  \033[90m{script}\033[0m")

        if desc in in_progress:
            print("  \033[33mSkipping: already in queue\033[0m")
            continue

        if args.dry:
            print("  (dry run)")
            continue

        payload = {
            "script": script,
            "job_desc": desc,
            "env_variables": {
                "PYTHONPATH": "./src",
                "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
            },
            "instance_type": "a100.1gpu",
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41",
            "n_workers": 1,
            "processes_per_worker": 1,
        }

        result = client.run_job(payload=payload)
        job_name = (result or {}).get("job_name", "???")
        print(f"  \033[32mSubmitted: {job_name}\033[0m")
        submitted.append({"name": model["name"], "job_name": job_name, "desc": desc})

    if submitted:
        print(f"\n{len(submitted)} jobs submitted.")
        for s in submitted:
            print(f"  {s['name']}: {s['job_name']}")
    return submitted


if __name__ == "__main__":
    main()
