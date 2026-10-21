"""Launch the matched native SPS/Standard progressive cramming pair.

Uses the existing project's MLS profile, interpreter and artifact layout.
Jobs execute a committed code snapshot; completion is recorded only after exit 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "artifacts" / "sps_cramming"
PYTHON = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
SOURCE_RUNS = REPO.parent / "sps-artifacts" / "runs" / "out"

experiment_configs = [
    {
        "name": "standard_pilot_1b_batch1",
        "checkpoint": str(SOURCE_RUNS / "s_standard_pilot_1b" / "ckpt_tokens_1000341504_final.pt"),
    },
    {
        "name": "sps_w0_pilot_1b_batch1",
        "checkpoint": str(SOURCE_RUNS / "s_sps_w0_pilot_1b" / "ckpt_tokens_1000341504_final.pt"),
    },
]
# Re-measure the same 48 texts actually processed by the original batch-four runs.
# Single-sample cramming is required for all future launches.
TRAINING_ARGS = {
    "remove_unused_columns": False,
    "num_alignment_layers": 1,
    "loss_type": "cross_entropy",
    "max_sequence_length": 4096,
    "warmup_steps": 100,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_optimization_steps_per_sample": 10000,
    "max_optimization_steps_per_token": 1000,
    "learning_rate": 0.1,
    "progressive_train": True,
    "embedding_init_method": "random0.02",
    "limit_dataset_items": 48,
    "dataset_name": "LarryLovestein/pg19_1k",
    "random_seed": 42,
    "dtype": "bfloat16",
    "number_of_mem_tokens": 1,
    "no_bos_token": True,
    "report_to": "none",
    "tokenized_dataset_path": str(
        ARTIFACTS
        / "code"
        / "2683216db54e0ae8f51a69fa467c2dc07f5509f9"
        / "artifacts/cache/tokenized_datasets/dataset_fa91d5c47dbf1180"
    ),
}


# Completed single-sample baselines are reused instead of launching duplicates.
BASELINE_RUNS = {
    "standard_pilot_1b_batch1": ARTIFACTS / "standard_pilot_1b_batch1_a5db9c4c6583",
    "sps_w0_pilot_1b_batch1": ARTIFACTS / "sps_w0_pilot_1b_batch1_48d1890a7c90",
}
lr_sweep_experiment_configs = [
    dict(
        name=f"{exp['name']}_lr{lr:g}",
        checkpoint=exp["checkpoint"],
        training_args={"learning_rate": lr},
        baseline_run=str(BASELINE_RUNS[exp["name"]]),
    )
    for exp in experiment_configs
    for lr in (0.01, 0.03, 0.3, 1.0)
]


def cluster_path(path):
    """Translate the notebook mount to the established MLS compute-node mount."""
    path = Path(path)
    notebook_root = Path("/mnt/virtual_ai0001053-00054_SR004-nfs2")
    if path.is_relative_to(notebook_root):
        return str(Path("/workspace-SR004.nfs2") / path.relative_to(notebook_root))
    return str(path)


def build_config(exp, commit):
    training_args = dict(TRAINING_ARGS, **exp.get("training_args", {}))
    design = dict(experiment=exp, training_args=training_args, commit=commit)
    signature = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()[:12]
    output = ARTIFACTS / f"{exp['name']}_{signature}"
    stage = ARTIFACTS / "code" / commit
    args = dict(training_args, model_checkpoint=cluster_path(exp["checkpoint"]), output_dir=cluster_path(output))
    args["tokenized_dataset_path"] = cluster_path(args["tokenized_dataset_path"])
    command = [PYTHON, "scripts/activation_distillation.py"]
    for key, value in args.items():
        command.extend([f"--{key}", str(value)])
    script = " && ".join(
        [
            f"cd {shlex.quote(cluster_path(stage))}",
            shlex.join(command),
            shlex.join([PYTHON, "scripts/validation/validate_sps_batch1_run.py", "--run-dir", cluster_path(output)]),
            f"touch {shlex.quote(cluster_path(output / 'completed'))}",
        ]
    )
    return dict(
        design=design,
        output=output,
        stage=stage,
        signature=signature,
        script=script,
        job_desc=f"CH: sps_cramming {exp['name']} {signature} #d.tarasov",
    )


def staged_pythonpath(stage):
    return f"{cluster_path(stage)}:{cluster_path(stage / 'src')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--suite", choices=("batch1", "lr_sweep"), default="batch1")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    experiments = lr_sweep_experiment_configs if args.suite == "lr_sweep" else experiment_configs
    configs = [build_config(exp, commit) for exp in experiments]
    if args.dry:
        for cfg in configs:
            print(cfg["job_desc"], cfg["script"], sep="\n")
        return
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=REPO):
        raise RuntimeError("Commit tracked changes before submission")
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    client, options = training_job_api_from_profile("default")
    running = {j.get("job_desc") for j in get_in_progress_jobs()}
    stage = configs[0]["stage"]
    if not stage.exists():
        stage.parent.mkdir(parents=True, exist_ok=True)
        temporary = stage.with_name(stage.name + ".tmp")
        temporary.mkdir(exist_ok=False)
        archive = subprocess.Popen(["git", "archive", commit], cwd=REPO, stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(temporary)], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait():
            raise RuntimeError("git archive failed")
        temporary.rename(stage)
    launched = []
    for cfg in configs:
        if (cfg["output"] / "completed").exists() or cfg["job_desc"] in running:
            print("Skipping existing:", cfg["job_desc"])
            continue
        cfg["output"].mkdir(parents=True, exist_ok=True)
        (cfg["output"] / "design.json").write_text(json.dumps(cfg["design"], indent=2))
        result = client.run_job(
            payload={
                "script": cfg["script"],
                "job_desc": cfg["job_desc"],
                "env_variables": {
                    "PYTHONPATH": staged_pythonpath(stage),
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "8",
                    "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
                },
                "instance_type": "a100.1gpu",
                "queue_name": "fusionbrainlab-job",
                "region": options["region"],
                "type": "binary_exp",
                "shm_size_class": "large",
                "base_image": "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41",
                "n_workers": 1,
                "processes_per_worker": 1,
            }
        )
        record = dict(result=result, output=str(cfg["output"]), commit=commit)
        (cfg["output"] / "job.json").write_text(json.dumps(record, indent=2, default=str))
        launched.append(record)
        print(json.dumps(record, default=str), flush=True)
    print(json.dumps({"jobs": launched}, default=str))


if __name__ == "__main__":
    main()
