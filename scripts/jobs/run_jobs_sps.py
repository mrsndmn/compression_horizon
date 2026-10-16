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
    {"name": "standard_pilot_1b", "checkpoint": str(SOURCE_RUNS / "s_standard_pilot_1b" / "ckpt_tokens_1000341504_final.pt")},
    {"name": "sps_w0_pilot_1b", "checkpoint": str(SOURCE_RUNS / "s_sps_w0_pilot_1b" / "ckpt_tokens_1000341504_final.pt")},
]
# Matched baseline from run_training.py: progressive + CE, one memory token.
TRAINING_ARGS = {
    "remove_unused_columns": False,
    "num_alignment_layers": 1,
    "loss_type": "cross_entropy",
    "max_sequence_length": 4096,
    "warmup_steps": 100,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "max_optimization_steps_per_sample": 10000,
    "max_optimization_steps_per_token": 1000,
    "learning_rate": 0.1,
    "progressive_train": True,
    "embedding_init_method": "random0.02",
    "limit_dataset_items": 50,
    "dataset_name": "LarryLovestein/pg19_1k",
    "random_seed": 42,
    "dtype": "bfloat16",
    "number_of_mem_tokens": 1,
    "no_bos_token": True,
    "report_to": "none",
}


def build_config(exp, commit):
    design = dict(experiment=exp, training_args=TRAINING_ARGS, commit=commit)
    signature = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()[:12]
    output = ARTIFACTS / f"{exp['name']}_{signature}"
    stage = ARTIFACTS / "code" / commit
    args = dict(TRAINING_ARGS, model_checkpoint=exp["checkpoint"], output_dir=str(output))
    command = [PYTHON, "scripts/activation_distillation.py"]
    for key, value in args.items():
        command.extend([f"--{key}", str(value)])
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(stage))}",
            shlex.join(command),
            f"touch {shlex.quote(str(output / 'completed'))}",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    configs = [build_config(exp, commit) for exp in experiment_configs]
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
                "env_variables": {"PYTHONPATH": str(stage / "src"), "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface"},
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
