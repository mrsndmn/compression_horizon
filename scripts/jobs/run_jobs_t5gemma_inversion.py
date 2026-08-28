#!/usr/bin/env python3
"""Launcher for the T5Gemma encoder-inversion experiment (3 arms, train + downstream eval).

Tests whether cramming *through an encoder* preserves more downstream capability than cramming
into a decoder-only model of the same family. Spec:
``run/deep-interview/deep-interview-t5gemma-encoder-inversion.md``. Design decisions:
``docs/adr/0005-encoder-effect-attributed-by-injection-site-control.md`` and
``docs/adr/0006-progressive-cramming-instead-of-length-calibration.md``.

Three arms, all at N=1 memory embedding and an identical step budget:

* ``gemma2-baseline``      -- google/gemma-2-2b, decoder-only cramming (the existing method).
* ``t5gemma-enc-input``    -- t5gemma-2b-2b-prefixlm, memory at the encoder input (the hypothesis).
* ``t5gemma-enc-output``   -- t5gemma-2b-2b-prefixlm, memory injected straight into the encoder's
  output space so the encoder never runs (the control that attributes the effect to the encoder).

T5Gemma-2b-2b's encoder and decoder each have exactly gemma-2-2b's layer shape and share its
tokenizer, so the pair is family-matched and bits/token are directly comparable.

Every shared constant below is copied from the existing launchers rather than re-chosen:
progressive constants from ``run_jobs_progressive.py``, downstream-eval constants from
``run_jobs_hellaswag_evaluate.py``.

Usage:
  python scripts/jobs/run_jobs_t5gemma_inversion.py --stage train --dry
  python scripts/jobs/run_jobs_t5gemma_inversion.py --stage train
  python scripts/jobs/run_jobs_t5gemma_inversion.py --stage eval
"""

from __future__ import annotations

import argparse
import os

from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

# ── Shared constants (copied verbatim from run_jobs_progressive.py) ─────────────
DATASET_NAME = "LarryLovestein/pg19_1k"
LIMIT_DATASET_ITEMS = 50
MAX_SEQ_LEN = 4096
MAX_OPTIMIZATION_STEPS_PER_SAMPLE = 10_000
MAX_OPTIMIZATION_STEPS_PER_TOKEN = 1_000
EMBEDDING_INIT_METHOD = "random0.02"
WARMUP_STEPS = 100

# ── Downstream-eval constants (copied from run_jobs_hellaswag_evaluate.py usage) ─
EVAL_LIMIT_SAMPLES = 512
EVAL_NUM_COMPRESSION_TOKENS = 1
EVAL_MAX_OPTIMIZATION_STEPS = 1000
EVAL_BATCH_SIZE = 32
EVAL_BENCHMARKS = [
    ("hellaswag", "scripts.hellaswag_compress_evaluate", "artifacts/hellaswag_evaluation"),
    ("arc", "scripts.arc_compress_evaluate", "artifacts/arc_evaluation"),
]

# ── Job infrastructure (matches every other launcher in this repo) ──────────────
PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
INSTANCE_TYPE = "a100.1gpu"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
QUEUE_NAME = "fusionbrainlab-job"
HF_HOME = "/workspace-SR004.nfs2/.cache/huggingface"

# ── The experiment matrix ───────────────────────────────────────────────────────
# lr 0.1 is the family default used by every non-Pythia model in run_jobs_progressive.py
# (including gemma-3-4b), so both arms inherit it rather than being separately tuned.
LEARNING_RATE = 0.1

ARMS = [
    {
        "name": "gemma2-baseline",
        "model_checkpoint": "google/gemma-2-2b",
        # Decoder-only: no injection site applies.
        "injection_site": None,
        "learning_rate": LEARNING_RATE,
    },
    {
        "name": "t5gemma-enc-input",
        "model_checkpoint": "google/t5gemma-2b-2b-prefixlm",
        "injection_site": "encoder_input",
        "learning_rate": LEARNING_RATE,
    },
    {
        "name": "t5gemma-enc-output",
        "model_checkpoint": "google/t5gemma-2b-2b-prefixlm",
        "injection_site": "encoder_output",
        "learning_rate": LEARNING_RATE,
    },
]


def _arm_suffix(arm: dict, base: str) -> str:
    """Distinguish the two T5Gemma arms, which share a checkpoint and differ only by injection site."""
    if arm["injection_site"] is None:
        return base
    return f"{base}_inj_{arm['injection_site']}"


def render_train_job(arm: dict) -> tuple[list[str], str, str]:
    """Build (cmd_args, exp_suffix, out_dir) for one progressive-cramming arm.

    The suffix construction mirrors run_jobs_progressive.render_job so these runs sit next to the
    existing progressive artifacts and the "exists, skip" idempotency check behaves the same way.
    """
    model_checkpoint = arm["model_checkpoint"]
    model_short = model_checkpoint.split("/")[-1]
    exp_suffix = f"sl_{MAX_SEQ_LEN}_{model_short}"

    cmd_args = [
        "--remove_unused_columns False",
        "--num_alignment_layers 1",
        "--loss_type cross_entropy",
        f"--max_sequence_length {MAX_SEQ_LEN}",
        f"--warmup_steps {WARMUP_STEPS}",
        f"--model_checkpoint {model_checkpoint}",
        "--per_device_train_batch_size 1",
        f"--max_optimization_steps_per_sample {MAX_OPTIMIZATION_STEPS_PER_SAMPLE}",
        f"--max_optimization_steps_per_token {MAX_OPTIMIZATION_STEPS_PER_TOKEN}",
        f"--learning_rate {arm['learning_rate']}",
        "--progressive_train 1",
        f"--embedding_init_method {EMBEDDING_INIT_METHOD}",
        f"--limit_dataset_items {LIMIT_DATASET_ITEMS}",
        f"--dataset_name {DATASET_NAME}",
    ]

    dataset_suffix = DATASET_NAME.split("/")[-1] if "/" in DATASET_NAME else DATASET_NAME
    exp_suffix = f"{exp_suffix}_ds_{dataset_suffix}"
    exp_suffix = f"{exp_suffix}_limit_{LIMIT_DATASET_ITEMS}"
    exp_suffix = f"{exp_suffix}_lr_{arm['learning_rate']}"

    if arm["injection_site"] is not None:
        cmd_args.append(f"--encoder_decoder_injection_site {arm['injection_site']}")
    exp_suffix = _arm_suffix(arm, exp_suffix)

    out_dir_name = f"artifacts/experiments_progressive/{exp_suffix}"
    cmd_args.append(f"--output_dir {out_dir_name}")
    return cmd_args, exp_suffix, out_dir_name


def render_eval_job(arm: dict, benchmark: str, artifacts_root: str) -> tuple[list[str], str, str]:
    """Build (cmd_args, exp_suffix, out_dir) for one downstream compress-evaluate arm."""
    model_checkpoint = arm["model_checkpoint"]
    model_short = model_checkpoint.split("/")[-1]
    exp_suffix = model_short

    cmd_args = [
        f"--model_checkpoint {model_checkpoint}",
        f"--limit_samples {EVAL_LIMIT_SAMPLES}",
        f"--num_compression_tokens {EVAL_NUM_COMPRESSION_TOKENS}",
        f"--max_optimization_steps {EVAL_MAX_OPTIMIZATION_STEPS}",
        f"--learning_rate {arm['learning_rate']}",
        f"--batch_size {EVAL_BATCH_SIZE}",
        "--dtype bf16",
        "--loss_type cross_entropy",
        "--num_alignment_layers 0",
        # Attention knockout is on by default in these evaluators, but it masks attention to the
        # memory's *decoder* position -- which an encoder-decoder does not have, so the analysis has
        # no counterpart there. Disabled on every arm so all three measure the same thing; this
        # experiment reports Base / Acc / Conv%, not the knockout sweep.
        "--no_intervention",
    ]

    exp_suffix = f"{exp_suffix}_samples_{EVAL_LIMIT_SAMPLES}"
    exp_suffix = f"{exp_suffix}_lr_{arm['learning_rate']}"
    exp_suffix = f"{exp_suffix}_batch_{EVAL_BATCH_SIZE}"

    if arm["injection_site"] is not None:
        cmd_args.append(f"--encoder_decoder_injection_site {arm['injection_site']}")
    exp_suffix = _arm_suffix(arm, exp_suffix)

    out_dir_name = f"{artifacts_root}/{exp_suffix}"
    cmd_args.append(f"--output_dir {out_dir_name}")
    return cmd_args, exp_suffix, out_dir_name


def build_jobs(stage: str) -> list[dict]:
    """All (script, job_desc, out_dir) triples for the requested stage."""
    workdir = os.getcwd()
    jobs: list[dict] = []
    for arm in ARMS:
        if stage == "train":
            cmd_args, exp_suffix, out_dir = render_train_job(arm)
            script = f" cd {workdir} && {PYTHON_PATH} scripts/activation_distillation.py  {' '.join(cmd_args)}"
            jobs.append(
                {
                    "arm": arm["name"],
                    "out_dir": out_dir,
                    "script": script,
                    "job_desc": (
                        f"CH: t5gemma-inversion progressive {exp_suffix} "
                        f"#{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"
                    ),
                }
            )
            continue

        for benchmark, module, artifacts_root in EVAL_BENCHMARKS:
            cmd_args, exp_suffix, out_dir = render_eval_job(arm, benchmark, artifacts_root)
            script = f" cd {workdir} && {PYTHON_PATH} -m {module}  {' '.join(cmd_args)}"
            jobs.append(
                {
                    "arm": f"{arm['name']}/{benchmark}",
                    "out_dir": out_dir,
                    "script": script,
                    "job_desc": (
                        f"CH: t5gemma-inversion {benchmark} eval {exp_suffix} "
                        f"#{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"
                    ),
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the T5Gemma encoder-inversion experiment.")
    parser.add_argument("--stage", choices=("train", "eval"), required=True, help="Which stage to submit.")
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument("--arm", nargs="+", default=None, help="Filter arms by substring match on the arm name.")
    args = parser.parse_args()

    client, extra_options = training_job_api_from_profile("default")
    in_progress_job_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    jobs = build_jobs(args.stage)
    if args.arm:
        filters = [a.lower() for a in args.arm]
        jobs = [job for job in jobs if any(f in job["arm"].lower() for f in filters)]
        if not jobs:
            print(f"\033[33mNo arms matched the filter: {args.arm}\033[0m")
            return

    for job in jobs:
        if os.path.exists(job["out_dir"]):
            print("Experiment", job["out_dir"], "exists, skip.")
            continue
        if job["job_desc"] in in_progress_job_descs:
            print(f"\033[33mSkipping: job already in queue with description:\033[0m {job['job_desc']}")
            continue

        payload = {
            "script": job["script"],
            "job_desc": job["job_desc"],
            "env_variables": {
                "PYTHONPATH": "./src",
                "HF_HOME": HF_HOME,
            },
            "instance_type": INSTANCE_TYPE,
            "queue_name": QUEUE_NAME,
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": BASE_IMAGE,
            "n_workers": 1,
            "processes_per_worker": 1,
        }

        print(f"\033[32m Would launch [{job['arm']}]:\033[0m {job['job_desc']}")
        print(f"\033[90m     Command: {job['script']}\033[0m")
        print(f"\033[90m     Output dir: {job['out_dir']}\033[0m")

        if args.dry:
            continue

        result = client.run_job(payload=payload)
        print(job["out_dir"], result)


if __name__ == "__main__":
    main()
