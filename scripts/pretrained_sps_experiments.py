"""Executable experiment design for pretrained SPS cramming (not adaptation)."""

import hashlib
import json
from pathlib import Path

NOTEBOOK = Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov")
COMMON = NOTEBOOK if NOTEBOOK.exists() else Path("/workspace-SR004.nfs2/d.tarasov")
ROOT = COMMON / "sps-artifacts/cramming_pretrained_20261105"
SOURCE_ROOT = COMMON / "sps-artifacts"
UPSTREAM_COMMIT = "c4bc39c141d9004b5f2697a50f90d1e7e63ba767"
FAMILIES = {
    "qwen": dict(
        source="Qwen/Qwen3-1.7B-Base",
        revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        ce="ar-sps-qwen3-1.7b/runs/out/qwen3_staged_sps_ce/final.pt",
        kd="ar-sps-qwen3-1.7b/runs/out/qwen3_staged_sps_kd/final.pt",
    ),
    "smol": dict(
        source="HuggingFaceTB/SmolLM2-1.7B",
        revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
        ce="ar-sps-1.7b-ce/runs/out/smollm2_staged_sps/final.pt",
        kd="ar-sps-1.7b-kd/runs/out/smollm2_staged_sps_kd/final.pt",
    ),
}
COHORT = 48
SHARD_SIZE = 6
MAX_ACTIVE = 8
PREFLIGHT = dict(tokens=8, steps=3, atol=1e-5, rtol=1e-4, bf16_atol=0.25, bf16_rtol=0.05)
TRAINING_ARGS = dict(
    remove_unused_columns=False,
    num_alignment_layers=0,
    loss_type="cross_entropy",
    token_objective="squared_margin",
    loss_margin=0.05,
    convergence_margin=0.05,
    max_sequence_length=4096,
    warmup_steps=100,
    learning_rate=0.3,
    max_optimization_steps_per_sample=10000,
    max_optimization_steps_per_token=1000,
    progressive_train=True,
    embedding_init_method="random0.02",
    number_of_mem_tokens=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    random_seed=42,
    dtype="bfloat16",
    no_bos_token=True,
    report_to="none",
    dataset_name="LarryLovestein/pg19_1k",
    dataloader_num_workers=0,
)
experiment_configs = [
    dict(
        name=f"{family}_{arm}",
        family=family,
        arm=arm,
        checkpoint=None if arm == "ar" else str(SOURCE_ROOT / spec[arm]),
        source=spec["source"],
        revision=spec["revision"],
        source_path=str(
            SOURCE_ROOT / "hf/hub" / ("models--" + spec["source"].replace("/", "--")) / "snapshots" / spec["revision"]
        ),
    )
    for family, spec in FAMILIES.items()
    for arm in ("ar", "ce", "kd")
]
JOB_CONFIG = dict(
    instance_type="a100.1gpu",
    queue_name="fusionbrainlab-job",
    type="binary_exp",
    shm_size_class="large",
    base_image="cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41",
    n_workers=1,
    processes_per_worker=1,
)
PYTHON = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
CODE_FILES = (
    "scripts/pretrained_sps_experiments.py",
    "scripts/run_pretrained_sps_cramming.py",
    "scripts/pretrained_sps_worker.py",
    "src/compression_horizon/models/pretrained_sps.py",
    "src/compression_horizon/models/sps_vendor/pretrained.py",
    "scripts/analyze_sps_predict_embedding.py",
    "src/compression_horizon/train/trainers/progressive_cramming.py",
    "src/compression_horizon/train/trainers/base.py",
    "src/compression_horizon/train/loss.py",
    "src/compression_horizon/train/arguments.py",
    "src/compression_horizon/inference/generation.py",
    "scripts/greedy_reconstruction_eval.py",
    "scripts/validation/validate_sps_batch1_run.py",
)


def code_digest(repo):
    h = hashlib.sha256()
    names = set(CODE_FILES) | {str(p.relative_to(repo)) for p in (Path(repo) / "src").rglob("*.py")}
    for name in sorted(names):
        h.update(name.encode())
        h.update((Path(repo) / name).read_bytes())
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())
