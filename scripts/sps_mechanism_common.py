"""Shared saved inputs for mechanism probes; all cohort choices are in callers."""

import json
import subprocess
from pathlib import Path

import torch
from datasets import load_from_disk
from scripts import analyze_sps_cramming as jobs

from compression_horizon.models.sps_causal_lm import load_sps_checkpoint

# Same LR, seed and data in both runs. Use exact converged stages, not a memory
# optimized for a longer prefix and merely evaluated at a truncated length.
experiment_configs = [
    dict(name="standard", run=jobs.ROOT / "standard_pilot_1b_batch1_lr0.3_0f4b7f53eb25"),
    dict(name="sps_w0", run=jobs.ROOT / "sps_w0_pilot_1b_batch1_lr0.3_f59ca78c7aa7"),
]


def source_commit():
    stage = Path(__file__).resolve().parents[1].name
    return (
        stage
        if len(stage) == 40 and all(c in "0123456789abcdef" for c in stage)
        else subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )


def saved_inputs():
    result = {}
    cache = None
    for exp in experiment_configs:
        design = json.loads((exp["run"] / "design.json").read_text())
        if cache is None:
            cache = load_from_disk(jobs.cluster_path(design["training_args"]["tokenized_dataset_path"]))
        groups = {}
        for row in load_from_disk(str(exp["run"] / "progressive_prefixes")):
            groups.setdefault(int(row["sample_id"]), {})[int(row["stage_seq_len"])] = row
        result[exp["name"]] = dict(design=design, stages=groups)
    return result, cache


def converged(row):
    return row["final_convergence"] >= row["convergence_threshold"]


def load_model(design):
    # Keep the weights of the original BF16 experiment, use FP32 arithmetic.
    model = load_sps_checkpoint(jobs.cluster_path(design["experiment"]["checkpoint"]), torch_dtype=torch.bfloat16)
    return model.cuda().float().eval().requires_grad_(False)


def memory_tensor(row, model):
    return torch.tensor([row["embedding"]], device=model.device, dtype=torch.bfloat16).to(model.dtype)


def inputs(model, memory, ids):
    if ids.shape[0] != 1 or memory.shape[0] != 1:
        raise ValueError("Mechanism probes require batch size 1")
    return torch.cat((memory, model.get_input_embeddings()(ids)), dim=1)
