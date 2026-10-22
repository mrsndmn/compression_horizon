"""Paired artifact audit and bounded native-checkpoint optimization diagnostics.

Run --phase summary locally, --phase submit for a committed MLS probe, and
--phase plot to combine completed probe output with the artifact comparison.
All diagnostic experiment and job settings are defined here, never in Markdown.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

REPO = Path(__file__).resolve().parents[1]
SHARED_PROJECT = Path("/workspace-SR004.nfs2/d.tarasov/compression_horizon")
ROOT = SHARED_PROJECT / "artifacts" / "sps_cramming"
SOURCE_COMMIT = "2683216db54e0ae8f51a69fa467c2dc07f5509f9"
CACHE_ROOT = ROOT / "code" / SOURCE_COMMIT / "artifacts/cache/tokenized_datasets"
OUTPUT = ROOT / "diagnostics_20261021"
CHECKPOINTS = SHARED_PROJECT.parent / "sps-artifacts/runs/out"
SAMPLE_IDS = tuple(range(8))
BASELINE_LENGTH = 32
GEOMETRY_LENGTH = 16
REFINE_STEPS = 1000
experiment_configs = [
    dict(
        name="standard",
        run=ROOT / "standard_pilot_1b_cf237b7da147",
        checkpoint=CHECKPOINTS / "s_standard_pilot_1b/ckpt_tokens_1000341504_final.pt",
    ),
    dict(
        name="sps_w0",
        run=ROOT / "sps_w0_pilot_1b_fe8f0448fbaf",
        checkpoint=CHECKPOINTS / "s_sps_w0_pilot_1b/ckpt_tokens_1000341504_final.pt",
    ),
]
refinement_configs = [
    dict(name="bf16_lr0.1", dtype="bfloat16", lr=0.1),
    dict(name="bf16_lr0.03", dtype="bfloat16", lr=0.03),
    dict(name="bf16_lr0.3", dtype="bfloat16", lr=0.3),
    dict(name="fp32_lr0.1", dtype="float32", lr=0.1),
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


def read_arrow(path):
    with pa.memory_map(str(path), "r") as handle:
        return ipc.open_stream(handle).read_all()


def load_rows(exp):
    return read_arrow(exp["run"] / "progressive_prefixes/data-00000-of-00001.arrow").to_pylist()


def group_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["sample_id"], []).append(row)
    return {i: sorted(rs, key=lambda r: r["stage_seq_len"]) for i, rs in groups.items()}


def summarize_rows(rows):
    records = []
    for sample, stages in group_rows(rows).items():
        failed = next((r for r in stages if r["final_convergence"] < r["convergence_threshold"]), None)
        success = [r for r in stages if r["final_convergence"] >= r["convergence_threshold"]]
        records.append(
            dict(
                sample_id=sample,
                horizon=max([r["stage_seq_len"] for r in success] or [0]),
                first_failed_length=failed["stage_seq_len"] if failed else None,
                first_failed_accuracy=failed["final_convergence"] if failed else None,
                steps_at_first_failure=failed["steps_taken"] if failed else None,
            )
        )
    return records


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def summary():
    result = {}
    for exp in experiment_configs:
        rows = load_rows(exp)
        samples = summarize_rows(rows)
        horizons = [r["horizon"] for r in samples]
        result[exp["name"]] = dict(
            samples=samples,
            n=len(samples),
            mean_horizon=float(np.mean(horizons)),
            median_horizon=float(np.median(horizons)),
            exhausted_total_budget=sum(
                r["steps_at_first_failure"] >= 10000 for r in samples if r["steps_at_first_failure"] is not None
            ),
        )
    a, b = result["standard"]["samples"], result["sps_w0"]["samples"]
    by_id = {r["sample_id"]: r for r in b}
    if {r["sample_id"] for r in a} != set(by_id):
        raise ValueError("Models were not measured on the same sample IDs")
    result["paired_standard_wins"] = sum(r["horizon"] > by_id[r["sample_id"]]["horizon"] for r in a)
    write_json(OUTPUT / "summary.json", result)
    print(
        json.dumps(
            {k: {x: y for x, y in v.items() if x != "samples"} if isinstance(v, dict) else v for k, v in result.items()}
        ),
        flush=True,
    )
    return result


def token_cache():
    caches = sorted(CACHE_ROOT.glob("*/data-00000-of-00001.arrow"))
    if len(caches) != 2:
        raise ValueError(f"Expected both original tokenization caches at {CACHE_ROOT}")
    tables = [read_arrow(p).select(["input_ids", "attention_mask"]) for p in caches]
    if not tables[0].equals(tables[1]):
        raise ValueError("Tokenized inputs differ across the two original runs")
    return tables[0].to_pylist()


def failure_inputs(exp, tokens):
    groups = group_rows(load_rows(exp))
    failures = []
    for sample in SAMPLE_IDS:
        failed = next(r for r in groups[sample] if r["final_convergence"] < r["convergence_threshold"])
        length = failed["stage_seq_len"]
        if sum(tokens[sample]["attention_mask"][:length]) != length:
            raise ValueError("Diagnostic requires unpadded failed prefixes")
        failures.append(
            dict(sample_id=sample, length=length, embedding=failed["embedding"], recorded_accuracy=failed["final_convergence"])
        )
    return failures


def score_memory(model, memory, ids, mask):
    import torch
    from torch.nn import functional as F

    embeddings = model.get_input_embeddings()(ids)
    combined = torch.cat((memory.to(embeddings.dtype), embeddings), dim=1)
    attention = torch.cat((torch.ones_like(mask[:, :1]), mask), dim=1)
    # Position zero predicts target zero: the mem-only prefix must score the first token.
    logits = model(inputs_embeds=combined, attention_mask=attention).logits[:, :-1]
    per_token = F.cross_entropy(logits.flatten(0, 1), ids.flatten(), reduction="none").view_as(ids)
    losses = (per_token.float() * mask).sum(1) / mask.sum(1)
    accuracy = ((logits.argmax(-1) == ids) * mask).sum(1).float() / mask.sum(1)
    return losses, accuracy


def refine(model, memory_init, ids, mask, *, lr, max_steps):
    """Restart Adam on saved first-failure memories; no extra free parameters.

    This is a diagnostic restart, not resumption of unavailable optimizer moments.
    Freeze a sample before stepping once its currently measured memory is exact.
    """
    import torch

    memory = torch.nn.Parameter(memory_init.clone().float())
    # Match MyTrainingArguments Adam defaults used by the original runs.
    # Restarted moments and constant LR are explicit parts of this diagnostic.
    optimizer = torch.optim.AdamW([memory], lr=lr, betas=(0.9, 0.9), weight_decay=0.01)
    used = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    best = torch.zeros(ids.shape[0], device=ids.device)
    trace = []
    initial_accuracy = None
    for step in range(max_steps + 1):
        losses, accuracy = score_memory(model, memory, ids, mask)
        accuracy = accuracy.detach()
        if initial_accuracy is None:
            initial_accuracy = accuracy.clone()
        best = torch.maximum(best, accuracy)
        active = best < 1.0
        if step % 100 == 0 or step == max_steps or not active.any():
            trace.append(dict(step=step, losses=losses.detach().cpu().tolist(), accuracy=accuracy.cpu().tolist()))
        if step == max_steps or not active.any():
            break
        optimizer.zero_grad(set_to_none=True)
        losses[active].sum().backward()
        memory.grad[~active] = 0
        snapshot = memory.detach().clone()
        optimizer.step()
        with torch.no_grad():
            memory[~active] = snapshot[~active]
        used += active.long()
    return dict(
        initial_accuracy=initial_accuracy.cpu().tolist(),
        best_accuracy=best.cpu().tolist(),
        final_accuracy=accuracy.cpu().tolist(),
        steps=used.cpu().tolist(),
        trace=trace,
    )


def geometry(model, tokens, sample_ids=SAMPLE_IDS):
    """Natural-prefix quality and local sensitivity at matched length/initialization."""
    import torch
    from torch.nn import functional as F

    from compression_horizon.models.sps_vendor.components import apply_rotary_emb

    device = model.device
    ids = torch.tensor([tokens[i]["input_ids"][:BASELINE_LENGTH] for i in sample_ids], device=device)
    with torch.no_grad():
        logits = model(inputs_embeds=model.get_input_embeddings()(ids)).logits[:, :-1]
        ce = F.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten(), reduction="none").view(ids.shape[0], -1)
        baseline = dict(
            mean_nll_nats=ce.mean(1).cpu().tolist(),
            token_accuracy=(logits.argmax(-1) == ids[:, 1:]).float().mean(1).cpu().tolist(),
        )
    ids = ids[:, :GEOMETRY_LENGTH]
    torch.manual_seed(42)
    memory = (torch.randn(ids.shape[0], 1, model.config.hidden_size, device=device) * 0.02).requires_grad_(True)
    embeddings = model.get_input_embeddings()(ids)
    combined = torch.cat((memory.to(embeddings.dtype), embeddings), dim=1)
    masses = []
    handles = []

    def capture(module, args, kwargs):
        with torch.no_grad():
            x, freqs = args[:2]
            b, t, c = x.shape
            q, k, _ = module.c_attn(x).split(module.hidden_size, dim=2)
            q, k = apply_rotary_emb(q.view(b, t, module.n_head, -1), k.view(b, t, module.n_head, -1), freqs)
            scores = q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2) / math.sqrt(c / module.n_head)
            mask = kwargs["attn_block_mask"]
            weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
            # Readout query associated with each next-token target. Exclude final unused logit.
            readout = weights[:, :, 1::2, 0] if model.config.sps else weights[:, :, :, 0]
            masses.append(readout[:, :, :GEOMETRY_LENGTH].mean(1).cpu().tolist())

    for block in model.transformer.h:
        handles.append(block.attn.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        logits = model(inputs_embeds=combined).logits[:, :-1]
    finally:
        for handle in handles:
            handle.remove()
    loss = F.cross_entropy(logits.float().flatten(0, 1), ids.flatten(), reduction="none").view_as(ids)
    gradients = []
    for position in range(GEOMETRY_LENGTH):
        gradient = torch.autograd.grad(loss[:, position].sum(), memory, retain_graph=position + 1 < GEOMETRY_LENGTH)[0]
        gradients.append(gradient[:, 0].detach().float())
    jacobian = torch.stack(gradients, dim=1)
    norms = jacobian.norm(dim=-1)
    normalized = F.normalize(jacobian, dim=-1)
    cosine = normalized @ normalized.transpose(-1, -2)
    singular = torch.linalg.svdvals(jacobian)
    result = dict(
        baseline=baseline,
        token_gradient_norms=norms.cpu().tolist(),
        token_gradient_cosines=cosine.cpu().tolist(),
        gradient_singular_values=singular.cpu().tolist(),
        memory_attention_by_layer=masses,
    )
    return result


def probe():
    import torch

    from compression_horizon.models.sps_causal_lm import load_sps_checkpoint

    torch.set_num_threads(8)
    tokens = token_cache()
    report = dict(source_commit=SOURCE_COMMIT, sample_ids=list(SAMPLE_IDS), models={})
    for exp in experiment_configs:
        failed = failure_inputs(exp, tokens)
        lengths = [r["length"] for r in failed]
        max_len = max(lengths)
        ids = torch.tensor([tokens[i]["input_ids"][:max_len] for i in SAMPLE_IDS], device="cuda")
        mask = (torch.arange(max_len, device="cuda")[None] < torch.tensor(lengths, device="cuda")[:, None]).long()
        memory = torch.tensor([r["embedding"] for r in failed], device="cuda")
        record = dict(first_failures=[{k: v for k, v in r.items() if k != "embedding"} for r in failed], refinements={})
        report["models"][exp["name"]] = record
        for dtype_name in dict.fromkeys(c["dtype"] for c in refinement_configs):
            model = load_sps_checkpoint(exp["checkpoint"], torch_dtype=getattr(torch, dtype_name)).cuda()
            if dtype_name == "bfloat16":
                record["geometry"] = geometry(model, tokens)
                write_json(OUTPUT / "probe.partial.json", report)
            for cfg in [c for c in refinement_configs if c["dtype"] == dtype_name]:
                result = refine(model, memory, ids, mask, lr=cfg["lr"], max_steps=REFINE_STEPS)
                if dtype_name == "bfloat16":
                    expected = torch.tensor([r["recorded_accuracy"] for r in failed])
                    torch.testing.assert_close(torch.tensor(result["initial_accuracy"]), expected, atol=1e-6, rtol=0)
                record["refinements"][cfg["name"]] = result
                print(exp["name"], cfg["name"], "rescued", sum(x == 1 for x in result["best_accuracy"]), flush=True)
                write_json(OUTPUT / "probe.partial.json", report)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    write_json(OUTPUT / "probe.json", report)
    (OUTPUT / "completed").touch()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = summary()
    a = np.array([r["horizon"] for r in result["standard"]["samples"]])
    b = np.array([r["horizon"] for r in result["sps_w0"]["samples"]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(a, b, alpha=0.7)
    axes[0].plot([0, 50], [0, 50], ls="--", color="gray")
    axes[0].set(
        xlabel="Standard horizon (tokens)", ylabel="SPS w=0 horizon (tokens)", title="Matched texts, exact reconstruction"
    )
    for values, label in [(a, "Standard"), (b, "SPS w=0")]:
        lengths = np.arange(1, 51)
        axes[1].plot(lengths, [(values >= n).mean() for n in lengths], label=label)
    axes[1].set(
        xlabel="Prefix length (tokens)",
        ylabel="Fraction reaching this length",
        title="Progressive success under original optimizer",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "horizons.png", dpi=180)
    plt.close(fig)


def cluster_path(path):
    path = Path(path)
    root = Path("/mnt/virtual_ai0001053-00054_SR004-nfs2")
    return str(Path("/workspace-SR004.nfs2") / path.relative_to(root)) if path.is_relative_to(root) else str(path)


def submit(dry=False, *, entrypoint="scripts/analyze_sps_cramming.py", job_kind="sps_diagnostics", probe_args=()):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=REPO):
        raise RuntimeError("Commit tracked changes first")
    stage = ROOT / "code" / commit
    script = f"cd {shlex.quote(cluster_path(stage))} && " + shlex.join([PYTHON, entrypoint, "--phase", "probe", *probe_args])
    payload = dict(
        JOB_CONFIG,
        script=script,
        job_desc=f"CH: {job_kind} {commit[:12]} #d.tarasov",
        env_variables={
            "PYTHONPATH": f"{cluster_path(stage)}:{cluster_path(stage / 'src')}",
            "OMP_NUM_THREADS": "8",
            "PYTHONUNBUFFERED": "1",
        },
    )
    if dry:
        print(json.dumps(payload, indent=2))
        return
    if any((OUTPUT / name).exists() for name in ["job.json", "completed", "submission-intent.json"]):
        raise RuntimeError("Existing diagnostic submission; inspect it before another run")
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    client, options = training_job_api_from_profile("default")
    if payload["job_desc"] in {j.get("job_desc") for j in get_in_progress_jobs()}:
        raise RuntimeError("A matching diagnostic job is already active")
    if not stage.exists():
        temporary = stage.with_suffix(".tmp")
        temporary.mkdir(parents=True, exist_ok=False)
        archive = subprocess.Popen(["git", "archive", commit], cwd=REPO, stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(temporary)], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait():
            raise RuntimeError("git archive failed")
        temporary.rename(stage)
    payload["region"] = options["region"]
    write_json(OUTPUT / "submission-intent.json", dict(commit=commit, payload=payload))
    response = client.run_job(payload=payload)
    write_json(OUTPUT / "job.json", dict(commit=commit, result=response))
    print(json.dumps(response), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["summary", "probe", "plot", "submit", "dry"], default="summary")
    args = parser.parse_args()
    {"summary": summary, "probe": probe, "plot": plot, "submit": submit, "dry": lambda: submit(dry=True)}[args.phase]()
