"""Layerwise causal memory-attention ablation on matched saved cramming prefixes.

--phase submit launches a snapshot MLS evaluation; --phase plot redraws artifacts.
No model weights or saved memory embeddings are optimized.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scripts import analyze_sps_cramming as jobs
from scripts.greedy_reconstruction_eval import select_rows
from torch.nn import functional as F

from compression_horizon.models.sps_causal_lm import load_sps_checkpoint

ROOT = jobs.ROOT
OUTPUT = ROOT / "memory_ablation_20261022"
experiment_configs = [
    dict(name="standard", run=ROOT / "standard_pilot_1b_batch1_a5db9c4c6583"),
    dict(name="sps_w0", run=ROOT / "sps_w0_pilot_1b_batch1_48d1890a7c90"),
]
ARITHMETIC = ("bfloat16", "float32")
FAMILIES = ("single", "prefix", "suffix")
EXPECTED_SAMPLES = 48


@contextmanager
def block_memory_attention(model, layers):
    """Remove key-0 edges for q>=1; preserve memory self-edge and all other edges.

    In SPS, q=1 is the prediction readout paired with the memory. Historical
    predict keys stay blocked by the original w=0 mask. Each hook clones its
    own layer's mask, preventing shared-mask mutation across layers.
    """
    handles = []

    def intervene(module, args, kwargs):
        mask = kwargs["attn_block_mask"].clone()
        mask[..., 1:, 0] = False
        return args, dict(kwargs, attn_block_mask=mask)

    try:
        for layer in layers:
            handles.append(model.transformer.h[layer].attn.register_forward_pre_hook(intervene, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def intervention_layers(family, layer, n_layers):
    if family == "single":
        return [layer]
    if family == "prefix":
        return list(range(layer + 1))
    if family == "suffix":
        return list(range(layer, n_layers))
    raise ValueError(family)


@torch.no_grad()
def score(model, memory, ids):
    assert memory.shape[0] == ids.shape[0] == 1
    inputs = torch.cat((memory, model.get_input_embeddings()(ids)), dim=1)
    logits = model(inputs_embeds=inputs).logits[:, :-1]
    loss = F.cross_entropy(logits.float().flatten(0, 1), ids.flatten(), reduction="none")
    return dict(nll=loss.cpu().tolist(), correct=(logits.argmax(-1) == ids)[0].cpu().tolist())


def load_inputs():
    groups = {}
    targets = {}
    for exp in experiment_configs:
        design = json.loads((exp["run"] / "design.json").read_text())
        audit = json.loads((exp["run"] / "single_sample_validation.json").read_text())
        rows = select_rows(load_from_disk(str(exp["run"] / "progressive_prefixes")), "progr")
        assert [r["sample_id"] for r in rows] == list(range(EXPECTED_SAMPLES))
        ids = {r["sample_id"]: r["target_ids"] for r in audit["samples"]}
        for row in rows:
            assert row["final_convergence"] >= row["convergence_threshold"]
            assert row["num_input_tokens"] == len(ids[row["sample_id"]])
        groups[exp["name"]] = dict(design=design, rows=rows, ids=ids)
        for sample, tokens in ids.items():
            if sample in targets:
                length = min(len(tokens), len(targets[sample]))
                assert tokens[:length] == targets[sample][:length]
                targets[sample] = tokens[:length]
            else:
                targets[sample] = tokens
    return groups, targets


def probe():
    torch.set_num_threads(8)
    groups, targets = load_inputs()
    stage_name = Path(__file__).resolve().parents[1].name
    commit = (
        stage_name
        if len(stage_name) == 40 and all(c in "0123456789abcdef" for c in stage_name)
        else subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    report = dict(
        source_commit=commit,
        interpretation="Key-0 attention removal, q>=1. Memory residual remains. Common target length per matched sample; memory was optimized for its original model-specific horizon. FP32 uses BF16-quantized weights and memory cast to FP32.",
        models={},
    )
    for exp in experiment_configs:
        group = groups[exp["name"]]
        checkpoint = group["design"]["experiment"]["checkpoint"]
        checkpoint = jobs.cluster_path(checkpoint)
        model = load_sps_checkpoint(checkpoint, torch_dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
        n_layers = len(model.transformer.h)
        model_result = dict(run=str(exp["run"]), checkpoint=checkpoint, n_layers=n_layers, arithmetic={})
        report["models"][exp["name"]] = model_result
        for arithmetic in ARITHMETIC:
            model.to(dtype=getattr(torch, arithmetic))
            samples = []
            model_result["arithmetic"][arithmetic] = samples
            for row in group["rows"]:
                sample = row["sample_id"]
                ids = torch.tensor([targets[sample]], device="cuda")
                # Preserve the same quantized input when comparing arithmetic modes.
                memory = torch.tensor([row["embedding"]], device="cuda", dtype=torch.bfloat16).to(model.dtype)
                baseline = score(model, memory, ids)
                record = dict(sample_id=sample, length=ids.shape[1], baseline=baseline, interventions={})
                samples.append(record)
                for family in FAMILIES:
                    interventions = []
                    record["interventions"][family] = interventions
                    for layer in range(n_layers):
                        layers = intervention_layers(family, layer, n_layers)
                        with block_memory_attention(model, layers):
                            result = score(model, memory, ids)
                        interventions.append(result)
                print(exp["name"], arithmetic, sample, flush=True)
                jobs.write_json(OUTPUT / "probe.partial.json", report)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    jobs.write_json(OUTPUT / "probe.json", report)
    plot()
    (OUTPUT / "completed").touch()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUTPUT / "probe.json").read_text())
    colors = dict(standard="tab:blue", sps_w0="tab:orange")
    titles = dict(single="Only this layer", prefix="Layers 1 through selected", suffix="Selected through last layer")
    summary = {}
    for metric, filename, ylabel in [
        ("nll", "layer_nll.png", "Change in NLL (nats/token)"),
        ("correct", "layer_accuracy.png", "Change in accuracy (percentage points)"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
        for name, model in data["models"].items():
            for arithmetic, samples in model["arithmetic"].items():
                for ax, family in zip(axes, FAMILIES):
                    # Equal weight per text. Exclude token 1 to avoid comparing
                    # Standard's direct memory residual readout with SPS's predict slot.
                    delta = np.array(
                        [
                            [np.mean(r[metric][1:]) - np.mean(s["baseline"][metric][1:]) for r in s["interventions"][family]]
                            for s in samples
                        ]
                    )
                    if metric == "correct":
                        delta *= 100
                    mean = delta.mean(0)
                    summary[f"{name}/{arithmetic}/{family}/{metric}"] = mean.tolist()
                    x = np.arange(1, len(mean) + 1)
                    ax.plot(
                        x, mean, color=colors[name], ls="-" if arithmetic == "float32" else "--", label=f"{name} {arithmetic}"
                    )
                    ax.set(title=titles[family], xlabel="Layer (1-based)")
                    ax.axhline(0, color="gray", lw=0.6)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        fig.suptitle("Causal memory-key attention ablation; matched prefixes, tokens 2..L, batch 1")
        fig.tight_layout()
        fig.savefig(OUTPUT / filename, dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (name, model) in zip(axes, data["models"].items()):
        samples = model["arithmetic"]["float32"]
        longest = max(s["length"] for s in samples)
        heat = np.full((model["n_layers"], longest), np.nan)
        for layer in range(model["n_layers"]):
            for position in range(longest):
                delta = [
                    s["interventions"]["single"][layer]["nll"][position] - s["baseline"]["nll"][position]
                    for s in samples
                    if s["length"] > position
                ]
                heat[layer, position] = np.mean(delta)
        # Use a shared scale so colors are comparable between models.
        summary[f"{name}/position_nll"] = heat.tolist()
    scale = max(np.nanmax(np.abs(np.array(summary[f"{name}/position_nll"]))) for name in data["models"])
    for ax, (name, model) in zip(axes, data["models"].items()):
        heat = np.array(summary[f"{name}/position_nll"])
        im = ax.imshow(
            heat,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            origin="lower",
            extent=(0.5, heat.shape[1] + 0.5, 0.5, heat.shape[0] + 0.5),
        )
        ax.set(title=name, xlabel="Target position (1-based)")
    axes[0].set_ylabel("Ablated layer (1-based)")
    fig.colorbar(im, ax=axes.tolist(), label="Change in NLL (nats/token)", shrink=0.8)
    fig.suptitle("Single-layer ablation; FP32 arithmetic on BF16-quantized weights/inputs")
    fig.savefig(OUTPUT / "position_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    jobs.write_json(OUTPUT / "summary.json", summary)


def submit(dry=False):
    jobs.OUTPUT = OUTPUT
    jobs.submit(dry=dry, entrypoint="scripts/analyze_sps_memory_ablation.py", job_kind="sps_memory_ablation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["probe", "plot", "submit", "dry"], default="dry")
    args = parser.parse_args()
    {"probe": probe, "plot": plot, "submit": submit, "dry": lambda: submit(dry=True)}[args.phase]()
