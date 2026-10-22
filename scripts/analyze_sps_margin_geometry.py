"""Local token-margin controllability at the first failed SPS prefix.

Jacobian rank is local to a selected memory and competitor set. It is not a
capacity bound. Models see the identical target prefix and have the same width.
"""

import argparse
import gc
import json

import numpy as np
import torch
from scripts import analyze_sps_cramming as jobs
from scripts.sps_mechanism_common import (
    converged,
    experiment_configs,
    inputs,
    load_model,
    memory_tensor,
    saved_inputs,
    source_commit,
)

OUTPUT = jobs.ROOT / "mechanism_margin_geometry_20261022"
SAMPLE_IDS = tuple(range(8))
TARGET_MARGIN = 0.05
ROW_RANK_RTOL = 1e-6
TRUST_RADII = (0.001, 0.01, 0.05)
STATES = ("warm_start", "optimized")


def target_margins(logits, ids):
    if logits.shape[0] != 1 or logits.shape[:2] != ids.shape:
        raise ValueError("Expected single-sample aligned target logits")
    candidates = logits.detach().clone()
    candidates.scatter_(-1, ids[..., None], -torch.inf)
    rival = candidates.argmax(-1)
    return (logits.gather(-1, ids[..., None]) - logits.gather(-1, rival[..., None]))[0, :, 0]


def margin_statistics(margins, jacobian):
    j = jacobian.detach().double().cpu()
    m = margins.detach().double().cpu()
    norms = j.norm(dim=-1)
    normalized = j / norms[:, None].clamp_min(1e-12)
    spectrum = torch.linalg.svdvals(normalized)
    old = j[:-1]
    _, s, vh = torch.linalg.svd(old, full_matrices=False)
    rank = int((s > s[0] * ROW_RANK_RTOL).sum()) if len(s) and s[0] > 0 else 0
    free = j[-1] - (vh[:rank].T @ (vh[:rank] @ j[-1]))
    free_norm = float(free.norm())
    ratio = free_norm / max(float(norms[-1]), 1e-12)
    direction = free / free.norm().clamp_min(1e-12)
    required = max(TARGET_MARGIN - float(m[-1]), 0) / free_norm if free_norm > 1e-10 else None
    cosine = normalized[:-1] @ normalized[-1]
    return (
        dict(
            margins=m.tolist(),
            gradient_norms=norms.tolist(),
            normalized_singular_values=spectrum.tolist(),
            gradient_gram=(j @ j.T).tolist(),
            old_row_rank=rank,
            new_gradient_free_ratio=ratio,
            required_linear_step_l2=required,
            negative_cosine_fraction=float((cosine < -1e-3).double().mean()),
            min_old_margin=float(m[:-1].min()),
            new_margin=float(m[-1]),
            max_protected_direction_derivative=float((old @ direction).abs().max()),
        ),
        direction,
    )


def geometry(model, row, ids):
    memory = memory_tensor(row, model).requires_grad_(True)
    logits = model(inputs_embeds=inputs(model, memory, ids)).logits[:, :-1]
    margins = target_margins(logits, ids)
    jacobian = torch.stack(
        [
            torch.autograd.grad(margins[i], memory, retain_graph=i + 1 < len(margins))[0].flatten().detach()
            for i in range(len(margins))
        ]
    )
    stats, direction = margin_statistics(margins, jacobian)
    stats["memory_norm"] = float(memory.detach().norm())
    stats["correct_tokens"] = int((logits.detach().argmax(-1) == ids).sum())
    stats["stored_converged"] = bool(converged(row))
    stats["local_trials"] = []
    for radius in TRUST_RADII:
        step = radius * stats["memory_norm"]
        moved = memory.detach() + step * direction.to(device=memory.device, dtype=memory.dtype).view_as(memory)
        with torch.no_grad():
            actual_logits = model(inputs_embeds=inputs(model, moved, ids)).logits[:, :-1]
            actual = target_margins(actual_logits, ids)
        predicted = margins.detach().double().cpu() + step * (jacobian.double().cpu() @ direction)
        stats["local_trials"].append(
            dict(
                relative_radius=radius,
                linear_new_margin=float(predicted[-1]),
                linear_min_old_margin=float(predicted[:-1].min()),
                actual_relative_radius=float((moved - memory.detach()).norm()) / stats["memory_norm"],
                actual_new_margin=float(actual[-1]),
                actual_min_old_margin=float(actual[:-1].min()),
                correct_tokens=int((actual_logits.argmax(-1) == ids).sum()),
            )
        )
    return stats


def probe():
    torch.set_num_threads(8)
    groups, cache = saved_inputs()
    lengths = {}
    for sid in SAMPLE_IDS:
        stages = groups["sps_w0"]["stages"][sid]
        length = next(n for n, row in sorted(stages.items()) if not converged(row))
        for exp in experiment_configs:
            assert converged(groups[exp["name"]]["stages"][sid][length - 1])
        lengths[sid] = length
    result = dict(source_commit=source_commit(), sample_ids=list(SAMPLE_IDS), models={})
    for exp in experiment_configs:
        group = groups[exp["name"]]
        model = load_model(group["design"])
        record = dict(run=str(exp["run"]), samples=[])
        result["models"][exp["name"]] = record
        for sid in SAMPLE_IDS:
            length = lengths[sid]
            ids = torch.tensor([cache[sid]["input_ids"][:length]], device=model.device)
            sample = dict(sample_id=sid, target_length=length, states={})
            record["samples"].append(sample)
            for state in STATES:
                row = group["stages"][sid][length - 1 if state == "warm_start" else length]
                sample["states"][state] = geometry(model, row, ids)
                jobs.write_json(OUTPUT / "probe.partial.json", result)
                print(exp["name"], sid, state, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    jobs.write_json(OUTPUT / "probe.json", result)
    plot()
    (OUTPUT / "completed").touch()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUTPUT / "probe.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    summary = {}
    groups = []
    for name, model in data["models"].items():
        for state in STATES:
            rows = [s["states"][state] for s in model["samples"]]
            groups.append((name, state, rows))
    for x, (name, state, rows) in enumerate(groups):
        ratio = [s["new_gradient_free_ratio"] for s in rows]
        conflict = [s["negative_cosine_fraction"] for s in rows]
        step = [s["required_linear_step_l2"] / s["memory_norm"] for s in rows if s["required_linear_step_l2"] is not None]
        summary[f"{name}/{state}"] = dict(
            mean_free_ratio=float(np.mean(ratio)),
            mean_negative_cosine_fraction=float(np.mean(conflict)),
            median_required_relative_step=float(np.median(step)) if step else None,
            mean_new_margin=float(np.mean([s["new_margin"] for s in rows])),
        )
        for ax, values in zip(axes, (ratio, conflict, step)):
            ax.scatter(np.full(len(values), x), values, alpha=0.65)
            if values:
                ax.plot([x - 0.2, x + 0.2], [np.median(values)] * 2, color="black")
    for ax, title in zip(
        axes,
        (
            "New-token gradient outside old constraint span",
            "Conflicting old-token margin directions",
            "Linear step to positive new margin / memory norm",
        ),
    ):
        ax.set(title=title)
        ax.set_xticks(range(len(groups)), [f"{n}\n{s}" for n, s, _ in groups], rotation=15, fontsize=8)
    fig.suptitle("Local margin geometry; same target prefix at SPS first failure; FP32, batch 1")
    fig.tight_layout()
    fig.savefig(OUTPUT / "margin_geometry.png", dpi=180)
    plt.close(fig)
    jobs.write_json(OUTPUT / "summary.json", summary)


def submit(dry=False):
    jobs.OUTPUT = OUTPUT
    jobs.submit(dry=dry, entrypoint="scripts/analyze_sps_margin_geometry.py", job_kind="sps_margin_geometry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["probe", "plot", "dry", "submit"], default="dry")
    args = parser.parse_args()
    {"probe": probe, "plot": plot, "dry": lambda: submit(True), "submit": submit}[args.phase]()
