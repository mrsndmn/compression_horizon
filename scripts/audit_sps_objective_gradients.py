"""Read-only audit of CE preferences and gradient cancellation at rescue points."""

import argparse
import json

import numpy as np
import torch
from scripts import analyze_sps_cramming as jobs
from scripts.analyze_sps_margin_geometry import TARGET_MARGIN, target_margins
from scripts.sps_mechanism_common import inputs, load_model, memory_tensor, saved_inputs, source_commit
from torch.nn import functional as F

OUTPUT = jobs.ROOT / "mechanism_objective_gradients_20261022"
SOURCE = jobs.ROOT / "mechanism_margin_rescue_20261022" / "probe.json"
POINTS = ("initial", "adam_ce_lr03", "adam_ce_lr1", "adam_hinge_lr03", "trust_margin")


def gradient_decomposition(logits, memory, ids):
    losses = F.cross_entropy(logits.float().flatten(0, 1), ids.flatten(), reduction="none")
    wrong = (logits.detach().argmax(-1) != ids).flatten()
    margins = target_margins(logits, ids)
    count = losses.numel()
    ce = losses.mean()
    wrong_loss = (losses * wrong).sum() / count
    correct_loss = (losses * (~wrong)).sum() / count
    hinge = (TARGET_MARGIN - margins).clamp_min(0).square().mean()

    def grad(loss):
        return torch.autograd.grad(loss, memory, retain_graph=True)[0].detach().flatten().double()

    all_g = grad(ce)
    wrong_g = grad(wrong_loss)
    correct_g = grad(correct_loss)
    hinge_g = grad(hinge)
    worst = torch.autograd.grad(margins[margins.detach().argmin()], memory)[0].detach().flatten().double()

    def cosine(a, b):
        denom = float(a.norm() * b.norm())
        return float(a @ b) / denom if denom > 1e-12 else None

    return dict(
        ce=float(ce.detach()),
        wrong_tokens=int(wrong.sum()),
        length=count,
        min_margin=float(margins.detach().min()),
        correct_ce=float(correct_loss.detach()),
        wrong_ce=float(wrong_loss.detach()),
        total_gradient_norm=float(all_g.norm()),
        correct_gradient_norm=float(correct_g.norm()),
        wrong_gradient_norm=float(wrong_g.norm()),
        gradient_additivity_error=float((all_g - correct_g - wrong_g).norm()),
        correct_wrong_gradient_cosine=cosine(correct_g, wrong_g),
        cancellation_ratio=float(all_g.norm()) / max(float(correct_g.norm() + wrong_g.norm()), 1e-12),
        ce_descent_worst_margin_cosine=cosine(-all_g, worst),
        hinge_descent_worst_margin_cosine=cosine(-hinge_g, worst),
    )


def probe():
    torch.set_num_threads(8)
    saved = json.loads(SOURCE.read_text())
    groups, cache = saved_inputs()
    group = groups["sps_w0"]
    model = load_model(group["design"])
    report = dict(source_commit=source_commit(), source=str(SOURCE), samples=[])
    for sample in saved["samples"]:
        sid = sample["sample_id"]
        length = sample["target_length"]
        row = group["stages"][sid][length]
        ids = torch.tensor([cache[sid]["input_ids"][:length]], device=model.device)
        record = dict(sample_id=sid, target_length=length, points={})
        report["samples"].append(record)
        for point in POINTS:
            memory = (
                memory_tensor(row, model)
                if point == "initial"
                else torch.tensor([sample["arms"][point]["memory"]], device=model.device, dtype=model.dtype)
            )
            memory = memory.detach().requires_grad_(True)
            logits = model(inputs_embeds=inputs(model, memory, ids)).logits[:, :-1]
            record["points"][point] = gradient_decomposition(logits, memory, ids)
        jobs.write_json(OUTPUT / "probe.partial.json", report)
    jobs.write_json(OUTPUT / "probe.json", report)
    plot()
    (OUTPUT / "completed").touch()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUTPUT / "probe.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = dict(zip(POINTS, ["gray", "tab:blue", "tab:orange", "tab:green", "tab:red"]))
    for point in POINTS:
        rows = [s["points"][point] for s in data["samples"]]
        axes[0].scatter([r["wrong_tokens"] for r in rows], [r["ce"] for r in rows], label=point, color=colors[point])
    x = np.arange(len(data["samples"]))
    for point, offset in [("initial", -0.2), ("adam_ce_lr03", 0), ("adam_ce_lr1", 0.2)]:
        rows = [s["points"][point] for s in data["samples"]]
        axes[1].bar(x + offset, [r["cancellation_ratio"] for r in rows], width=0.2, label=point, color=colors[point])
        axes[2].bar(
            x + offset, [r["ce_descent_worst_margin_cosine"] or 0 for r in rows], width=0.2, label=point, color=colors[point]
        )
    axes[0].set(xlabel="Wrong target tokens", ylabel="Mean cross entropy", title="Does CE prefer an incorrect solution?")
    axes[1].set(
        xlabel="Sample index", ylabel="||g_all|| / (||g_correct||+||g_wrong||)", title="Cancellation between token groups"
    )
    axes[2].set(
        xlabel="Sample index", ylabel="Cosine with worst-margin gradient", title="CE descent toward improving worst token"
    )
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Fixed SPS checkpoint and memory dimension; FP32; read-only saved-solution audit")
    fig.tight_layout()
    fig.savefig(OUTPUT / "objective_gradient_audit.png", dpi=180)
    plt.close(fig)
    summary = {}
    for point in POINTS:
        rows = [s["points"][point] for s in data["samples"]]
        summary[point] = {k: float(np.mean([r[k] for r in rows])) for k in ["ce", "wrong_tokens", "cancellation_ratio"]}
    summary["hinge_perfect_but_higher_ce_than_incorrect_ce03"] = sum(
        s["points"]["adam_hinge_lr03"]["wrong_tokens"] == 0
        and s["points"]["adam_ce_lr03"]["wrong_tokens"] > 0
        and s["points"]["adam_hinge_lr03"]["ce"] > s["points"]["adam_ce_lr03"]["ce"]
        for s in data["samples"]
    )
    jobs.write_json(OUTPUT / "summary.json", summary)


def submit(dry=False):
    jobs.OUTPUT = OUTPUT
    jobs.submit(dry=dry, entrypoint="scripts/audit_sps_objective_gradients.py", job_kind="sps_objective_gradients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["probe", "plot", "dry", "submit"], default="dry")
    args = parser.parse_args()
    {"probe": probe, "plot": plot, "dry": lambda: submit(True), "submit": submit}[args.phase]()
