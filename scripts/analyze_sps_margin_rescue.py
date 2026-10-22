"""Test whether first failed SPS prefixes are recoverable with unchanged memory size.

Diagnostic restarts, not resumption of missing Adam state. Per-arm VJP budgets
match; forward counts and elapsed times are recorded separately. FP32 solutions
are validated in FP32 and in actual BF16 teacher forcing and greedy decoding.
"""

import argparse
import json
import time

import numpy as np
import torch
from scipy.optimize import minimize
from scripts import analyze_sps_cramming as jobs
from scripts.analyze_sps_margin_geometry import TARGET_MARGIN, target_margins
from scripts.sps_mechanism_common import converged, inputs, load_model, memory_tensor, saved_inputs, source_commit
from scripts.validation.validate_sps_batch1_run import check_sample
from torch.nn import functional as F

from compression_horizon.models.sps_causal_lm import load_sps_tokenizer

OUTPUT = jobs.ROOT / "mechanism_margin_rescue_20261022"
SAMPLE_IDS = tuple(range(4))
VJP_BUDGET = 500
experiment_configs = [
    dict(name="adam_ce_lr03", method="ce", lr=0.3),
    dict(name="adam_ce_lr1", method="ce", lr=1.0),
    dict(name="adam_hinge_lr03", method="hinge", lr=0.3),
    dict(name="trust_margin", method="trust"),
]
INITIAL_RADIUS = 0.002
MIN_RADIUS = 1e-5
MAX_RADIUS = 0.01
BACKTRACKS = 6
QP_TOL = 1e-4


def minimum_norm_step(margins, jacobian):
    """Approximate min ||delta|| subject to m + J delta >= target margin.

    Solve the nonnegative dual after row normalization. Positive margins supply
    slack; they are not unnecessarily frozen. Return feasibility diagnostics so
    failed/inconsistent linearized problems are never reported as valid steps.
    """
    j = jacobian.detach().double().cpu().numpy()
    m = margins.detach().double().cpu().numpy()
    norms = np.maximum(np.linalg.norm(j, axis=1), 1e-12)
    a = j / norms[:, None]
    b = (TARGET_MARGIN - m) / norms
    gram = a @ a.T

    def objective(lam):
        product = gram @ lam
        return 0.5 * float(lam @ product) - float(b @ lam), product - b

    fit = minimize(
        objective,
        np.zeros(len(m)),
        jac=True,
        bounds=[(0, None)] * len(m),
        method="L-BFGS-B",
        options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-9},
    )
    step = a.T @ fit.x
    violation = float(np.maximum(TARGET_MARGIN - m - j @ step, 0).max())
    finite = bool(np.isfinite(step).all() and np.isfinite(violation))
    return torch.from_numpy(step), dict(
        solver_success=bool(fit.success),
        feasible=finite and violation < QP_TOL,
        max_violation=violation if finite else None,
        iterations=int(fit.nit),
    )


def rescue_memory(forward, ids, initial, config, budget=VJP_BUDGET):
    """Optimize one memory. A scalar loss backward counts as one VJP; J has L."""
    if initial.shape[0] != 1 or ids.shape[0] != 1:
        raise ValueError("Single-sample cramming only")
    started = time.perf_counter()
    memory = torch.nn.Parameter(initial.detach().clone())
    optimizer = (
        None
        if config["method"] == "trust"
        else torch.optim.AdamW([memory], lr=config["lr"], betas=(0.9, 0.9), weight_decay=0.01)
    )
    used = 0
    forwards = 0
    trace = []
    best_margin = -float("inf")
    best = initial.detach().clone()
    radius = INITIAL_RADIUS
    reason = "budget"
    qp_failures = 0

    def evaluate(point, grad=False):
        nonlocal forwards
        forwards += 1
        with torch.set_grad_enabled(grad):
            logits = forward(point)
            margins = target_margins(logits, ids)
        return logits, margins

    def remember(margins, point):
        nonlocal best_margin, best
        current = float(margins.detach().min())
        if current > best_margin:
            best_margin = current
            best = point.detach().clone()

    def merit(margins):
        return float((TARGET_MARGIN - margins.detach()).clamp_min(0).square().mean())

    length = ids.shape[1]
    while True:
        logits, margins = evaluate(memory, True)
        if not torch.isfinite(margins).all():
            if not np.isfinite(best_margin):
                raise ValueError("Initial memory has nonfinite margins")
            reason = "nonfinite_current"
            break
        remember(margins, memory)
        trace.append(
            dict(
                vjp=used,
                forwards=forwards,
                min_margin=float(margins.detach().min()),
                correct=int((logits.detach().argmax(-1) == ids).sum()),
                radius=radius,
            )
        )
        if best_margin >= TARGET_MARGIN:
            reason = "target_margin"
            break
        needed = length if config["method"] == "trust" else 1
        if used + needed > budget:
            break
        if config["method"] != "trust":
            loss = (
                F.cross_entropy(logits.flatten(0, 1), ids.flatten())
                if config["method"] == "ce"
                else (TARGET_MARGIN - margins).clamp_min(0).square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            used += 1
            continue
        jac = torch.stack(
            [torch.autograd.grad(margins[i], memory, retain_graph=i + 1 < length)[0].flatten().detach() for i in range(length)]
        )
        used += length
        delta, qp = minimum_norm_step(margins, jac)
        trace[-1]["qp"] = qp
        if not qp["feasible"]:
            qp_failures += 1
            reason = "linearized_qp_failed"
            break
        delta = delta.to(device=memory.device, dtype=memory.dtype)
        scale = float(initial.norm())
        delta = delta * min(1.0, radius * scale / max(float(delta.norm()), 1e-12))
        old = memory.detach().clone()
        old_merit = merit(margins)
        accepted = False
        for k in range(BACKTRACKS):
            step = delta * (0.5**k)
            candidate = old + step.view_as(old)
            _, actual = evaluate(candidate)
            remember(actual, candidate)
            if best_margin >= TARGET_MARGIN:
                reason = "target_margin"
                accepted = True
                with torch.no_grad():
                    memory.copy_(best)
                break
            predicted = margins.detach() + jac @ step
            predicted_gain = old_merit - merit(predicted)
            actual_gain = old_merit - merit(actual)
            ratio = actual_gain / max(predicted_gain, 1e-12)
            if actual_gain > 0 and ratio > 0.05:
                with torch.no_grad():
                    memory.copy_(candidate)
                radius = (
                    min(MAX_RADIUS, radius * 1.5)
                    if ratio > 0.75 and k == 0
                    else max(MIN_RADIUS, radius * 0.5) if ratio < 0.25 or k > 0 else radius
                )
                accepted = True
                break
        if not accepted:
            radius *= 0.5
            if radius < MIN_RADIUS:
                reason = "trust_radius_exhausted"
                break
    return dict(
        memory=best,
        best_min_margin=best_margin,
        gradient_evaluations=used,
        model_forwards=forwards,
        elapsed_seconds=time.perf_counter() - started,
        reason=reason,
        qp_failures=qp_failures,
        trace=trace,
    )


def probe():
    torch.set_num_threads(8)
    groups, cache = saved_inputs()
    group = groups["sps_w0"]
    model = load_model(group["design"])
    tokenizer = load_sps_tokenizer()
    result = dict(source_commit=source_commit(), source_run=group["design"], vjp_budget=VJP_BUDGET, samples=[])
    for sid in SAMPLE_IDS:
        length, row = next((n, r) for n, r in sorted(group["stages"][sid].items()) if not converged(r))
        ids = torch.tensor([cache[sid]["input_ids"][:length]], device=model.device)
        initial = memory_tensor(row, model)
        sample = dict(sample_id=sid, target_length=length, stored_accuracy=float(row["final_convergence"]), arms={})
        result["samples"].append(sample)
        with torch.no_grad():
            initial_logits = model(inputs_embeds=inputs(model, initial, ids)).logits[:, :-1]
            sample["initial_correct_tokens"] = int((initial_logits.argmax(-1) == ids).sum())
        model.to(dtype=torch.bfloat16)
        sample["initial_bf16_validation"] = check_sample(model, tokenizer, ids, initial.to(torch.bfloat16))
        assert abs(sample["initial_bf16_validation"]["teacher_replay_accuracy"] - sample["stored_accuracy"]) < 1e-6
        model.float()
        for config in experiment_configs:
            outcome = rescue_memory(lambda e: model(inputs_embeds=inputs(model, e, ids)).logits[:, :-1], ids, initial, config)
            best = outcome.pop("memory")
            outcome["memory"] = best[0].cpu().tolist()
            outcome["fp32_validation"] = check_sample(model, tokenizer, ids, best)
            model.to(dtype=torch.bfloat16)
            outcome["bf16_validation"] = check_sample(model, tokenizer, ids, best.to(torch.bfloat16))
            model.float()
            sample["arms"][config["name"]] = outcome
            print(sid, config["name"], outcome["best_min_margin"], outcome["bf16_validation"]["greedy_exact"], flush=True)
            jobs.write_json(OUTPUT / "probe.partial.json", result)
    jobs.write_json(OUTPUT / "probe.json", result)
    plot()
    (OUTPUT / "completed").touch()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUTPUT / "probe.json").read_text())
    fig, axes = plt.subplots(1, len(data["samples"]), figsize=(15, 4), sharey=True)
    summary = {}
    for ax, sample in zip(np.atleast_1d(axes), data["samples"]):
        for name, out in sample["arms"].items():
            ax.plot([r["vjp"] for r in out["trace"]], [r["min_margin"] for r in out["trace"]], label=name)
        ax.axhline(0, color="gray", lw=0.7)
        ax.axhline(TARGET_MARGIN, color="gray", lw=0.7, ls=":")
        ax.set(title=f"Sample {sample['sample_id']}, L={sample['target_length']}", xlabel="VJP evaluations")
    axes[0].set_ylabel("Worst target-vs-competitor margin")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Restart from failed SPS memory; same dimension and frozen checkpoint, batch 1")
    fig.tight_layout()
    fig.savefig(OUTPUT / "margin_rescue.png", dpi=180)
    plt.close(fig)
    for config in experiment_configs:
        outcomes = [s["arms"][config["name"]] for s in data["samples"]]
        summary[config["name"]] = {
            f"{precision}_{metric}": sum(
                o[f"{precision}_validation"][metric] == (1.0 if metric == "teacher_replay_accuracy" else True) for o in outcomes
            )
            for precision in ["fp32", "bf16"]
            for metric in ["teacher_replay_accuracy", "greedy_exact"]
        }
    jobs.write_json(OUTPUT / "summary.json", summary)


def submit(dry=False):
    jobs.OUTPUT = OUTPUT
    jobs.submit(dry=dry, entrypoint="scripts/analyze_sps_margin_rescue.py", job_kind="sps_margin_rescue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["probe", "plot", "dry", "submit"], default="dry")
    args = parser.parse_args()
    {"probe": probe, "plot": plot, "dry": lambda: submit(True), "submit": submit}[args.phase]()
