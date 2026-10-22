"""Causal stream routes, K/V counterfactuals, and query context sensitivity."""

import argparse
import gc

import numpy as np
import torch
from scripts import analyze_sps_cramming as jobs
from scripts.analyze_sps_memory_ablation import score
from scripts.sps_mechanism_common import (
    converged,
    experiment_configs,
    inputs,
    load_model,
    memory_tensor,
    saved_inputs,
    source_commit,
)

from compression_horizon.analysis.native_memory_interventions import (
    capture_memory_kv,
    capture_readout_queries,
    patch_memory_kv,
    route_knockout,
)

OUTPUT = jobs.ROOT / "mechanism_routes_20261022"
SAMPLE_IDS = tuple(range(16))
TARGET_LENGTH = 12
PATCH_KINDS = ("key", "value", "both")


def probe():
    torch.set_num_threads(8)
    groups, cache = saved_inputs()
    report = dict(source_commit=source_commit(), target_length=TARGET_LENGTH, sample_ids=list(SAMPLE_IDS), models={})
    for exp in experiment_configs:
        group = groups[exp["name"]]
        for sid in SAMPLE_IDS:
            assert converged(group["stages"][sid][TARGET_LENGTH])
        model = load_model(group["design"])
        n_layers = len(model.transformer.h)
        record = dict(run=str(exp["run"]), samples=[])
        report["models"][exp["name"]] = record
        for index, sid in enumerate(SAMPLE_IDS):
            donor_id = SAMPLE_IDS[(index + 1) % len(SAMPLE_IDS)]
            row = group["stages"][sid][TARGET_LENGTH]
            donor_row = group["stages"][donor_id][TARGET_LENGTH]
            memory = memory_tensor(row, model)
            donor_memory = memory_tensor(donor_row, model)
            donor_memory = donor_memory * (memory.norm() / donor_memory.norm().clamp_min(1e-12))
            ids = torch.tensor([cache[sid]["input_ids"][:TARGET_LENGTH]], device=model.device)
            donor_ids = torch.tensor([cache[donor_id]["input_ids"][:TARGET_LENGTH]], device=model.device)
            original = inputs(model, memory, ids)
            baseline = score(model, memory, ids)
            own_kv = capture_memory_kv(model, original)
            donor_kv = capture_memory_kv(model, inputs(model, donor_memory, ids))
            qa = capture_readout_queries(model, original)
            qb = capture_readout_queries(model, inputs(model, memory, donor_ids))
            query_change = [
                float((qa[i][:, 1:] - qb[i][:, 1:]).norm() / qa[i][:, 1:].norm().clamp_min(1e-12)) for i in range(n_layers)
            ]
            sample = dict(
                sample_id=sid,
                donor_id=donor_id,
                baseline=baseline,
                query_context_change=query_change,
                knockout={},
                patch={},
                controls={},
            )
            record["samples"].append(sample)
            for kind in PATCH_KINDS:
                with patch_memory_kv(model, range(n_layers), own_kv, kind):
                    own_score = score(model, memory, ids)
                assert max(abs(a - b) for a, b in zip(own_score["nll"], baseline["nll"])) < 1e-4
                sample["controls"][f"self_patch_{kind}"] = own_score
            sample["controls"]["donor_input"] = score(model, donor_memory, ids)
            for route in ("all", "context", "predict") if model.config.sps else ("all",):
                effects = []
                sample["knockout"][route] = effects
                for selected in ([i] for i in range(n_layers)):
                    with route_knockout(model, selected, route):
                        effects.append(score(model, memory, ids))
                with route_knockout(model, range(n_layers), route):
                    sample["controls"][f"all_layers_{route}"] = score(model, memory, ids)
            for kind in PATCH_KINDS:
                effects = []
                sample["patch"][kind] = effects
                for i in range(n_layers):
                    with patch_memory_kv(model, [i], donor_kv, kind):
                        effects.append(score(model, memory, ids))
                with patch_memory_kv(model, range(n_layers), donor_kv, kind):
                    sample["controls"][f"all_layers_{kind}"] = score(model, memory, ids)
            jobs.write_json(OUTPUT / "probe.partial.json", report)
            print(exp["name"], sid, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    jobs.write_json(OUTPUT / "probe.json", report)
    plot()
    (OUTPUT / "completed").touch()


def plot():
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUTPUT / "probe.json").read_text())
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, model in data["models"].items():
        rows = model["samples"]
        base = [np.mean(s["baseline"]["nll"][1:]) for s in rows]
        for route in rows[0]["knockout"]:
            a = np.array([[np.mean(v["nll"][1:]) - b for v in s["knockout"][route]] for s, b in zip(rows, base)])
            label = f"{name}: {route}"
            axes[0].plot(np.arange(1, a.shape[1] + 1), a.mean(0), label=label)
            summary[f"{name}/route/{route}"] = a.mean(0).tolist()
        query = np.array([s["query_context_change"] for s in rows])
        axes[1].plot(np.arange(1, query.shape[1] + 1), query.mean(0), label=name)
        summary[f"{name}/query_change"] = query.mean(0).tolist()
        kinds = list(rows[0]["patch"])
        for kind in kinds:
            a = np.array([[np.mean(v["nll"][1:]) - b for v in s["patch"][kind]] for s, b in zip(rows, base)])
            axes[2].plot(np.arange(1, a.shape[1] + 1), a.mean(0), label=f"{name}: {kind}")
            summary[f"{name}/patch/{kind}"] = a.mean(0).tolist()
        summary[f"{name}/controls"] = {
            c: float(np.mean([np.mean(s["controls"][c]["nll"][1:]) - b for s, b in zip(rows, base)]))
            for c in rows[0]["controls"]
        }
    for ax in axes:
        ax.set_xlabel("Layer (1-based)")
        ax.legend(fontsize=7)
        ax.axhline(0, color="gray", lw=0.5)
    axes[0].set(title="Remove memory-key route", ylabel="Change in NLL, tokens 2..L")
    axes[1].set(title="Swap ordinary context; same memory", ylabel="Relative readout-query change")
    axes[2].set(title="Patch memory K/V from norm-matched donor", ylabel="Change in NLL, tokens 2..L")
    fig.suptitle("Matched exact-stage memories; frozen models, FP32 on BF16-quantized weights; batch 1")
    fig.tight_layout()
    fig.savefig(OUTPUT / "mechanism_routes.png", dpi=180)
    plt.close(fig)
    jobs.write_json(OUTPUT / "summary.json", summary)


def submit(dry=False):
    jobs.OUTPUT = OUTPUT
    jobs.submit(dry=dry, entrypoint="scripts/analyze_sps_memory_routes.py", job_kind="sps_memory_routes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["probe", "plot", "dry", "submit"], default="dry")
    args = parser.parse_args()
    {"probe": probe, "plot": plot, "dry": lambda: submit(True), "submit": submit}[args.phase]()
