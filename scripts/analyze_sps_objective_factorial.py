"""Validate and summarize the completed native objective/convergence factorial.

No model inference or submission. All run selection/settings come from the
training registry. Text-paired bootstrap intervals describe this fixed cohort,
not seed variance or an architecture-only effect. Greedy prefix length is for
one saved final memory, not a search over earlier stages or a greedy horizon.
"""

import argparse
import csv
import json

import numpy as np
from scripts.jobs.run_jobs_sps import (
    ARTIFACTS,
    OBJECTIVE_BASELINES,
    TRAINING_ARGS,
    experiment_configs,
    objective_experiment_configs,
)
from scripts.sps_mechanism_common import source_commit

OUTPUT = ARTIFACTS / "objective_factorial_analysis_20261022"
CELLS = ("ce_raw", "ce_strict", "hinge_raw", "hinge_strict")
LABELS = ("CE / argmax", "CE / margin", "Hinge / argmax", "Hinge / margin")
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 42


def leading_correct(target, generated):
    for i, (a, b) in enumerate(zip(target, generated)):
        if a != b:
            return i
    return min(len(target), len(generated))


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist())


def validate_samples(report, expected):
    samples = sorted(report["samples"], key=lambda s: s["sample_id"])
    if report["n_samples"] != expected or [s["sample_id"] for s in samples] != list(range(expected)):
        raise ValueError("Missing or duplicated sample IDs")
    for sample in samples:
        target = sample["target_ids"]
        generated = sample["generated_ids"]
        if not target or sample["horizon"] != len(target):
            raise ValueError("Expected a nonempty converged saved prefix")
        correct = sum(a == b for a, b in zip(target, generated))
        exact = target == generated
        if sample["greedy_exact"] != exact or not np.isclose(sample["greedy_accuracy"], correct / len(target)):
            raise ValueError("Greedy metrics disagree with exact IDs")
        tf_correct = sample.get("teacher_replay_correct_tokens", round(sample["teacher_replay_accuracy"] * len(target)))
        if tf_correct != len(target) or not np.isclose(sample["teacher_replay_accuracy"], 1):
            raise ValueError("Selected teacher-forced prefix does not replay exactly")
        sample["greedy_initial_prefix"] = leading_correct(target, generated)
    if not np.isclose(report["mean_horizon"], np.mean([s["horizon"] for s in samples])):
        raise ValueError("Reported horizon mean mismatch")
    if report["greedy_exact_samples"] != sum(s["greedy_exact"] for s in samples):
        raise ValueError("Reported exact count mismatch")
    return samples


def load_runs():
    from datasets import load_from_disk

    cache = load_from_disk(TRAINING_ARGS["tokenized_dataset_path"])
    runs = {}
    for base in experiment_configs:
        model = "standard" if "standard" in base["name"] else "sps_w0"
        cells = [("ce_raw", OBJECTIVE_BASELINES[base["name"]], dict(learning_rate=0.3))]
        for exp in objective_experiment_configs:
            if exp["checkpoint"] != base["checkpoint"]:
                continue
            label = exp["name"].removeprefix(base["name"] + "_")
            candidates = [
                p
                for p in ARTIFACTS.glob(exp["name"] + "_*")
                if (p / "design.json").exists()
                and json.loads((p / "design.json").read_text())["experiment"]["name"] == exp["name"]
            ]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous or missing run for {exp['name']}")
            cells.append((label, candidates[0], exp["training_args"]))
        if {c[0] for c in cells} != set(CELLS):
            raise ValueError("Incomplete factorial")
        for label, path, overrides in cells:
            if not (path / "completed").exists():
                raise ValueError(f"Run not completed: {path}")
            design = json.loads((path / "design.json").read_text())
            expected_settings = dict(TRAINING_ARGS, **overrides)
            if design["training_args"] != expected_settings or design["experiment"]["checkpoint"] != base["checkpoint"]:
                raise ValueError(f"Mismatched run design: {path}")
            report = json.loads((path / "single_sample_validation.json").read_text())
            if report["training_commit"] != design["commit"]:
                raise ValueError("Validation commit mismatch")
            samples = validate_samples(report, expected_settings["limit_dataset_items"])
            for s in samples:
                if s["target_ids"] != cache[s["sample_id"]]["input_ids"][: s["horizon"]]:
                    raise ValueError("Target differs from pinned exact token IDs")
            runs[(model, label)] = dict(path=str(path), design=design, samples=samples)
    return runs


def analyze():
    runs = load_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = dict(
        analysis_commit=source_commit(), intervals="paired text bootstrap; fixed checkpoints and seed", runs=[], contrasts={}
    )
    vectors = {}
    for (model, cell), run in runs.items():
        samples = run["samples"]
        horizon = np.array([s["horizon"] for s in samples])
        prefix = np.array([s["greedy_initial_prefix"] for s in samples])
        vectors[(model, cell)] = horizon
        summary["runs"].append(
            dict(
                model=model,
                cell=cell,
                path=run["path"],
                training_commit=run["design"]["commit"],
                mean_horizon=paired_interval(horizon),
                mean_greedy_initial_prefix=paired_interval(prefix),
                greedy_exact_samples=sum(s["greedy_exact"] for s in samples),
                mean_greedy_accuracy=float(np.mean([s["greedy_accuracy"] for s in samples])),
                n_samples=len(samples),
            )
        )
    for cell in CELLS:
        gap = vectors[("standard", cell)] - vectors[("sps_w0", cell)]
        summary["contrasts"][f"gap_{cell}"] = paired_interval(gap)
        summary["contrasts"][f"gap_change_{cell}"] = paired_interval(
            gap - (vectors[("standard", "ce_raw")] - vectors[("sps_w0", "ce_raw")])
        )
    for model in ("standard", "sps_w0"):
        for cell in CELLS[1:]:
            summary["contrasts"][f"{model}_{cell}_minus_ce_raw"] = paired_interval(
                vectors[(model, cell)] - vectors[(model, "ce_raw")]
            )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with (OUTPUT / "per_sample.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "cell", "sample_id", "teacher_horizon", "greedy_exact", "greedy_initial_prefix"])
        for (model, cell), run in runs.items():
            writer.writerows(
                (model, cell, s["sample_id"], s["horizon"], s["greedy_exact"], s["greedy_initial_prefix"])
                for s in run["samples"]
            )
    plot(summary)
    print(json.dumps(summary, indent=2))


def plot(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(CELLS))
    for model, label, color, offset in (("standard", "Standard", "tab:blue", -0.18), ("sps_w0", "SPS w=0", "tab:orange", 0.18)):
        rows = [next(r for r in summary["runs"] if r["model"] == model and r["cell"] == cell) for cell in CELLS]
        for axis, metric in ((axes[0], "mean_horizon"), (axes[2], "mean_greedy_initial_prefix")):
            means = np.array([r[metric]["mean"] for r in rows])
            ci = np.array([r[metric]["ci95"] for r in rows]).T
            axis.bar(
                x + offset,
                means,
                width=0.36,
                color=color,
                label=label,
                yerr=np.maximum(np.stack((means - ci[0], ci[1] - means)), 0),
                capsize=3,
            )
        counts = [r["greedy_exact_samples"] for r in rows]
        bars = axes[1].bar(x + offset, [n / r["n_samples"] for n, r in zip(counts, rows)], width=0.36, color=color, label=label)
        axes[1].bar_label(bars, labels=[f"{n}/48" for n in counts], fontsize=8, padding=3)
    axes[0].set(title="Teacher-forced reconstruction length", ylabel="Mean tokens (95% text bootstrap CI)")
    axes[1].set(title="BF16 greedy exact at saved final prefix", ylabel="Fraction of texts", ylim=(0, 1.17))
    axes[2].set(title="Actual initial correct greedy prefix", ylabel="Mean tokens for saved final memory")
    for ax in axes:
        ax.set_xticks(x, LABELS, rotation=20, ha="right")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Same 48 texts, native checkpoints, batch 1; checkpoint/optimizer settings in training registry\nGreedy metrics evaluate one final memory per text; they are not maximal greedy horizons",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "objective_factorial.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    analyze()
