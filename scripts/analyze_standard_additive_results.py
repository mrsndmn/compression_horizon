"""Audit the completed additive comparison and probe saved vectors without training.

All cohort/training choices come from analyze_standard_additive. Optional CPU
probes change only the history injection at fixed failed-stage vectors. Their
FP32 arithmetic over the original BF16 weights is explicitly separate from
the GPU reconstruction measurements.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scripts import analyze_standard_additive as training

experiment_configs = [
    dict(name="prefix_only_same_vector", history_scale=0.0),
    dict(name="weaker_addition", history_scale=0.1),
    dict(name="trained_addition", history_scale=1.0),
    dict(name="constant_inputs", history_scale=None),
]


def analyze(output=None, probe=False):
    merged = training.ROOT / "merged"
    if not (merged / "completed").exists():
        raise ValueError("Merge the successfully completed training shards first")
    output = merged if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((merged / "single_sample_validation.json").read_text())
    control = json.loads((training.BASELINE / "single_sample_validation.json").read_text())
    cache = load_from_disk(training.resolved_training_args(training.experiment_configs[0])["tokenized_dataset_path"])
    samples = training.checked_samples(report, range(training.COHORT_SIZE), cache)
    controls = training.checked_samples(control, range(training.COHORT_SIZE), cache)
    raw = load_from_disk(merged / "additive_prefixes")
    fields = [
        "sample_id",
        "stage_index",
        "initialization_embedding",
        "embedding",
        "steps_taken",
        "final_loss",
        "final_convergence",
        "stage_seq_len",
    ]
    originals = load_from_disk(training.BASELINE / "progressive_prefixes").select_columns(fields)
    first_controls = {r["sample_id"]: r for r in originals if r["stage_index"] == 0}
    grouped = {sid: [] for sid in range(training.COHORT_SIZE)}
    for row in raw.select_columns(fields):
        grouped[row["sample_id"]].append(row)
    records, terminal = [], {}
    for sample, base in zip(samples, controls):
        sid = sample["sample_id"]
        rows = sorted(grouped[sid], key=lambda r: r["stage_index"])
        first, last = rows[0], rows[-1]
        terminal[sid] = last
        records.append(
            dict(
                sample_id=sid,
                baseline_horizon=base["horizon"],
                additive_horizon=sample["horizon"],
                horizon_difference=sample["horizon"] - base["horizon"],
                baseline_greedy_prefix=training.leading_correct(base["target_ids"], base["generated_ids"]),
                additive_greedy_prefix=training.leading_correct(sample["target_ids"], sample["generated_ids"]),
                first_vector_equal=bool(np.array_equal(first["embedding"], first_controls[sid]["embedding"])),
                first_steps_equal=first["steps_taken"] == first_controls[sid]["steps_taken"],
                first_loss_equal=first["final_loss"] == first_controls[sid]["final_loss"],
                final_length=last["stage_seq_len"],
                total_updates=last["steps_taken"],
                last_stage_updates=last["steps_taken"] - (rows[-2]["steps_taken"] if len(rows) > 1 else 0),
                initial_norm=float(np.linalg.norm(first["initialization_embedding"])),
                first_norm=float(np.linalg.norm(first["embedding"])),
                final_norm=float(np.linalg.norm(last["embedding"])),
            )
        )
    differences = [r["horizon_difference"] for r in records]
    summary = dict(
        training_commit=report["source_commit"],
        analysis_commit=training.source_commit(),
        n_samples=len(samples),
        raw_stages=len(raw),
        horizon_histogram=dict(sorted(Counter(s["horizon"] for s in samples).items())),
        wins=sum(d > 0 for d in differences),
        ties=sum(d == 0 for d in differences),
        losses=sum(d < 0 for d in differences),
        first_stage_equal={
            key: sum(r[key] for r in records) for key in ("first_vector_equal", "first_steps_equal", "first_loss_equal")
        },
        last_stage_updates=dict(Counter(r["last_stage_updates"] for r in records)),
        mean_total_updates=float(np.mean([r["total_updates"] for r in records])),
        mean_final_norm=float(np.mean([r["final_norm"] for r in records])),
        mean_horizon=float(np.mean([s["horizon"] for s in samples])),
        baseline_horizon=float(np.mean([s["horizon"] for s in controls])),
        mean_greedy_prefix=float(np.mean([r["additive_greedy_prefix"] for r in records])),
        baseline_greedy_prefix=float(np.mean([r["baseline_greedy_prefix"] for r in records])),
        greedy_exact=report["greedy_exact_samples"],
        baseline_greedy_exact=control["greedy_exact_samples"],
        teacher_exact=report["teacher_replay_exact_samples"],
        paired_horizon=training.paired_interval(differences),
    )
    authoritative = json.loads((merged / "summary.json").read_text())
    assert summary["mean_horizon"] == authoritative["mean_horizon"]
    assert summary["paired_horizon"] == authoritative["paired_horizon_gain"]
    with (output / "per_sample.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    training.jobs.write_json(output / "paired_analysis.json", summary)
    plot(summary, records, output / "paired_comparison.png")
    if probe:
        probe_vectors(terminal, cache, output)
    print(json.dumps(summary, indent=2))
    return summary


@torch.no_grad()
def probe_vectors(terminal, cache, output):
    torch.set_num_threads(2)
    model = training.load_sps_checkpoint(training.CHECKPOINT, torch_dtype=torch.bfloat16).float()
    records = []
    for sid, row in sorted(terminal.items()):
        ids = torch.tensor([cache[sid]["input_ids"][: row["stage_seq_len"]]])
        z = torch.tensor([row["embedding"]], dtype=torch.float32)
        tokens = model.get_input_embeddings()(ids)
        norm_ratio = float(z.norm() / tokens.norm(dim=-1).mean())
        for config in experiment_configs:
            alpha = config["history_scale"]
            inputs = z.expand(1, ids.shape[1] + 1, -1) if alpha is None else torch.cat((z, tokens + alpha * z), dim=1)
            out = model(inputs_embeds=inputs, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, :-1]
            logits = out.logits[:, :-1].float()
            predicted = logits.argmax(-1)
            relative_variation = (hidden - hidden.mean(dim=1, keepdim=True)).norm() / hidden.norm()
            records.append(
                dict(
                    sample_id=sid,
                    condition=config["name"],
                    norm_ratio=norm_ratio,
                    accuracy=float((predicted == ids).float().mean()),
                    distinct_predictions=int(predicted.unique().numel()),
                    hidden_relative_variation=float(relative_variation),
                    input_cosine=float(torch.nn.functional.cosine_similarity(inputs[:, 1:-1], inputs[:, :1], dim=-1).mean()),
                )
            )
    summary = {
        config["name"]: {
            field: float(np.mean([r[field] for r in records if r["condition"] == config["name"]]))
            for field in ("norm_ratio", "accuracy", "distinct_predictions", "hidden_relative_variation", "input_cosine")
        }
        for config in experiment_configs
    }
    training.jobs.write_json(
        output / "mechanism_probe.json",
        dict(
            analysis_commit=training.source_commit(),
            checkpoint=str(training.CHECKPOINT),
            arithmetic="CPU FP32 over the original BF16 weights and saved BF16 vectors; no optimization",
            scope="48 terminal failed stages; fixed vectors, changed history injection only",
            norm_ratio_definition="Unscaled ||z|| / mean target-token embedding norm, including the final target",
            hidden_variation_definition="||H - mean_positions(H)||_F / ||H||_F over target readout positions",
            experiment_configs=experiment_configs,
            summary=summary,
            samples=records,
        ),
    )
    print(json.dumps(summary, indent=2))


def plot(summary, records, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    x = np.arange(2)
    for shift, values, color, label in [
        (-0.18, [summary["baseline_horizon"], summary["baseline_greedy_prefix"]], "#3978ac", "Prefix only"),
        (0.18, [summary["mean_horizon"], summary["mean_greedy_prefix"]], "#d77d35", "Prefix + additive"),
    ]:
        bars = axes[0].bar(x + shift, values, 0.36, color=color, label=label)
        axes[0].bar_label(bars, fmt="%.3f", padding=3)
    height = 1.4 * max(
        summary["baseline_horizon"], summary["baseline_greedy_prefix"], summary["mean_horizon"], summary["mean_greedy_prefix"]
    )
    axes[0].set(xticks=x, xticklabels=["Teacher-forced", "Greedy initial prefix"], ylabel="Mean tokens", ylim=(0, height))
    axes[0].legend(frameon=False)
    axes[1].scatter(
        [r["baseline_horizon"] for r in records], [r["additive_horizon"] for r in records], color="#d77d35", s=22, alpha=0.7
    )
    maximum = max(r["baseline_horizon"] for r in records) + 5
    axes[1].plot([0, maximum], [0, maximum], "--", color=".55", lw=1)
    axes[1].set(
        xlabel="Prefix-only teacher horizon",
        ylabel="Additive teacher horizon",
        xlim=(0, maximum),
        ylim=(0, maximum),
        title=f"Lower horizon on {summary['losses']}/{summary['n_samples']} texts",
    )
    counts = summary["horizon_histogram"]
    bars = axes[2].bar([str(k) for k in counts], list(counts.values()), color="#d77d35")
    axes[2].bar_label(bars, padding=3)
    axes[2].set(xlabel="Additive teacher horizon (tokens)", ylabel="Texts", ylim=(0, max(counts.values()) * 1.15))
    fig.suptitle(f"Standard: one vector, same {summary['n_samples']} texts, paired initialization, matched hinge budget")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Read-only CPU analysis at saved failed vectors")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    analyze(args.output_dir, args.probe)
