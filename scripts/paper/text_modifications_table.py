import argparse
import glob
import json
import os
from types import SimpleNamespace

import numpy as np
from datasets import load_from_disk
from scripts.results.results import (
    aggregate_non_progressive,
    aggregate_progressive,
    load_dataset_rows,
)
from sklearn.decomposition import PCA
from tabulate import tabulate
from tqdm.auto import tqdm

from compression_horizon.utils import hlines_to_booktabs, to_mean_std_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full cramming results table.")
    parser.add_argument(
        "--tablefmt",
        default="latex",
        help="Tabulate table format (e.g., plain, github, latex, grid).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_filename = "full_cramming_table_cache.json"
    cache_version = 3

    experiments_list = [
        # Llama-3.2-1B
        {"train": "progr", "type": "Base", "id": "sl_2048_Meta-Llama-3.1-8B_ds_pg19_lr_0.1"},
        {"train": "progr", "type": "Random", "id": "sl_2048_Meta-Llama-3.1-8B_ds_pg19-random-suffix-shuffle-64_lr_0.1"},  # TODO
        {
            "train": "progr",
            "type": "Sampled",
            "id": "sl_2048_Meta-Llama-3.1-8B_ds_pg19-model-sampled-llama3.1-8B-prefix-64-max_len-2048_lr_0.1",
        },
        {"train": "progr", "type": "Lowercased", "id": "sl_2048_Meta-Llama-3.1-8B_ds_pg19-lowercased-partial-64_lr_0.1"},
    ]

    columns = ["Experiment", "Tokens", "Info Gain", "PCA 99%"]

    def format_experiment_label(summary, fallback_label: str) -> str:
        parts = []
        if summary.model_checkpoint:
            parts.append(str(summary.model_checkpoint))

        label = "-".join(parts).strip()
        if not label:
            label = fallback_label

        return label

    def summary_to_cache(summary) -> dict:
        return {
            "dataset_type": summary.dataset_type,
            "model_checkpoint": summary.model_checkpoint,
            "run_hash": summary.run_hash,
            "information_gain_bits_mean": summary.information_gain_bits_mean,
            "information_gain_bits_std": summary.information_gain_bits_std,
            "final_convergence_mean": summary.final_convergence_mean,
            "final_convergence_std": summary.final_convergence_std,
            "number_of_compressed_tokens": summary.number_of_compressed_tokens,
            "number_of_compressed_tokens_std": summary.number_of_compressed_tokens_std,
            "max_sequence_length": summary.max_sequence_length,
            "trajectory_length_mean": getattr(summary, "trajectory_length_mean", None),
            "trajectory_length_std": getattr(summary, "trajectory_length_std", None),
            "pca_99_mean": getattr(summary, "pca_99_mean", None),
            "pca_99_std": getattr(summary, "pca_99_std", None),
        }

    def summary_from_cache(data: dict) -> SimpleNamespace:
        return SimpleNamespace(**data)

    def load_cache(run_dir: str, ds_path: str) -> SimpleNamespace | None:
        cache_path = os.path.join(run_dir, cache_filename)
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
        except (json.JSONDecodeError, OSError):
            return None
        if payload.get("version") != cache_version:
            return None
        if payload.get("dataset_mtime") != os.path.getmtime(ds_path):
            return None
        summary_data = payload.get("summary")
        if not isinstance(summary_data, dict):
            return None
        return summary_from_cache(summary_data)

    def save_cache(run_dir: str, ds_path: str, summary) -> None:
        cache_path = os.path.join(run_dir, cache_filename)
        payload = {
            "version": cache_version,
            "dataset_mtime": os.path.getmtime(ds_path),
            "summary": summary_to_cache(summary),
        }
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file)

    def flatten_embedding(embedding) -> np.ndarray:
        emb = np.asarray(embedding, dtype=np.float32)
        return emb.reshape(-1)

    def compute_trajectory_length(embeddings: list[np.ndarray]) -> float:
        if len(embeddings) < 2:
            return 0.0
        trajectory_length = 0.0
        for i in range(len(embeddings) - 1):
            dist = np.linalg.norm(embeddings[i + 1] - embeddings[i])
            trajectory_length += dist
        return float(trajectory_length)

    def compute_num_pca_explained_99_var(embeddings: list[np.ndarray]) -> float:
        if len(embeddings) < 2:
            return float("nan")
        X = np.stack(embeddings, axis=0)
        if X.shape[0] < 2:
            return float("nan")
        n_samples, n_features = X.shape
        max_pca_components = min(512, n_samples - 1, n_features)
        if max_pca_components < 1:
            return float("nan")
        pca = PCA(n_components=max_pca_components, random_state=42)
        pca.fit(X)
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)
        num_pca_for99_var = (cumulative_var < 0.99).sum()
        if num_pca_for99_var == max_pca_components:
            num_pca_for99_var = -1
        return float(num_pca_for99_var)

    def summarize_values(values: list[float]) -> dict | None:
        if len(values) == 0:
            return None
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    def compute_progressive_trajectory_metrics(ds_path: str) -> dict | None:
        try:
            ds = load_from_disk(ds_path)
        except Exception:
            return None
        if "embedding" not in ds.column_names:
            return None

        rows = [dict(r) for r in ds]
        by_sample: dict[int, list[dict]] = {}
        for row in rows:
            sample_id = row.get("sample_id")
            if sample_id is None:
                continue
            by_sample.setdefault(int(sample_id), []).append(row)

        trajectory_lengths = []
        pca_99_values = []

        for sample_rows in by_sample.values():
            sample_rows.sort(key=lambda x: int(x.get("stage_index", 0)))
            sample_embeddings = []
            has_embeddings = True
            for row in sample_rows:
                embedding = row.get("embedding")
                if embedding is None:
                    has_embeddings = False
                    break
                sample_embeddings.append(flatten_embedding(embedding))
            if not has_embeddings or len(sample_embeddings) == 0:
                continue
            trajectory_lengths.append(compute_trajectory_length(sample_embeddings))
            pca_99 = compute_num_pca_explained_99_var(sample_embeddings)
            if not np.isnan(pca_99):
                pca_99_values.append(pca_99)

        if len(trajectory_lengths) == 0 and len(pca_99_values) == 0:
            return None

        return {
            "trajectory_length": summarize_values(trajectory_lengths),
            "pca_99_var": summarize_values(pca_99_values),
        }

    def format_metric_cell(mean_val: float | None, std_val: float | None, precision: int) -> str:
        if mean_val is None or std_val is None:
            return "nan"
        return to_mean_std_cell(
            mean_val,
            std_val,
            use_latex=(args.tablefmt == "latex"),
            float_precision=precision,
        )

    ordered_summaries = []
    for experiment in tqdm(experiments_list, desc="Processing Runs"):
        rows = None
        summary = None
        if experiment["train"] == "full":
            full_exp_name = glob.glob(f"artifacts/experiments/*{experiment['id']}/")
            assert len(full_exp_name) == 1, f"experiments hashes must be unique: {full_exp_name}"
            run_dir = full_exp_name[0]
            full_exp_name = os.path.join(run_dir, "compressed_prefixes")
            if os.path.isdir(full_exp_name):
                summary = load_cache(run_dir, full_exp_name)
                if summary is None:
                    rows = load_dataset_rows(full_exp_name)
                    summary = aggregate_non_progressive(full_exp_name, rows)
                    if summary is not None:
                        save_cache(run_dir, full_exp_name, summary)
        elif experiment["train"] == "progr":
            run_dir = f"artifacts/experiments_progressive/{experiment['id']}"
            full_ds_path = os.path.join(run_dir, "progressive_prefixes")
            if os.path.isdir(full_ds_path):
                summary = load_cache(run_dir, full_ds_path)
                if summary is None:
                    rows = load_dataset_rows(full_ds_path)
                    summary = aggregate_progressive(full_ds_path, rows)
        else:
            raise ValueError(f"Unknown train type: {experiment['train']}")

        if summary is None:
            print("Failed to load:", experiment)
            continue

        if summary.dataset_type == "progressive_prefixes":
            run_dir_for_metrics = getattr(summary, "run_dir", run_dir)
            full_ds_path = os.path.join(run_dir_for_metrics, "progressive_prefixes")
            has_metrics = (
                getattr(summary, "trajectory_length_mean", None) is not None
                and getattr(summary, "trajectory_length_std", None) is not None
                and getattr(summary, "pca_99_mean", None) is not None
                and getattr(summary, "pca_99_std", None) is not None
            )
            if not has_metrics:
                metrics = compute_progressive_trajectory_metrics(full_ds_path)
                if metrics is not None:
                    traj_stats = metrics.get("trajectory_length")
                    pca_stats = metrics.get("pca_99_var")
                    summary.trajectory_length_mean = None if not traj_stats else traj_stats.get("mean")
                    summary.trajectory_length_std = None if not traj_stats else traj_stats.get("std")
                    summary.pca_99_mean = None if not pca_stats else pca_stats.get("mean")
                    summary.pca_99_std = None if not pca_stats else pca_stats.get("std")
                else:
                    summary.trajectory_length_mean = None
                    summary.trajectory_length_std = None
                    summary.pca_99_mean = None
                    summary.pca_99_std = None
                save_cache(run_dir_for_metrics, full_ds_path, summary)
        else:
            summary.trajectory_length_mean = None
            summary.trajectory_length_std = None
            summary.pca_99_mean = None
            summary.pca_99_std = None

        ordered_summaries.append(summary)

    result_table_rows = []
    for i, summary in enumerate(ordered_summaries):

        exp_data = experiments_list[i]
        experiment = format_experiment_label(summary, fallback_label=str(summary.run_hash or ""))
        info_gain = to_mean_std_cell(
            summary.information_gain_bits_mean,
            summary.information_gain_bits_std,
            use_latex=(args.tablefmt == "latex"),
            float_precision=0,
        )
        is_progressive = summary.dataset_type == "progressive_prefixes"
        if not is_progressive:
            max_tokens = summary.max_sequence_length
        else:
            # max_tokens = summary.number_of_compressed_tokens
            max_tokens = to_mean_std_cell(
                summary.number_of_compressed_tokens,
                summary.number_of_compressed_tokens_std,
                use_latex=(args.tablefmt == "latex"),
                float_precision=0,
            )

        # exp_type = "Progr." if is_progressive else "Full"
        # traj_length = format_metric_cell(summary.trajectory_length_mean, summary.trajectory_length_std, precision=0)
        pca_99 = format_metric_cell(summary.pca_99_mean, summary.pca_99_std, precision=0)
        result_table_rows.append([exp_data["type"], max_tokens, info_gain, pca_99])
        # result_table_rows.append([exp_data["type"], max_tokens, traj_length, pca_99, info_gain])

    result = tabulate(result_table_rows, headers=columns, tablefmt=args.tablefmt)
    result = result.replace("\\textbackslash{}", "\\")
    result = result.replace("\\$", "$")
    result = result.replace("\\{", "{")
    result = result.replace("\\}", "}")
    result = result.replace("P-", "Pythia")
    result = result.replace("L3.2-", "Llama-3.2-")
    result = result.replace("L3.1-", "Llama-3.1-")

    import re

    result = re.sub(r"REMOVE.+", "", result)

    if args.tablefmt == "latex":
        result = hlines_to_booktabs(result)

    print(result)


if __name__ == "__main__":
    main()
