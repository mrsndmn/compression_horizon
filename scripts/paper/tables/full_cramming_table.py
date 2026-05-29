import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Load scripts/results/results.py directly: the sibling scripts/results.py
# file (unrelated, broken) shadows the package on `import scripts.results...`.
_RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "results.py"
_spec = importlib.util.spec_from_file_location("_paper_results_helpers", _RESULTS_PATH)
assert _spec is not None and _spec.loader is not None
_results = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_paper_results_helpers", _results)
_spec.loader.exec_module(_results)
aggregate_non_progressive = _results.aggregate_non_progressive
aggregate_prefix_tuning = _results.aggregate_prefix_tuning
aggregate_progressive = _results.aggregate_progressive
load_dataset_rows = _results.load_dataset_rows

from tabulate import tabulate  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from compression_horizon.utils import hlines_to_booktabs, to_mean_std_cell  # noqa: E402

TYPE_TO_SLUG = {
    "full_cramming": "full_vs_progressive",
    "full_cramming_apendix": "full_vs_progressive_appendix",
    "prefix_tuning": "prefix_tuning_accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full cramming results table.")
    parser.add_argument(
        "--tablefmt",
        default="latex",
        help="Tabulate table format (e.g., plain, github, latex, grid).",
    )
    parser.add_argument(
        "--type",
        choices=["full_cramming", "prefix_tuning", "full_cramming_apendix"],
        default="full_cramming",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set, write the rendered table to <save-dir>/<slug>.tex (slug derived from --type).",
    )
    parser.add_argument(
        "--greedy-precision",
        type=int,
        default=2,
        help="Decimal places for the greedy-accuracy percent column (default 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.type == "full_cramming":
        cache_filename = "full_cramming_table_cache.json"
        cache_version = 1

        experiments_list = [
            # # Llama-3.2-1B
            # # {"train": "full", "id": "4e378cf3"},  # 256
            # {"train": "full", "id": "af92266e"},  # 512
            # {"train": "progr", "id": "sl_4096_Llama-3.2-1B_lr_0.1"},
            # # Llama-3.2-3B
            # # {"train": "full", "id": "7359e14b"},  # 512
            # {"train": "full", "id": "ef2ea924"},  # 1024
            # {"train": "progr", "id": "sl_4096_Llama-3.2-3B_lr_0.1"},
            # Llama-3.1-8B (50-sample re-run, lr matches progressive)
            {"train": "full", "id": "eae01b69"},  # 1568, lr 0.1
            {"train": "progr", "id": "sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1"},
            # Pythia 160M
            # {"train": "full", "id": "dbced9cc"},  # 32
            # # {"train": "full", "id": "6a93af63"},  # 64
            # {"train": "progr", "id": "sl_4096_pythia-160m_lr_0.5"},
            # # Pythia 410M
            # # {"train": "full", "id": "328bdbfb"},  # 96
            # {"train": "full", "id": "22d7b7db"},  # 128
            # {"train": "progr", "id": "sl_4096_pythia-410m_lr_0.5"},
            # Pythia 1.4B (50-sample re-run, lr matches progressive)
            # 256-token Full row intentionally omitted from the main body (kept in the appendix).
            {"train": "full", "id": "eabb5144"},  # 512, lr 0.5
            {"train": "progr", "id": "sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5"},
        ]
    elif args.type == "full_cramming_apendix":
        cache_filename = "full_cramming_table_cache.json"
        cache_version = 1

        experiments_list = [
            # Llama-3.2-1B
            # {"train": "full", "id": "4e378cf3"},  # 256
            {"train": "full", "id": "af92266e"},  # 512
            {"train": "progr", "id": "sl_4096_Llama-3.2-1B_lr_0.1"},
            # Llama-3.2-3B
            # {"train": "full", "id": "7359e14b"},  # 512
            {"train": "full", "id": "ef2ea924"},  # 1024
            {"train": "progr", "id": "sl_4096_Llama-3.2-3B_lr_0.1"},
            # Llama-3.1-8B
            # {"train": "full", "id": "dfbe32b8"},  # 1024
            {"train": "full", "id": "b5aef07e"},  # 1568
            {"train": "progr", "id": "sl_4096_Meta-Llama-3.1-8B_lr_0.1"},
            # Pythia 160M
            {"train": "full", "id": "dbced9cc"},  # 32
            # {"train": "full", "id": "6a93af63"},  # 64
            {"train": "progr", "id": "sl_4096_pythia-160m_lr_0.5"},
            # Pythia 410M
            # {"train": "full", "id": "328bdbfb"},  # 96
            {"train": "full", "id": "22d7b7db"},  # 128
            {"train": "progr", "id": "sl_4096_pythia-410m_lr_0.5"},
            # Pythia 1.4B
            # {"train": "full", "id": "f3296f56"},  # 160
            {"train": "full", "id": "a1e58eb5"},  # 256
            {"train": "progr", "id": "sl_4096_pythia-1.4b_lr_0.5"},
        ]
    elif args.type == "prefix_tuning":
        cache_filename = "prefix_tuning_table_cache.json"
        cache_version = 1

        # TODO pythia

        experiments_list = [
            # Llama-3.2-1B
            # TODO
            {"train": "progr", "id": "sl_4096_Llama-3.2-1B_lr_0.1"},
            # {"train": "prefix", "id": "pt_sl_1024_Llama-3.2-1B"},
            # {"train": "prefix", "id": "pt_sl_2048_Llama-3.2-1B"},
            {"train": "prefix", "id": "pt_sl_4096_Llama-3.2-1B"},
            {"train": "prefix", "id": "pt_sl_8192_Llama-3.2-1B"},
            {"train": "prefix", "id": "pt_sl_16384_Llama-3.2-1B"},
            # Llama-3.2-3B
            {"train": "progr", "id": "sl_4096_Llama-3.2-3B_lr_0.1"},
            # {"train": "prefix", "id": "pt_sl_1024_Llama-3.2-3B"},
            # {"train": "prefix", "id": "pt_sl_2048_Llama-3.2-3B"},
            # {"train": "prefix", "id": "pt_sl_4096_Llama-3.2-3B"},
            {"train": "prefix", "id": "pt_sl_8192_Llama-3.2-3B"},
            # {"train": "prefix", "id": "pt_sl_16384_Llama-3.2-3B"}, # OOM
            #
            {"train": "progr", "id": "sl_4096_Meta-Llama-3.1-8B_lr_0.1"},
            # {"train": "prefix", "id": "pt_sl_1024_Meta-Llama-3.1-8B"},
            # {"train": "prefix", "id": "pt_sl_2048_Meta-Llama-3.1-8B"},
            # {"train": "prefix", "id": "pt_sl_4096_Meta-Llama-3.1-8B"},
            {"train": "prefix", "id": "pt_sl_8192_Meta-Llama-3.1-8B"},
            # {"train": "prefix", "id": "pt_sl_16384_Meta-Llama-3.1-8B"}, # OOM
            # Llama-3.1-8B TODO
            # Pythia TODO
            {"train": "progr", "id": "sl_4096_pythia-160m_lr_0.5"},
            {"train": "prefix", "id": "pt_sl_1024_pythia-160m"},
            {"train": "prefix", "id": "pt_sl_2048_pythia-160m"},
            {"train": "prefix", "id": "pt_sl_4096_pythia-160m"},
            {"train": "prefix", "id": "pt_sl_8192_pythia-160m"},
            {"train": "prefix", "id": "pt_sl_16384_pythia-160m"},
            {"train": "progr", "id": "sl_4096_pythia-410m_lr_0.5"},
            # {"train": "prefix", "id": "pt_sl_1024_pythia-410m"},
            {"train": "prefix", "id": "pt_sl_2048_pythia-410m"},
            {"train": "prefix", "id": "pt_sl_4096_pythia-410m"},
            {"train": "prefix", "id": "pt_sl_8192_pythia-410m"},
            {"train": "prefix", "id": "pt_sl_16384_pythia-410m"},
            {"train": "progr", "id": "sl_4096_pythia-1.4b_lr_0.5"},
            # {"train": "prefix", "id": "pt_sl_1024_pythia-1.4b"},
            # {"train": "prefix", "id": "pt_sl_2048_pythia-1.4b"},
            {"train": "prefix", "id": "pt_sl_4096_pythia-1.4b"},
            {"train": "prefix", "id": "pt_sl_8192_pythia-1.4b"},
            {"train": "prefix", "id": "pt_sl_16384_pythia-1.4b"},
        ]

    # The full-cramming tables (main + appendix) report teacher-forcing accuracy
    # in percent; the prefix-tuning variant keeps the [0, 1] portion.
    accuracy_in_percent = args.type in ("full_cramming", "full_cramming_apendix")

    def format_pct_cell(mean_frac, std_frac, precision: int = 2) -> str:
        """Fixed-decimal percent cell (no trailing-zero stripping), from a [0, 1] fraction."""
        if mean_frac is None:
            return ""
        mean_str = f"{mean_frac * 100:.{precision}f}"
        if std_frac is None:
            return mean_str
        std_str = f"{std_frac * 100:.{precision}f}"
        if args.tablefmt == "latex":
            return f"{mean_str} {{\\small $\\pm$ {std_str}}}"
        return f"{mean_str} ± {std_str}"

    if args.type == "prefix_tuning":
        columns = ["Experiment", "Type", "Tokens", "Accuracy"]
    else:
        # TF / Greedy are grouped under a single "Accuracy (\%)" multicolumn header
        # (injected below) to keep the single-column main-body table within width.
        tf_header = "TF" if accuracy_in_percent else "Accuracy"
        columns = ["Type", "Tokens", tf_header, "Greedy"]

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

    def load_greedy_cache(run_dir: str) -> tuple[float | None, float | None] | None:
        """Read greedy_accuracy_cache.json (written by scripts/greedy_reconstruction_eval.py)."""
        cache_path = os.path.join(run_dir, "greedy_accuracy_cache.json")
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        return payload.get("greedy_match_mean"), payload.get("greedy_match_std")

    ordered_summaries = []
    ordered_dirs = []
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
                    if summary is not None:
                        save_cache(run_dir, full_ds_path, summary)
        elif experiment["train"] == "prefix":
            run_dir = f"artifacts/experiments_prefix_tuning/{experiment['id']}"
            full_ds_path = os.path.join(run_dir, "prefix_tuning_prefixes")
            if os.path.isdir(full_ds_path):
                summary = load_cache(run_dir, full_ds_path)
                if summary is None:
                    rows = load_dataset_rows(full_ds_path)
                    summary = aggregate_prefix_tuning(full_ds_path, rows)
                    if summary is not None:
                        save_cache(run_dir, full_ds_path, summary)
        else:
            raise ValueError(f"Unknown train type: {experiment['train']}")

        if summary is None:
            print("Failed to load:", experiment)
            continue

        ordered_summaries.append(summary)
        ordered_dirs.append(run_dir)

    result_table_rows = []
    prev_exp = None
    for i, summary in enumerate(ordered_summaries):
        experiment = format_experiment_label(summary, fallback_label=str(summary.run_hash or ""))
        # info_gain = to_mean_std_cell(
        #     summary.information_gain_bits_mean,
        #     summary.information_gain_bits_std,
        #     use_latex=(args.tablefmt == "latex"),
        #     float_precision=0,
        # )
        is_progressive = summary.dataset_type == "progressive_prefixes"
        is_prefix_tuning = summary.dataset_type == "prefix_tuning_prefixes"
        if not is_progressive:
            if accuracy_in_percent:
                accuracy = format_pct_cell(summary.final_convergence_mean, summary.final_convergence_std)
            else:
                accuracy = to_mean_std_cell(
                    summary.final_convergence_mean,
                    summary.final_convergence_std,
                    use_latex=(args.tablefmt == "latex"),
                    float_precision=3,
                )
            max_tokens = summary.max_sequence_length
        else:
            accuracy = "100.00" if accuracy_in_percent else "1.0"
            # max_tokens = summary.number_of_compressed_tokens
            max_tokens = to_mean_std_cell(
                summary.number_of_compressed_tokens,
                summary.number_of_compressed_tokens_std,
                use_latex=(args.tablefmt == "latex"),
                float_precision=0,
            )

        if args.type != "prefix_tuning":
            if prev_exp is None or prev_exp != experiment:
                num_cols = len(columns)
                result_table_rows.append([f"\\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{{experiment}}}}} \\\\ REMOVE"])
        else:
            if prev_exp is None or prev_exp != experiment:
                result_table_rows.append(["\\midrule REMOVE "])

        if is_progressive:
            exp_type = "Prog." if args.type != "prefix_tuning" else "Progr."
        elif is_prefix_tuning:
            exp_type = "Full PrefixT."
        else:
            exp_type = "Full"
        if args.type == "prefix_tuning":
            result_table_rows.append([experiment, exp_type, max_tokens, accuracy])
        else:
            if is_progressive and accuracy_in_percent:
                greedy_cell = "100.00"
            else:
                greedy = load_greedy_cache(ordered_dirs[i])
                if greedy is None or greedy[0] is None:
                    greedy_cell = "--"
                else:
                    greedy_cell = format_pct_cell(greedy[0], greedy[1], precision=args.greedy_precision)
            result_table_rows.append([exp_type, max_tokens, accuracy, greedy_cell])
        if args.type != "prefix_tuning":
            if is_progressive and i != len(ordered_summaries) - 1:
                if "L3.1" in experiment:
                    result_table_rows.append(["\\midrule \\midrule REMOVE "])
                else:
                    result_table_rows.append(["\\midrule REMOVE "])
        # else:
        #     if i > 0 and i % 4 == 0 and i != len(ordered_summaries) - 1:
        #         result_table_rows.append(["\midrule REMOVE "])

        prev_exp = experiment

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
        if args.type in ("full_cramming", "full_cramming_apendix"):
            # Right-align the numeric columns, tighten inter-column spacing, and
            # group TF/Greedy under a single "Accuracy (\%)" header so the table
            # fits the single-column width.
            result = result.replace(
                "\\begin{tabular}{llll}",
                "\\begin{tabular}{lrrr}",
                1,
            )
            group_header = " & & \\multicolumn{2}{c}{Accuracy (\\%)} \\\\\n" "\\cmidrule(lr){3-4}\n"
            result = result.replace("\\toprule\n", "\\toprule\n" + group_header, 1)

    print(result)

    # Provenance stamp consumed by paper/lint_paper.py: the PG19 sample count is
    # parsed from each source dir name (limit_<N>); main-body tables must be 50.
    sample_counts = sorted({m.group(1) for d in ordered_dirs if (m := re.search(r"limit_(\d+)", d))})
    if len(sample_counts) == 1:
        n_samples_stamp = sample_counts[0]
    elif sample_counts:
        n_samples_stamp = ",".join(sample_counts)
    else:
        n_samples_stamp = "unknown"

    if args.save_dir is not None:
        slug = TYPE_TO_SLUG[args.type]
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}.tex"
        stamp = f"% paper-lint: n_samples={n_samples_stamp}\n"
        out_path.write_text(stamp + result + "\n", encoding="utf-8")
        print(f"\nSaved 'tab:{slug}' to {out_path}")


if __name__ == "__main__":
    main()
