"""Rephrase the ARC benchmark (Easy or Challenge) via an OpenAI-compatible LLM API.

For each item, paraphrases `question` and every entry of `choices.text` independently.
`choices.label` and `answerKey` are preserved byte-identical so the rephrased dataset
is a drop-in replacement for the existing eval script (`scripts/arc_compress_evaluate.py`)
when invoked with `--dataset_path`.

Two extra columns are added for traceability:
- `question_paraphrase_failed: bool` — True if the question paraphrase exhausted retries.
- `choices_paraphrase_failed: list[bool]` — analogous per choice (variable length).
On failure the corresponding text field is replaced with
`PARAPHRASE_FAILED_SENTINEL` ("<paraphrase_failed>") rather than left as the
upstream original, to prevent answer-leakage in mixed-paraphrase rows.

ARC items have variable choice counts (typically 4 or 5, occasionally 3).

Example:
    python scripts/data/generate_arc_paraphrases.py \
        --arc_subset ARC-Challenge \
        --split validation \
        --limit 5 \
        --concurrency 8 \
        --output_dir artifacts/arc_paraphrases
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import replace

from datasets import Dataset, load_dataset
from scripts.data.paraphrase_client import (
    AsyncParaphraseClient,
    add_client_args,
    client_config_from_args,
    paraphrase_grouped,
)

logger = logging.getLogger(__name__)

ARC_SUBSETS = ("ARC-Challenge", "ARC-Easy")
EXTRA_COLUMNS = ("question_paraphrase_failed", "choices_paraphrase_failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rephrase ARC via OpenAI-compatible LLM API.")
    add_client_args(parser)
    parser.add_argument("--arc_subset", type=str, default="ARC-Challenge", choices=list(ARC_SUBSETS))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="artifacts/arc_paraphrases")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument(
        "--hub_dataset_id",
        type=str,
        default=None,
        help=(
            "HF Hub dataset id (e.g. 'username/arc-challenge-paraphrased'). Required if --push_to_hub. "
            "When using --temperatures with more than one value, this string must contain a "
            "'{temperature}' placeholder, e.g. 'username/arc-challenge-paraphrased-t{temperature}'."
        ),
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default=None,
        help=(
            "Comma-separated list of temperatures to run sequentially (e.g. '0.0,0.3,0.7'). "
            "Each temperature writes a subdirectory of --output_dir named 'temp_{T:.2f}'. "
            "Overrides --temperature when set. Other --top_p / --max_tokens / --seed flags apply to every pass."
        ),
    )
    return parser.parse_args()


async def rephrase_dataset(client: AsyncParaphraseClient, dataset: Dataset) -> Dataset:
    for required in ("question", "choices", "answerKey"):
        if required not in dataset.column_names:
            raise ValueError(f"ARC dataset missing required column: {required!r}")
    for col in EXTRA_COLUMNS:
        if col in dataset.column_names:
            raise ValueError(f"ARC dataset already has reserved output column: {col!r}")

    grouped_texts: list[list[str]] = []
    for item in dataset:
        choices = item["choices"]
        if not isinstance(choices, dict) or "text" not in choices or "label" not in choices:
            raise ValueError(f"unexpected ARC choices structure for item {item.get('id')}: {choices!r}")
        grouped_texts.append([item["question"], *choices["text"]])

    grouped_results = await paraphrase_grouped(client, grouped_texts, desc="arc")

    new_questions: list[str] = []
    new_choices: list[dict] = []
    question_failed: list[bool] = []
    choices_failed: list[list[bool]] = []
    for item, group in zip(dataset, grouped_results):
        q_text, q_ok = group[0]
        choice_results = group[1:]
        new_questions.append(q_text)
        new_choices.append({"text": [t for t, _ in choice_results], "label": list(item["choices"]["label"])})
        question_failed.append(not q_ok)
        choices_failed.append([not ok for _, ok in choice_results])

    new_data = {col: list(dataset[col]) for col in dataset.column_names}
    new_data["question"] = new_questions
    new_data["choices"] = new_choices
    new_data["question_paraphrase_failed"] = question_failed
    new_data["choices_paraphrase_failed"] = choices_failed

    total_failures = sum(question_failed) + sum(sum(row) for row in choices_failed)
    total_fields = sum(1 + len(row) for row in choices_failed)
    print(f"Done. Per-field paraphrase failures: {total_failures} / {total_fields}")

    return Dataset.from_dict(new_data)


def _resolve_temperatures(args: argparse.Namespace) -> tuple[list[float], bool]:
    """Return (temperatures, per_temp_output). per_temp_output=True means each
    temperature writes to a subdirectory of args.output_dir."""
    if args.temperatures is None:
        return [args.temperature], False
    temps = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]
    if not temps:
        raise ValueError("--temperatures was given but parsed to an empty list")
    return temps, len(temps) > 1


async def main_async(args: argparse.Namespace) -> None:
    temperatures, per_temp_output = _resolve_temperatures(args)

    if args.push_to_hub:
        if not args.hub_dataset_id:
            raise ValueError("--push_to_hub requires --hub_dataset_id")
        if per_temp_output and "{temperature}" not in args.hub_dataset_id:
            raise ValueError(
                "--push_to_hub with multiple --temperatures requires '{temperature}' "
                "placeholder in --hub_dataset_id (e.g. 'user/arc-challenge-paraphrased-t{temperature}')"
            )

    print(f"Loading allenai/ai2_arc subset={args.arc_subset} split={args.split}...")
    dataset = load_dataset("allenai/ai2_arc", args.arc_subset, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Loaded {len(dataset)} items.")

    base_config = client_config_from_args(args)
    for temp in temperatures:
        print(f"\n=== temperature={temp} ===")
        config = replace(base_config, temperature=temp)
        client = AsyncParaphraseClient(config)
        rephrased = await rephrase_dataset(client, dataset)

        out_dir = args.output_dir if not per_temp_output else os.path.join(args.output_dir, f"temp_{temp:.2f}")
        print(f"Saving rephrased dataset to {out_dir}...")
        rephrased.save_to_disk(out_dir)

        if args.push_to_hub:
            hub_id = args.hub_dataset_id.format(temperature=f"{temp:.2f}") if per_temp_output else args.hub_dataset_id
            print(f"Pushing to hub: {hub_id}")
            rephrased.push_to_hub(hub_id)

        print(f"Done. {len(rephrased)} items written for temperature={temp}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
