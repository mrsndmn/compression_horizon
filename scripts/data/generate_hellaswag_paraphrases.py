"""Rephrase the HellaSwag benchmark via an OpenAI-compatible LLM API (default: local vLLM).

For each item, paraphrases `ctx` and every entry of `endings` independently.
`label` (the index of the correct ending) is preserved byte-identical so the
rephrased dataset is a drop-in replacement for the existing eval script
(`scripts/hellaswag_compress_evaluate.py`) when invoked with `--dataset_path`.

Two extra columns are added for traceability:
- `ctx_paraphrase_failed: bool` — True if the context paraphrase exhausted retries.
- `endings_paraphrase_failed: list[bool]` (length 4) — analogous per ending.
On failure the corresponding text field is replaced with
`PARAPHRASE_FAILED_SENTINEL` ("<paraphrase_failed>") rather than left as the
upstream original, to prevent answer-leakage in mixed-paraphrase rows.

Example:
    python scripts/data/generate_hellaswag_paraphrases.py \
        --split validation \
        --limit 5 \
        --concurrency 8 \
        --output_dir artifacts/hellaswag_paraphrases
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

NUM_ENDINGS = 4
EXTRA_COLUMNS = ("ctx_paraphrase_failed", "endings_paraphrase_failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rephrase HellaSwag via OpenAI-compatible LLM API.")
    add_client_args(parser)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="artifacts/hellaswag_paraphrases")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument(
        "--hub_dataset_id",
        type=str,
        default=None,
        help=(
            "HF Hub dataset id (e.g. 'username/hellaswag-paraphrased'). Required if --push_to_hub. "
            "When using --temperatures with more than one value, this string must contain a "
            "'{temperature}' placeholder, e.g. 'username/hellaswag-paraphrased-t{temperature}'."
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
    for required in ("ctx", "endings", "label"):
        if required not in dataset.column_names:
            raise ValueError(f"HellaSwag dataset missing required column: {required!r}")
    for col in EXTRA_COLUMNS:
        if col in dataset.column_names:
            raise ValueError(f"HellaSwag dataset already has reserved output column: {col!r}")

    grouped_texts: list[list[str]] = []
    for item in dataset:
        endings = item["endings"]
        if len(endings) != NUM_ENDINGS:
            raise ValueError(f"expected {NUM_ENDINGS} endings, got {len(endings)} for item {item.get('ind')}")
        grouped_texts.append([item["ctx"], *endings])

    grouped_results = await paraphrase_grouped(client, grouped_texts, desc="hellaswag")

    new_ctx: list[str] = []
    new_endings: list[list[str]] = []
    ctx_failed: list[bool] = []
    endings_failed: list[list[bool]] = []
    for group in grouped_results:
        ctx_text, ctx_ok = group[0]
        ending_results = group[1:]
        new_ctx.append(ctx_text)
        new_endings.append([t for t, _ in ending_results])
        ctx_failed.append(not ctx_ok)
        endings_failed.append([not ok for _, ok in ending_results])

    new_data = {col: list(dataset[col]) for col in dataset.column_names}
    new_data["ctx"] = new_ctx
    new_data["endings"] = new_endings
    new_data["ctx_paraphrase_failed"] = ctx_failed
    new_data["endings_paraphrase_failed"] = endings_failed

    total_failures = sum(ctx_failed) + sum(sum(row) for row in endings_failed)
    total_fields = len(dataset) * (1 + NUM_ENDINGS)
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
                "placeholder in --hub_dataset_id (e.g. 'user/hellaswag-paraphrased-t{temperature}')"
            )

    print(f"Loading Rowan/hellaswag split={args.split}...")
    dataset = load_dataset("Rowan/hellaswag", split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Loaded {len(dataset)} items.")

    if args.split == "test":
        # The Rowan/hellaswag test split has empty labels (private leaderboard).
        # Documented in the spec; warn so users aren't surprised downstream.
        logger.warning(
            "split='test' on Rowan/hellaswag has empty labels; rephrased data will not be "
            "usable for accuracy evaluation. Pass --split validation if you need labels."
        )

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
