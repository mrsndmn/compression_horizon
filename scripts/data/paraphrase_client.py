"""Shared async paraphrase client for benchmark rephrasing scripts.

Used by `generate_hellaswag_paraphrases.py` and `generate_arc_paraphrases.py`
to call an OpenAI-compatible API (default: local vLLM at http://localhost:8000/v1)
with bounded concurrency, deterministic generation, and retry-with-backoff.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from openai import APIError, AsyncOpenAI
from tqdm import tqdm

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

logger = logging.getLogger(__name__)


# Few-shot base-model completion prompt. The default vLLM target is the base
# model `unsloth/Meta-Llama-3.1-8B`, which has no chat template and follows
# zero-shot instructions poorly (it tends to either copy short text verbatim
# or continue fragments instead of paraphrasing). Few-shot examples are a
# much stronger steer for base models. The three exemplars below cover the
# three input shapes the benchmarks actually contain:
#   1. An atomic short answer (matches ARC `choices.text`).
#   2. A complete sentence-level question (matches ARC `question`).
#   3. A deliberately truncated fragment (matches HellaSwag `ctx`, whose
#      endings are designed to complete the fragment — we preserve the
#      truncation rather than completing it).
PARAPHRASE_PROMPT_TEMPLATE = (
    'Rewrite the text after "Text:" in different words while preserving its '
    "meaning, register, and any truncation. Output only the rewritten text on "
    'the line after "Rewritten:". Do not add explanations or extra examples.\n'
    "\n"
    "Text: Put the objects in groups.\n"
    "Rewritten: Sort the objects into separate groups.\n"
    "\n"
    "Text: What is the primary source of energy for Earth's weather?\n"
    "Rewritten: Which source mainly powers the weather on Earth?\n"
    "\n"
    "Text: A man is sitting on a roof. he\n"
    "Rewritten: A guy is perched on top of a roof. he\n"
    "\n"
    "Text: {text}\n"
    "Rewritten:"
)

# Stop sequences that bound the paraphrase output for the completions endpoint.
# The few-shot template uses single-newline separators between Text/Rewritten
# blocks, so "\nText:" is the primary stop. Belt-and-suspenders sequences
# guard against the model regenerating its own example headers.
PARAPHRASE_STOP_SEQUENCES = ["\nText:", "\n\nText:", "\nRewritten:"]

# Sentinel returned (with ok=False) when a per-field paraphrase fails after all
# retries. Returning a sentinel instead of the upstream original prevents an
# evaluation-validity hazard: in a multiple-choice item where 1 option fails to
# paraphrase, leaving it verbatim upstream while siblings are rephrased would
# create a stylistic tell that leaks the correct option.
PARAPHRASE_FAILED_SENTINEL = "<paraphrase_failed>"


@dataclass
class ParaphraseClientConfig:
    base_url: str = "http://localhost:9999/v1"
    api_key: str = "EMPTY"
    model: str = "unsloth/Meta-Llama-3.1-8B"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    concurrency: int = 32
    retries: int = 3
    backoff_base: float = 1.0
    backoff_cap: float = 30.0
    seed: int = 0


class AsyncParaphraseClient:
    """Wraps AsyncOpenAI with a global semaphore + retry policy."""

    def __init__(self, config: ParaphraseClientConfig):
        self.config = config
        # If base_url points at localhost, bypass any inherited HTTP(S)_PROXY env
        # vars. The user's environment sets HTTP_PROXY for outbound access, but
        # routing local vLLM through that proxy black-holes every request.
        host = (urlparse(config.base_url).hostname or "").lower()
        trust_env = host not in _LOCALHOST_HOSTS
        http_client = httpx.AsyncClient(trust_env=trust_env, timeout=httpx.Timeout(120.0))
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            http_client=http_client,
        )
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def paraphrase(self, text: str) -> tuple[str, bool]:
        """Paraphrase a single text.

        Returns:
            (rephrased_text, ok). On retry exhaustion returns
            (PARAPHRASE_FAILED_SENTINEL, False); caller can record
            paraphrase_failed=True. The sentinel (rather than the upstream
            original) prevents answer-leakage in multiple-choice rows.
        """
        # Empty / whitespace-only inputs cannot be meaningfully paraphrased.
        if text is None or not text.strip():
            return (text or "", True)

        prompt = PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
        async with self._semaphore:
            last_err: Exception | None = None
            for attempt in range(self.config.retries):
                try:
                    response = await self._client.completions.create(
                        model=self.config.model,
                        prompt=prompt,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_tokens=self.config.max_tokens,
                        stop=PARAPHRASE_STOP_SEQUENCES,
                        # vLLM honors `seed` via the OpenAI extra-body channel.
                        extra_body={"seed": self.config.seed},
                    )
                    content = response.choices[0].text
                    if content is None or not content.strip():
                        last_err = ValueError("empty response content")
                    else:
                        return (content.strip(), True)
                except (APIError, asyncio.TimeoutError) as exc:
                    last_err = exc
                except Exception as exc:  # noqa: BLE001 — log+retry on any client error
                    last_err = exc

                if attempt < self.config.retries - 1:
                    backoff = min(self.config.backoff_cap, self.config.backoff_base * (2**attempt))
                    sleep_for = backoff + random.uniform(0, self.config.backoff_base)
                    logger.warning(
                        "paraphrase attempt %d/%d failed (%s); sleeping %.1fs",
                        attempt + 1,
                        self.config.retries,
                        last_err,
                        sleep_for,
                    )
                    await asyncio.sleep(sleep_for)

            logger.error("paraphrase failed after %d attempts: %s", self.config.retries, last_err)
            return (PARAPHRASE_FAILED_SENTINEL, False)


async def paraphrase_grouped(
    client: AsyncParaphraseClient,
    grouped_texts: list[list[str]],
    desc: str = "paraphrase",
) -> list[list[tuple[str, bool]]]:
    """Paraphrase a list of text groups, preserving group structure.

    All texts are submitted in a single ``asyncio.gather`` so the client's
    internal semaphore can saturate continuously; this is meaningfully faster
    than chunked submission whose per-chunk tail latency would otherwise
    serialize throughput.

    Returns a list shaped like ``grouped_texts``, where each inner element is
    the ``(rephrased_text, ok)`` tuple for the corresponding input text.
    """
    flat_texts: list[str] = []
    group_lengths: list[int] = []
    for group in grouped_texts:
        flat_texts.extend(group)
        group_lengths.append(len(group))

    pbar = tqdm(total=len(flat_texts), desc=desc)

    async def _with_progress(text: str) -> tuple[str, bool]:
        result = await client.paraphrase(text)
        pbar.update(1)
        return result

    try:
        flat_results = await asyncio.gather(*(_with_progress(t) for t in flat_texts))
    finally:
        pbar.close()

    grouped_results: list[list[tuple[str, bool]]] = []
    cursor = 0
    for length in group_lengths:
        grouped_results.append(list(flat_results[cursor : cursor + length]))
        cursor += length
    return grouped_results


def add_client_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard paraphrase-client CLI flags to a parser."""
    parser.add_argument("--base_url", type=str, default="http://localhost:9999/v1")
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key; defaults to OPENAI_API_KEY env var if set, else 'EMPTY' (vLLM convention).",
    )
    parser.add_argument("--model", type=str, default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff_base", type=float, default=1.0)
    parser.add_argument("--backoff_cap", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)


def client_config_from_args(args: argparse.Namespace) -> ParaphraseClientConfig:
    api_key = args.api_key if args.api_key is not None else os.environ.get("OPENAI_API_KEY", "EMPTY")
    return ParaphraseClientConfig(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        retries=args.retries,
        backoff_base=args.backoff_base,
        backoff_cap=args.backoff_cap,
        seed=args.seed,
    )
