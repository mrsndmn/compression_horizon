#!/usr/bin/env python3
"""Render the causal attention-knockout figures used in the paper as PDFs.

Reads a HellaSwag evaluation ``results.json`` that contains an
``intervention_summary`` (produced by ``hellaswag_compress_evaluate.py`` with
``--intervention``) and writes two figures to ``paper/figures/``:

- ``attention_knockout_per_layer.pdf``  -- per-layer knockout (downstream +
  teacher-forced reconstruction accuracy, with attention-mass overlay).
- ``attention_knockout_cumulative.pdf`` -- forward and reverse cumulative
  knockout recovery curves.

Defaults target the Llama-3.1-8B run reported in the paper. Regenerate with:

    python scripts/paper/figures_attention_knockout.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bench_hs_results import (
    load_intervention_results,
    plot_cumulative_knockout,
    plot_per_layer_knockout,
)

DEFAULT_RESULTS = "artifacts/hellaswag_evaluation/" "hellaswag_Meta-Llama-3.1-8B_samples_512_lr_0.1_batch_128/results.json"
DEFAULT_FIGDIR = "paper/figures"
DEFAULT_LABEL = "Llama-3.1-8B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="results.json with intervention_summary")
    parser.add_argument("--figdir", default=DEFAULT_FIGDIR, help="output directory for the PDFs")
    parser.add_argument("--model-label", default=DEFAULT_LABEL, help="model label for plot titles")
    args = parser.parse_args()

    data = load_intervention_results(args.results)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    attention_mass = data.get("intervention_summary", {}).get("avg_attention_mass_per_layer")

    plot_per_layer_knockout(
        data,
        str(figdir / "attention_knockout_per_layer.pdf"),
        attention_mass=attention_mass,
        model_label=args.model_label,
    )
    plot_cumulative_knockout(
        data,
        str(figdir / "attention_knockout_cumulative.pdf"),
        model_label=args.model_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
