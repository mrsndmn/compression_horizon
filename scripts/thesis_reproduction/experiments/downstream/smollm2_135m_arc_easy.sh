#!/usr/bin/env bash
# Reproduce paper Table 10 / Section 5.6 (ARC-Easy column) on SmolLM2-135M.
#
# Same protocol as the HellaSwag script (8 PPL variants, Full Cramming) but
# on ARC-Easy validation. Question becomes the prefix; the four multiple-choice
# options are the candidate endings.
#
# Paper SmolLM2-1.7B row of Table 10:
#   Baseline=68.72%, Baseline endings=53.88%, Compression=55.87%,
#   Compression endings=41.63%, Compression only=30.92%.
#
# Cost: ~15-30 min on a single A100.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="downstream/smollm2_135m_arc_easy"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

mkdir -p "$OUTPUT_DIR"

uv run python scripts/thesis_reproduction/run_downstream_eval.py \
  --benchmark arc-easy \
  --model_checkpoint HuggingFaceTB/SmolLM2-135M \
  --output_dir "$OUTPUT_DIR" \
  --num_samples 50 \
  --max_sequence_length 128 \
  --max_optimization_steps 5000 \
  --learning_rate 0.1 \
  --embedding_init_method random0.02 \
  --dtype bf16 \
  --only_full_convergence

echo
echo "==============================================================================="
echo "Analysis finished. Comparison vs paper Table 10 (qualitative)..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
