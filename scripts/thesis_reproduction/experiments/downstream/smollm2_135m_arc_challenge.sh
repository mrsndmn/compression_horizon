#!/usr/bin/env bash
# Reproduce paper Table 10 / Section 5.6 (ARC-Challenge column) on SmolLM2-135M.
#
# Same protocol as ARC-Easy but on the harder Challenge split. Paper SmolLM2-1.7B
# row of Table 10:
#   Baseline=36.66%, Baseline endings=40.29%, Compression=28.62%,
#   Compression endings=31.36%, Compression only=22.94%.
#
# Cost: ~15-30 min on a single A100.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="downstream/smollm2_135m_arc_challenge"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

mkdir -p "$OUTPUT_DIR"

uv run python scripts/thesis_reproduction/run_downstream_eval.py \
  --benchmark arc-challenge \
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
