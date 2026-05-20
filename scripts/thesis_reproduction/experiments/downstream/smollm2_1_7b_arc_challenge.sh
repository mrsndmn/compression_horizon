#!/usr/bin/env bash
# Direct same-model reproduction of paper Table 5/10 (ARC-Challenge column)
# using SmolLM2-1.7B. Parameters mirror the original paper command exactly
# (same as ARC-Easy but --arc_subset ARC-Challenge).
#
# Uses the full ARC-Challenge validation split (299 instances).
#
# Cost: ~1 h on a single A100 80GB.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="downstream/smollm2_1_7b_arc_challenge"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

mkdir -p "$OUTPUT_DIR"

uv run python scripts/thesis_reproduction/run_downstream_eval.py \
  --benchmark arc-challenge \
  --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \
  --output_dir "$OUTPUT_DIR" \
  --num_samples 9999 \
  --max_sequence_length 128 \
  --max_optimization_steps 1000 \
  --cram_batch_size 64 \
  --learning_rate 0.1 \
  --num_compression_tokens 1 \
  --loss_type cross_entropy \
  --num_alignment_layers 0 \
  --embedding_init_method random0.02 \
  --dtype bf16 \
  --no_bos_token

echo
echo "==============================================================================="
echo "Analysis finished. Comparison vs paper Table 10 (perfectly-reconstructed subset)..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
