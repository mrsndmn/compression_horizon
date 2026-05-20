#!/usr/bin/env bash
# Direct same-model reproduction of paper Table 5/10 (ARC-Easy column) using
# SmolLM2-1.7B. Parameters mirror the original paper command exactly:
#
#   python scripts/arc_compress_evaluate.py \
#     --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \
#     --arc_subset ARC-Easy \
#     --num_compression_tokens 1 \
#     --max_optimization_steps 1000 \
#     --learning_rate 0.1 \
#     --batch_size 64 \
#     --dtype bf16 \
#     --loss_type cross_entropy \
#     --num_alignment_layers 0 \
#     --no_bos_token \
#     --no-intervention
#
# No --limit_samples flag → uses the full ARC-Easy validation split
# (570 instances). We pass --num_samples 9999 below; the loader clamps to
# the actual dataset size.
#
# Cost: ~1-2 h on a single A100 80GB.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="downstream/smollm2_1_7b_arc_easy"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

mkdir -p "$OUTPUT_DIR"

uv run python scripts/thesis_reproduction/run_downstream_eval.py \
  --benchmark arc-easy \
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
