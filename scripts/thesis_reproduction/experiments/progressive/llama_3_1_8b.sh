#!/usr/bin/env bash
# Generate ONE long progressive-cramming trajectory on Llama-3.1-8B for the
# Chapter-3 dimensionality figures (PCA / t-SNE / UMAP projections + Two-NN).
#
# The paper's Figure 3 (optimization trajectory) is a single length-1000
# sequence on Llama-3-8B; this run reproduces that setting so the thesis
# figures are a faithful analogue.
#
# HEAVY: progressive cramming a single PG19 sample on an 8B model runs to
# ~1000+ stages (Llama-3.1-8B progressive ~= 1064 tokens, paper Table 1),
# each stage being its own optimization sub-problem. Budget overnight on a
# single A100 80GB.
#
# Output progressive_prefixes/ feeds the post-hoc analysis:
#   experiments/dimreduction/llama_3_1_8b_progressive.sh
#
# Model: unsloth/Meta-Llama-3.1-8B is a non-gated mirror of meta-llama/
# Llama-3.1-8B; swap to the official id if you have access.
#
# For closest match to the paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# otherwise the script auto-falls-back to PyTorch's sdpa attention.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="progressive/llama_3_1_8b"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

uv run python scripts/thesis_reproduction/train.py \
  --model_checkpoint unsloth/Meta-Llama-3.1-8B \
  --dataset_name LarryLovestein/pg19_1k \
  --max_sequence_length 4096 \
  --limit_dataset_items 1 \
  --per_device_train_batch_size 1 \
  --max_optimization_steps_per_sample 10000 \
  --max_optimization_steps_per_token 1000 \
  --learning_rate 0.1 \
  --warmup_steps 100 \
  --embedding_init_method random0.02 \
  --loss_type cross_entropy \
  --progressive_train \
  --progressive_min_seq_len 1 \
  --progressive_step 1 \
  --progressive_convergence_threshold 1.0 \
  --dtype bf16 \
  --output_dir "$OUTPUT_DIR"

echo
echo "==============================================================================="
echo "Trajectory saved to ${OUTPUT_DIR}/progressive_prefixes/"
echo "Next: bash scripts/thesis_reproduction/experiments/dimreduction/llama_3_1_8b_progressive.sh"
echo "==============================================================================="
