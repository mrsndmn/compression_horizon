#!/usr/bin/env bash
# Reproduce paper Table 6 row "SmolLM2-1.7B lr=0.1 α=1.0 L=8" (Progressive +
# Activation Alignment on PG19, 50 samples):
#   compressed_tokens     = 242.6 ± 63.2
#   information_gain_bits = 880   ± 186
#   trajectory_length     = 894   ± 121
#   PCA 99%               = 26.46 ± 2.64
#
# Compare against the no-alignment baseline (same setup, same data) from
# Table 6:
#   compressed_tokens     = 338.3 ± 69.1
#   information_gain_bits = 1213  ± 166
# i.e. activation alignment reduces capacity by ~27% (regularization effect).
#
# Paper-fidelity parameters (Appendix A + Section 5.4):
#   - 50 samples from LarryLovestein/pg19_1k (paper repo's reproducible mirror).
#   - max_sequence_length = 4096 (paper cap).
#   - learning_rate = 0.1.
#   - 10,000 optimization steps per sample, capped at 1,000 per newly added token.
#   - random0.02 init, single compression token.
#   - Activation alignment: loss_type=cosine (paper Section 4.2 formula),
#     hybrid_alpha=1.0, num_alignment_layers=8.
#   - AdamW beta1=beta2=0.9, weight_decay=0.01, cosine_with_min_lr (min_lr=1e-3).
#
# Time on a single A100 80GB: ~2-4 h (Progressive on 1.7B with extra target-
# hidden-states forward per batch; early-stopping helps).
#
# For closest-possible match to paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# Otherwise the script auto-falls-back to PyTorch's sdpa attention.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="progressive_alignment/smollm2_1_7b"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

uv run python scripts/thesis_reproduction/train.py \
  --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \
  --dataset_name LarryLovestein/pg19_1k \
  --max_sequence_length 4096 \
  --limit_dataset_items 50 \
  --per_device_train_batch_size 25 \
  --max_optimization_steps_per_sample 10000 \
  --max_optimization_steps_per_token 1000 \
  --learning_rate 0.1 \
  --warmup_steps 100 \
  --embedding_init_method random0.02 \
  --loss_type cosine \
  --hybrid_alpha 1.0 \
  --num_alignment_layers 8 \
  --progressive_train \
  --progressive_min_seq_len 1 \
  --progressive_step 1 \
  --progressive_convergence_threshold 1.0 \
  --dtype bf16 \
  --output_dir "$OUTPUT_DIR"

echo
echo "==============================================================================="
echo "Training finished. Comparing with paper Table 6 (SmolLM2-1.7B, alignment row)..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
