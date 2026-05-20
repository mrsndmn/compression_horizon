#!/usr/bin/env bash
# Reproduce paper Table 6 row "SmolLM2-1.7B lr=0.1" (Progressive on PG19,
# 50 samples, no alignment, no low-dim — baseline progressive cramming):
#   compressed_tokens     = 338.3 ± 69.1
#   information_gain_bits = 1213  ± 166
#   trajectory_length     = 1056  ± 153
#   PCA 99%               = 33.16 ± 2.76
#
# This is the *same-model baseline* for the activation-alignment comparison
# (see progressive_alignment/smollm2_1_7b.sh, paper row 'α=1.0 L=8' has
# 242.6 / 880 / 894 / 26.46 — a ~27% capacity reduction from this baseline).
#
# Paper-fidelity parameters (Appendix A):
#   - 50 samples from LarryLovestein/pg19_1k (paper repo's reproducible mirror).
#   - max_sequence_length = 4096 (paper cap; SmolLM2-1.7B's natural cram is ~340).
#   - learning_rate = 0.1.
#   - 10,000 optimization steps per sample, capped at 1,000 per newly added token.
#   - random0.02 init, single compression token, cross-entropy loss only
#     (no activation alignment, no low-dim projection).
#   - AdamW beta1=beta2=0.9, weight_decay=0.01, cosine_with_min_lr (min_lr=1e-3).
#
# Time on a single A100 80GB: ~2-3 h (Progressive on 1.7B with ~338 stages
# average per sample; early-stopping per converged stage helps).
#
# For closest-possible match to paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# Otherwise the script auto-falls-back to PyTorch's sdpa attention.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="progressive/smollm2_1_7b"
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
  --loss_type cross_entropy \
  --progressive_train \
  --progressive_min_seq_len 1 \
  --progressive_step 1 \
  --progressive_convergence_threshold 1.0 \
  --dtype bf16 \
  --output_dir "$OUTPUT_DIR"

echo
echo "==============================================================================="
echo "Training finished. Comparing with paper Table 6 (SmolLM2-1.7B, no alignment)..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
