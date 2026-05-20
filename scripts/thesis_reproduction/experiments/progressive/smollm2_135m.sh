#!/usr/bin/env bash
# Reproduce paper Table 13, row "SmolLM2-135M lr=0.1" (Progressive on PG19):
#   compressed_tokens = 38.5 ± 14.1, information_gain_bits = 168 ± 66
#   (trajectory_length 178 ± 40 and PCA 99% 12.2 ± 2.96 are out-of-scope here —
#   require post-hoc trajectory analysis, added in a follow-up.)
#
# Paper-fidelity parameters (Appendix A + scripts/progressive_experiments_final_repro.sh):
#   - 50 samples from LarryLovestein/pg19_1k (paper repo's reproducible PG19 mirror).
#   - max_sequence_length = 4096 (paper cap; SmolLM2-135M's natural compressed_tokens ~ 38).
#   - learning_rate = 0.1 (Table 13 explicitly tags SmolLM2-135M with lr=0.1).
#   - 10,000 optimization steps per sample, capped at 1,000 per newly added token.
#   - random0.02 init, single compression token, cross-entropy loss only
#     (no activation alignment, no low-dim projection).
#   - AdamW beta1=beta2=0.9, weight_decay=0.01, cosine_with_min_lr scheduler (min_lr=1e-3).
#
# Time on a single A100: ~15-30 min.
#
# For closest-possible match to paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# Otherwise the script auto-falls-back to PyTorch's sdpa attention.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="progressive/smollm2_135m"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

uv run python scripts/thesis_reproduction/train.py \
  --model_checkpoint HuggingFaceTB/SmolLM2-135M \
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
echo "Training finished. Comparing with paper expected values..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
