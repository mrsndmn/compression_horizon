#!/usr/bin/env bash
# Reproduce paper Table 11, row "Llama-3.2-1B / Full":
#   compressed_tokens = 512, information_gain_bits = 1965 ± 244, final_convergence = 0.998 ± 0.
#
# Why this experiment is a stronger correctness check than Pythia-160M:
#   - paper accuracy 0.998 (near-perfect reconstruction) → very high signal.
#   - paper std = 0 → if our run gives < 0.99, it indicates a real problem with the refactor,
#     not sample variance. Pythia-160M's std=0.175 made it harder to diagnose.
#
# Paper-fidelity parameters (Appendix A + scripts/reproduction.py):
#   - 10 samples from LarryLovestein/pg19_1k (paper repo's reproducible PG19 mirror).
#   - max_sequence_length = 512 (fixed budget for Llama-3.2-1B Full).
#   - 10,000 optimization steps per sample.
#   - AdamW lr=0.01, beta1=beta2=0.9, weight_decay=0.01, cosine_with_min_lr scheduler (min_lr=1e-3).
#   - random0.02 init, single compression token (number_of_mem_tokens=1).
#   - cross-entropy loss only (no activation alignment, no low-dim projection).
#
# Time on a single A100: ~15-25 min.
#
# For closest-possible match to paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# Otherwise the script auto-falls-back to PyTorch's sdpa attention.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="full_cramming/llama_3_2_1b"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

uv run python scripts/thesis_reproduction/train.py \
  --model_checkpoint unsloth/Llama-3.2-1B \
  --dataset_name LarryLovestein/pg19_1k \
  --max_sequence_length 512 \
  --limit_dataset_items 10 \
  --per_device_train_batch_size 10 \
  --max_optimization_steps_per_sample 10000 \
  --learning_rate 0.01 \
  --warmup_steps 100 \
  --embedding_init_method random0.02 \
  --loss_type cross_entropy \
  --dtype bf16 \
  --output_dir "$OUTPUT_DIR"

echo
echo "==============================================================================="
echo "Training finished. Comparing with paper expected values..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
