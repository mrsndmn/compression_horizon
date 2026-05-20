#!/usr/bin/env bash
# Reproduce paper Table 11, row "Pythia160m / Full" with paper-fidelity settings:
#   compressed_tokens = 32, information_gain_bits = 105 ± 20, final_convergence = 0.684 ± 0.175.
#
# Paper-fidelity parameters (Appendix A + scripts/jobs/run_training.py):
#   - 50 samples from LarryLovestein/pg19_1k (paper repo's reproducible PG19 mirror)
#   - max_sequence_length = 32 (fixed budget for Pythia-160M Full)
#   - 10,000 optimization steps per sample
#   - AdamW lr=0.01, beta1=beta2=0.9, weight_decay=0.01, cosine_with_min_lr scheduler (min_lr=1e-3)
#   - random0.02 init, single compression token (number_of_mem_tokens=1)
#   - cross-entropy loss only (no activation alignment, no low-dim projection)
#
# Time on a single A100: ~5-10 min.
#
# For closest-possible match to paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# Otherwise the script auto-falls-back to PyTorch's sdpa attention (numerically very close).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="full_cramming/pythia_160m"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

uv run python scripts/thesis_reproduction/train.py \
  --model_checkpoint EleutherAI/pythia-160m \
  --dataset_name LarryLovestein/pg19_1k \
  --max_sequence_length 32 \
  --limit_dataset_items 50 \
  --per_device_train_batch_size 50 \
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
