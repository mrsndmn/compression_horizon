#!/usr/bin/env bash
# Reproduce paper Table 6 row "SmolLM2-1.7B lr=0.1, low-dim dim=256"
# (Progressive cramming on PG19 with a learned rank-k projection):
#   compressed_tokens     = 957.4 ± 142.5
#   information_gain_bits = 3271  ± 309
# i.e. low-dim ~ 3x increase over the no-projection baseline (338.3 / 1213).
# The reduction of the search space to a 256-dim subspace acts as a strong
# regularizer and lets each sample cram far longer prefixes.
#
# Why per_device_train_batch_size = 1 here (vs 25 in the baseline):
#   - Paper-faithful: `scripts/jobs/run_jobs_progressive.py:230` in commit
#     a0d39f6 hard-codes "--per_device_train_batch_size 1" for the low-dim
#     progressive jobs. With bs=1 + global=False, every sample gets its own
#     freshly-initialized projection W ∈ R^{hidden×256} that is trained from
#     scratch on that single text — effectively a per-sample basis.
#   - With bs>1 + global=False the same W would be shared across the batch
#     (broadcast over the batch axis of [batch, num_comp, low_dim]); this is
#     a *different* regime and not what Table 6 reports.
#
# Paper-fidelity parameters (Appendix A + Section 4.3 + Section 5.2):
#   - 50 samples from LarryLovestein/pg19_1k (paper repo's reproducible mirror).
#   - max_sequence_length = 4096 (paper cap).
#   - learning_rate = 0.1 (best LR per paper Section 5.2 sweep).
#   - 10,000 optimization steps per sample, capped at 1,000 per newly added token.
#   - random0.02 init, single compression token, cross-entropy loss only.
#   - low_dim_projection enabled, low_dim_size=256, projection is trained
#     (low_dim_projection_train defaults to True), no checkpoint warm-start.
#   - low_dim_projection_global is OFF — projection is rebuilt fresh for each
#     sample (because bs=1). For the alternative "shared basis across whole
#     dataset" mode see progressive_lowdim_global/smollm2_1_7b.sh.
#   - AdamW beta1=beta2=0.9, weight_decay=0.01, cosine_with_min_lr (min_lr=1e-3).
#
# Time on a single A100 80GB: ~6-10 h (low-dim crams ~3x longer prefixes per
# sample than baseline, and we now process 50 samples one at a time instead
# of 2 batches of 25 — expect roughly an order of magnitude more wall time).
#
# For closest-possible match to paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# Otherwise the script auto-falls-back to PyTorch's sdpa attention.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="progressive_lowdim/smollm2_1_7b"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"
mkdir -p "$OUTPUT_DIR"

# tee duplicates stdout+stderr to a log file so the terminal-output is
# preserved alongside ``progressive_prefixes/`` and ``analysis_summary.json``
# — handy if the terminal closes mid-run or you come back to numbers months
# later for the thesis writeup.
{
  uv run python scripts/thesis_reproduction/train.py \
    --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \
    --dataset_name LarryLovestein/pg19_1k \
    --max_sequence_length 4096 \
    --limit_dataset_items 50 \
    --per_device_train_batch_size 1 \
    --max_optimization_steps_per_sample 10000 \
    --max_optimization_steps_per_token 1000 \
    --learning_rate 0.1 \
    --warmup_steps 100 \
    --embedding_init_method random0.02 \
    --loss_type cross_entropy \
    --low_dim_projection \
    --low_dim_size 256 \
    --progressive_train \
    --progressive_min_seq_len 1 \
    --progressive_step 1 \
    --progressive_convergence_threshold 1.0 \
    --dtype bf16 \
    --output_dir "$OUTPUT_DIR"

  echo
  echo "==============================================================================="
  echo "Training finished. Comparing with paper Table 6 (SmolLM2-1.7B, dim=256)..."
  echo "==============================================================================="
  uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
} 2>&1 | tee "${OUTPUT_DIR}/train.log"
