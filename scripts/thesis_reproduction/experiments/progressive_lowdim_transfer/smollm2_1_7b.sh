#!/usr/bin/env bash
# Transfer test for the shared low-dim basis learned in
# ``progressive_lowdim_global/smollm2_1_7b.sh``.
#
# Pipeline:
#   1) Source run (progressive_lowdim_global, ALREADY MUST BE DONE) trains a
#      single nn.Linear(256, 2048) continuously across 50 PG19 texts
#      (indices [0, 50)). Output: low_dim_projection.pt — the "common
#      compression basis".
#   2) THIS run loads that frozen basis, sets requires_grad=False on
#      both W and b, and only optimizes the per-sample coefficients z_j
#      on a DISJOINT slice of PG19 (indices [50, 100), guaranteed by
#      --offset_dataset_items 50). Nothing in W,b moves.
#
# What this experiment measures: whether the 256-dim subspace learned on
# samples [0,50) is **universal for the PG19 domain**, i.e. whether new
# unseen texts can be compressed in the SAME subspace with no extra
# projection-side training. If cram_tokens / IG land close to the source
# run (~957 / ~3271), this is direct empirical evidence for the manifold
# hypothesis in Section 4.3: compression embeddings really do live on a
# low-dim manifold, and its basis is corpus-level, not text-specific.
#
# Slice disjointness guarantee (no leakage):
#   - tokenization.py:75-89 implements ``Dataset.select(range(offset, offset+limit))``
#     with no shuffle, so [0, 50) and [50, 100) share zero rows.
#   - The cache hash in tokenization.py:25 incorporates offset_dataset_items,
#     so the source-run and transfer-run datasets live in different cache files.
#
# Paper-fidelity parameters: identical to progressive_lowdim_global except
# for the frozen-projection / disjoint-slice deltas (so we isolate the
# transfer effect cleanly).
#
# Time on a single A100 80GB: ~6-10 h (similar to the source run, since
# z-optimization dominates wall time; the projection step is now a no-op).
#
# For closest-possible match to paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# Otherwise the script auto-falls-back to PyTorch sdpa attention.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="progressive_lowdim_transfer/smollm2_1_7b"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"
SOURCE_DIR="artifacts/thesis_reproduction/progressive_lowdim_global/smollm2_1_7b"
CHECKPOINT="${SOURCE_DIR}/low_dim_projection.pt"

# Fail early with a clear message if the source run hasn't produced the basis
# yet — otherwise we'd start an 8-hour run only to crash deep inside trainer.
if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: source basis not found at $CHECKPOINT" >&2
  echo "" >&2
  echo "Run the source experiment first:" >&2
  echo "  bash scripts/thesis_reproduction/experiments/progressive_lowdim_global/smollm2_1_7b.sh" >&2
  echo "" >&2
  echo "It produces ${CHECKPOINT} (the shared 256-dim basis trained on PG19 [0, 50))." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# tee duplicates stdout+stderr to ${OUTPUT_DIR}/train.log so terminal output
# survives session closes — handy when comparing results months later.
{
  echo "Loading FROZEN projection from: $CHECKPOINT"
  echo "Source slice:  PG19 [0, 50)   (training of W, b + per-sample z_j)"
  echo "Transfer slice: PG19 [50, 100)  (only per-sample z_j optimized; W, b frozen)"
  echo

  uv run python scripts/thesis_reproduction/train.py \
    --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \
    --dataset_name LarryLovestein/pg19_1k \
    --max_sequence_length 4096 \
    --limit_dataset_items 50 \
    --offset_dataset_items 50 \
    --per_device_train_batch_size 1 \
    --max_optimization_steps_per_sample 10000 \
    --max_optimization_steps_per_token 1000 \
    --learning_rate 0.1 \
    --warmup_steps 100 \
    --embedding_init_method random0.02 \
    --loss_type cross_entropy \
    --low_dim_projection \
    --low_dim_projection_checkpoint "$CHECKPOINT" \
    --low_dim_projection_train False \
    --low_dim_size 256 \
    --progressive_train \
    --progressive_min_seq_len 1 \
    --progressive_step 1 \
    --progressive_convergence_threshold 1.0 \
    --dtype bf16 \
    --output_dir "$OUTPUT_DIR"

  echo
  echo "==============================================================================="
  echo "Transfer training finished."
  echo
  echo "Compare three rows side-by-side to interpret the result:"
  echo "  progressive_lowdim/smollm2_1_7b           — per-sample basis, paper Table 6"
  echo "  progressive_lowdim_global/smollm2_1_7b    — shared basis, learned on [0, 50)"
  echo "  progressive_lowdim_transfer/smollm2_1_7b  — FROZEN shared basis on [50, 100)  ← this run"
  echo
  echo "Interpretation:"
  echo "  * transfer ≈ shared ≈ per-sample   → manifold hypothesis (Section 4.3) is"
  echo "    empirically confirmed: a 256-dim corpus-level basis suffices for unseen texts."
  echo "  * transfer << shared               → basis must be sample-specialized to reach"
  echo "    paper-faithful numbers; the manifold is text-conditional, not corpus-wide."
  echo "==============================================================================="
  uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
} 2>&1 | tee "${OUTPUT_DIR}/train.log"
