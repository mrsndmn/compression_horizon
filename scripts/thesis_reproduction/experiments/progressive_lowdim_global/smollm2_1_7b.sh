#!/usr/bin/env bash
# **NOT paper Table 6** — this is an *additional* experiment exploring the
# paper's central claim that the compression embedding lives on a shared
# low-dimensional manifold (Section 4.3, Section 5.2).
#
# Difference from progressive_lowdim/smollm2_1_7b.sh:
#   - This run uses --low_dim_projection_global, which means ONE nn.Linear
#     ∈ R^{hidden×256} lives through the whole 50-sample dataset.  Its
#     AdamW state and cosine LR-scheduler accumulate continuously — every
#     sample contributes gradients to the same W.  At the end we save the
#     trained W as `low_dim_projection.pt`, which encodes the *common
#     compression basis* learned across all 50 PG19 texts.
#
# What this tests:
#   1. **Existence of a shared basis.**  If cram_tokens / IG with global=True
#      come out close to paper Table 6's per-sample number (957.4 / 3271),
#      that's direct evidence the same 256-dim basis suffices for all texts —
#      the paper's manifold hypothesis is borne out.  If they collapse, the
#      basis has to be heavily per-sample-specialized.
#   2. **Transferability of the basis.**  The saved `low_dim_projection.pt`
#      can be loaded into a downstream run with --low_dim_projection_checkpoint
#      to test how many cram tokens a *frozen* basis (--no_low_dim_projection_train)
#      buys on a held-out text set.  Real "compression-basis transfer learning".
#
# Why --per_device_train_batch_size 1 here:
#   With global=True the *ideal* optimization target is "find a 256-dim
#   subspace simultaneously good for ALL 50 texts" — the right way to do
#   that is mini-batch gradient averaging at bs > 1. Two earlier attempts
#   confirmed that the right batch size on A100 80GB for this setup
#   doesn't exist:
#     - bs=50 (whole dataset in one batch): OOM at the start (~18 GB short).
#     - bs=25 (two-batch averaging): ran for 39 hours, OOM on the last
#       still-active sample at seq_len≈1154 (attention activations on a
#       25 × 1154² × num_layers tensor + CE forward exceed 80 GB even after
#       24/25 samples had been marked skipped, because the trainer keeps
#       running forward on the full batch).
#
#   So we accept the bs=1 trade-off for this *additional* (non-paper)
#   experiment: ONE Linear lives through the whole run, but it sees
#   samples sequentially — catastrophic forgetting in pure form. Sample 0
#   trains W under text #0 for 10k steps, sample 1 then pushes W away to
#   fit text #1, etc. Sample 49 sees a W trained on 49 prior texts;
#   sample 0 effectively saw a near-random W. The resulting metrics are
#   a noisier estimator of the "shared basis" question than bs=N mini-
#   batches would have been, but the alternative (bs>1) doesn't fit in
#   memory at the high seq_len's progressive cramming hits on SmolLM2-1.7B.
#
#   This is acceptable because the experiment is supplementary to the
#   paper-faithful row (progressive_lowdim/smollm2_1_7b, bs=1, no global).
#   If the shared basis is corpus-universal, even a bs=1 sequential run
#   should land in the same ballpark as the per-sample-basis paper row.
#
# All other paper-fidelity parameters (Appendix A) identical to the
# non-global script: lr=0.1, max_seq_len=4096, 10k steps/sample, 1k steps/
# token, random0.02 init, single compression token, cross-entropy loss,
# cosine_with_min_lr.
#
# Time on a single A100 80GB: ~6-10 h (same order as the non-global
# variant, since wall time is dominated by per-sample forward passes
# and bs=1 means each sample is processed in turn).
#
# PYTORCH_ALLOC_CONF=expandable_segments:True reduces allocator
# fragmentation. It costs nothing in correctness and can shave a couple
# GB of headroom. (`PYTORCH_CUDA_ALLOC_CONF` was the old name; recent
# torch versions emit a deprecation warning.)
#
# For closest-possible match to paper, install flash-attn first:
#   uv pip install flash-attn --no-build-isolation
# Otherwise the script auto-falls-back to PyTorch's sdpa attention.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="progressive_lowdim_global/smollm2_1_7b"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"
mkdir -p "$OUTPUT_DIR"

# Reduces CUDA allocator fragmentation — helps fit large batches without
# changing semantics. See PyTorch's note in the OOM traceback hint.
# (PYTORCH_CUDA_ALLOC_CONF is the deprecated name; both work for now.)
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# tee duplicates stdout+stderr to ``train.log`` in the output dir so the
# terminal-output is preserved alongside the shared-basis artifact
# (``low_dim_projection.pt``), the raw per-stage rows
# (``progressive_prefixes/``), and the JSON summary
# (``analysis_summary.json``).  Convenient for the thesis writeup when you
# come back to numbers months later.
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
    --low_dim_projection_global \
    --low_dim_size 256 \
    --progressive_train \
    --progressive_min_seq_len 1 \
    --progressive_step 1 \
    --progressive_convergence_threshold 1.0 \
    --dtype bf16 \
    --output_dir "$OUTPUT_DIR"

  echo
  echo "==============================================================================="
  echo "Training finished. Run-shared basis saved to:"
  echo "  ${OUTPUT_DIR}/low_dim_projection.pt"
  echo
  echo "Compare with the per-sample-basis variant (progressive_lowdim/smollm2_1_7b.sh)"
  echo "to see whether a SHARED 256-dim basis suffices for all 50 texts (paper's"
  echo "manifold-hypothesis test, Section 4.3 + Section 5.2)."
  echo "==============================================================================="
  uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
} 2>&1 | tee "${OUTPUT_DIR}/train.log"
