#!/usr/bin/env bash
# Reproduce paper Figure 5 (PCA reconstruction ablation) on SmolLM2-135M.
#
# Pure post-hoc analysis: reads the per-stage embeddings saved by
# experiments/progressive/smollm2_135m.sh and:
#   1. fits PCA on each sample's full trajectory of stage embeddings;
#   2. reconstructs the final converged embedding e* from the top-k
#      principal components for k in a grid;
#   3. measures teacher-forced reconstruction accuracy of e*_k on the
#      original prefix;
#   4. averages accuracy(k) across samples → Figure-5-style curve.
#
# Paper claim (page 6): for Llama-3.1-8B, PCA-99% ≈ 74 components but full
# reconstruction needs significantly more. For SmolLM2-135M our trajectory
# analyzer gave PCA-99% ≈ 11; we expect accuracy at k=11 to still be below
# near-perfect, and saturating only at larger k.
#
# Cost: minutes on a single A100 (~450 forwards: 50 samples × ~9 k-values).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="pca_reconstruction/smollm2_135m_progressive"
SOURCE_EXPERIMENT="progressive/smollm2_135m"
SOURCE_DIR="artifacts/thesis_reproduction/${SOURCE_EXPERIMENT}"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

if [ ! -d "${SOURCE_DIR}/progressive_prefixes" ]; then
  echo "ERROR: ${SOURCE_DIR}/progressive_prefixes does not exist."
  echo "Run scripts/thesis_reproduction/experiments/progressive/smollm2_135m.sh first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

uv run python scripts/thesis_reproduction/run_pca_reconstruction.py \
  --source_dir "$SOURCE_DIR" \
  --model_checkpoint HuggingFaceTB/SmolLM2-135M \
  --output_dir "$OUTPUT_DIR" \
  --num_samples 50 \
  --k_grid 1,2,4,8,11,16,24,32,48 \
  --dtype bf16

echo
echo "==============================================================================="
echo "Analysis finished. Qualitative gate vs paper Figure 5..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
