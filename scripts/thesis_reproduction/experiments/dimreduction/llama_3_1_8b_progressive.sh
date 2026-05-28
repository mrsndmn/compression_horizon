#!/usr/bin/env bash
# Chapter-3 dimensionality figures for the Llama-3.1-8B progressive trajectory.
#
# Two analyses on the trajectory saved by experiments/progressive/llama_3_1_8b.sh:
#
#   1. run_trajectory_landscape.py -- paper Figure 3: the trajectory projected
#      onto its first two PCA components, drawn over the local accuracy
#      landscape at anchor prefix lengths. Needs the model (teacher-forced
#      accuracy on a grid of plane points) -> GPU, ~10-20 min.
#
#   2. run_dimreduction.py -- non-linear t-SNE / UMAP scatter projections of
#      the same trajectory plus a Two-NN intrinsic-dimension estimate. Pure
#      post-hoc, no model, seconds. t-SNE / UMAP get no accuracy landscape:
#      unlike PCA they are not invertible, so a plane point has no embedding.
#
# Requires the optional 'umap-learn' package (declared in pyproject.toml):
#   uv sync
#
# Smoke test without the heavy Llama run -- point --source_dir at any existing
# Progressive run, e.g. artifacts/thesis_reproduction/progressive/smollm2_135m.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="dimreduction/llama_3_1_8b_progressive"
SOURCE_EXPERIMENT="progressive/llama_3_1_8b"
SOURCE_DIR="artifacts/thesis_reproduction/${SOURCE_EXPERIMENT}"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"
MODEL_CHECKPOINT="unsloth/Meta-Llama-3.1-8B"

if [ ! -d "${SOURCE_DIR}/progressive_prefixes" ]; then
  echo "ERROR: ${SOURCE_DIR}/progressive_prefixes does not exist."
  echo "Run scripts/thesis_reproduction/experiments/progressive/llama_3_1_8b.sh first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[1/2] PCA trajectory + accuracy landscape (paper Figure 3)..."
uv run python scripts/thesis_reproduction/run_trajectory_landscape.py \
  --source_dir "$SOURCE_DIR" \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --prefix_lengths 100,200,400,800,1000 \
  --grid_resolution 50 \
  --dtype bf16

echo
echo "[2/2] Non-linear t-SNE / UMAP projections + Two-NN..."
uv run python scripts/thesis_reproduction/run_dimreduction.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --methods tsne,umap

echo
echo "==============================================================================="
echo "Figures + JSON written to ${OUTPUT_DIR}"
echo "  trajectory_landscape_sample<sid>.png   PCA plane + accuracy landscape"
echo "  dimreduction_sample<sid>_tsne.png      non-linear t-SNE scatter"
echo "  dimreduction_sample<sid>_umap.png      non-linear UMAP scatter"
echo "==============================================================================="
