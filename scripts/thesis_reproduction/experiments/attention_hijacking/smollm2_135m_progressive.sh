#!/usr/bin/env bash
# Reproduce paper Table 3 / Section 5.5 (Attention Hijacking) on SmolLM2-135M.
#
# This script does NOT re-train: it consumes the embeddings produced by
# scripts/thesis_reproduction/experiments/progressive/smollm2_135m.sh, which
# must have completed first (its output_dir is the source_dir below).
#
# What it computes per sample (paper eqs. 7-8):
#   m_l(s) = (1/(s-1)) * sum_{q=1..s-1} A_l(q, 0)
#   m̄_l   = (1/|S|)   * sum_{s in S} m_l(s)
# evaluated twice: with the trained compression embedding prepended, and with
# the BOS embedding prepended. The maximum m̄_l (in %) and Pearson correlation
# between the two profiles map to paper Table 3 columns.
#
# The paper does NOT include a 135M row — comparison with the SmolLM2-1.7B
# reference is qualitative; the verdict is "compression_mass ≥ 30% AND
# correlation ≥ 0.5" (configured in expected.json under `qualitative`).
#
# Time: ~10-20 min on a single A100 (50 samples, eager attention).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="attention_hijacking/smollm2_135m_progressive"
SOURCE_EXPERIMENT="progressive/smollm2_135m"
SOURCE_DIR="artifacts/thesis_reproduction/${SOURCE_EXPERIMENT}"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

if [ ! -d "${SOURCE_DIR}/progressive_prefixes" ]; then
  echo "ERROR: ${SOURCE_DIR}/progressive_prefixes does not exist."
  echo "Run scripts/thesis_reproduction/experiments/progressive/smollm2_135m.sh first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

uv run python scripts/thesis_reproduction/run_attention_hijacking.py \
  --source_dir "$SOURCE_DIR" \
  --model_checkpoint HuggingFaceTB/SmolLM2-135M \
  --output_dir "$OUTPUT_DIR" \
  --dtype bf16 \
  --num_samples 50

echo
echo "==============================================================================="
echo "Analysis finished. Comparing with paper Table 3 (qualitative)..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
