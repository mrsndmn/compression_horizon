#!/usr/bin/env bash
# Causal attention-knockout probe on SmolLM2-135M progressive cramming.
# Paper Section 4.4 + Reviewer 1 W2 response:
#   - per-layer KO degrades reconstruction at early layers but not late layers
#   - forward cumulative KO collapses reconstruction once early layers are masked
#   - reverse cumulative KO does not recover until the masked window reaches early layers
#
# Consumes the embeddings already saved by:
#   scripts/thesis_reproduction/experiments/progressive/smollm2_135m.sh
#
# Cost: ~10-30 min on a single A100 (50 samples * 3 regimes * 30 layers
# under eager attention).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="attention_knockout/smollm2_135m_progressive"
SOURCE_EXPERIMENT="progressive/smollm2_135m"
SOURCE_DIR="artifacts/thesis_reproduction/${SOURCE_EXPERIMENT}"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

if [ ! -d "${SOURCE_DIR}/progressive_prefixes" ]; then
  echo "ERROR: ${SOURCE_DIR}/progressive_prefixes does not exist."
  echo "Run scripts/thesis_reproduction/experiments/progressive/smollm2_135m.sh first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

uv run python scripts/thesis_reproduction/run_attention_knockout.py \
  --source_dir "$SOURCE_DIR" \
  --model_checkpoint HuggingFaceTB/SmolLM2-135M \
  --output_dir "$OUTPUT_DIR" \
  --dtype bf16 \
  --num_samples 50

echo
echo "==============================================================================="
echo "Analysis finished. Qualitative early-vs-late asymmetry gate..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
