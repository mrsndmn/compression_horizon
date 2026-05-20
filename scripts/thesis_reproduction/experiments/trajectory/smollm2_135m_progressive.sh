#!/usr/bin/env bash
# Reproduce paper Table 13 columns "Trajectory Length" + "PCA 99%" for
# SmolLM2-135M progressive cramming (Section 5.1).
#
# Pure post-hoc analysis: reads the per-stage embeddings already saved by
# experiments/progressive/smollm2_135m.sh under progressive_prefixes/ and
# computes:
#   L_traj  (eq. 3): sum_{k=1..n-1} ||e^(k+1) - e^(k)||_2
#   PCA 99%        : min #components reaching 99% cumulative variance per sample,
#                     averaged across samples.
#
# Paper SmolLM2-135M lr=0.1 row of Table 13:
#   Trajectory Length = 178 ± 40
#   PCA 99%           = 12.2 ± 2.96
#
# Runtime: seconds (no GPU required, no model loading).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="trajectory/smollm2_135m_progressive"
SOURCE_EXPERIMENT="progressive/smollm2_135m"
SOURCE_DIR="artifacts/thesis_reproduction/${SOURCE_EXPERIMENT}"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

if [ ! -d "${SOURCE_DIR}/progressive_prefixes" ]; then
  echo "ERROR: ${SOURCE_DIR}/progressive_prefixes does not exist."
  echo "Run scripts/thesis_reproduction/experiments/progressive/smollm2_135m.sh first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

uv run python scripts/thesis_reproduction/run_trajectory.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples 50

echo
echo "==============================================================================="
echo "Analysis finished. Comparing with paper Table 13..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
