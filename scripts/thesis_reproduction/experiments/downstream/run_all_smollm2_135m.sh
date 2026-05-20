#!/usr/bin/env bash
# Run all three downstream benchmarks on SmolLM2-135M sequentially and print
# the Table-10 ours-vs-paper summary at the end.
#
# Each per-benchmark step caches its compression_embeddings.pt — re-running
# this script after a partial completion skips already-trained embeddings.
#
# Total cost on a single A100: ~45-90 min (mostly cramming).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SCRIPT_DIR="scripts/thesis_reproduction/experiments/downstream"

echo "==============================================================================="
echo "[1/3] HellaSwag"
echo "==============================================================================="
bash "${SCRIPT_DIR}/smollm2_135m_hellaswag.sh"

echo
echo "==============================================================================="
echo "[2/3] ARC-Easy"
echo "==============================================================================="
bash "${SCRIPT_DIR}/smollm2_135m_arc_easy.sh"

echo
echo "==============================================================================="
echo "[3/3] ARC-Challenge"
echo "==============================================================================="
bash "${SCRIPT_DIR}/smollm2_135m_arc_challenge.sh"

echo
echo "==============================================================================="
echo "Table-10 reproduction summary (all three benchmarks side-by-side)"
echo "==============================================================================="
uv run python scripts/thesis_reproduction/summarize_downstream_table10.py
