#!/usr/bin/env bash
# Run all three Table-10 benchmarks on SmolLM2-1.7B with paper-exact command
# parameters (1024 / full-validation samples, 1000 cramming steps, batch=64,
# --no_bos_token), then print the side-by-side ours-vs-paper summary.
#
# Each per-benchmark run saves BOTH Table-5 (all samples) and Table-10
# (perfectly-reconstructed subset) summaries in its downstream_eval.json.
#
# Total cost: ~4-6 h on a single A100 80GB.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SCRIPT_DIR="scripts/thesis_reproduction/experiments/downstream"

echo "==============================================================================="
echo "[1/3] HellaSwag  (SmolLM2-1.7B, 1024 samples)"
echo "==============================================================================="
bash "${SCRIPT_DIR}/smollm2_1_7b_hellaswag.sh"

echo
echo "==============================================================================="
echo "[2/3] ARC-Easy   (SmolLM2-1.7B, full validation = 570 samples)"
echo "==============================================================================="
bash "${SCRIPT_DIR}/smollm2_1_7b_arc_easy.sh"

echo
echo "==============================================================================="
echo "[3/3] ARC-Challenge (SmolLM2-1.7B, full validation = 299 samples)"
echo "==============================================================================="
bash "${SCRIPT_DIR}/smollm2_1_7b_arc_challenge.sh"

echo
echo "==============================================================================="
echo "Table-10 reproduction summary (SmolLM2-1.7B side-by-side with paper)"
echo "==============================================================================="
uv run python scripts/thesis_reproduction/summarize_downstream_table10.py --model smollm2_1_7b
