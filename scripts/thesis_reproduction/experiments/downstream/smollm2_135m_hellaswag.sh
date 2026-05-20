#!/usr/bin/env bash
# Reproduce paper Table 10 / Section 5.6 (HellaSwag column) on SmolLM2-135M.
#
# Full Cramming on the HellaSwag ctx of 50 validation instances, then 4-way
# multiple-choice scoring under all 8 PPL variants from Table 10 of the paper:
#   1. baseline                — full PPL of (context + ending)
#   2. baseline_endings        — PPL of endings only
#   3. compression             — full PPL with compression token prepended
#   4. compression_edge        — same + include the comp→first-prefix logit (off-by-one fix)
#   5. compression_endings     — endings-only PPL with compression context
#   6. compression_only        — full PPL with compression *replacing* context
#   7. compression_only_edge   — same + include comp→first-ending logit
#   8. compression_only_endings— endings-only PPL with compression replacing context
#
# Paper SmolLM2-1.7B row of Table 10 (perfectly-reconstructed subset):
#   Baseline=52.41%, Baseline endings=53.30%, Compression=36.47%,
#   Compression endings=40.70%, Compression only=34.57%.
# (Paper has no SmolLM2-135M row; comparison is qualitative.)
#
# Pass --only_full_convergence (uncomment below) to match Table 10's subset
# definition exactly; otherwise the report uses all 50 samples (Table 5 style).
#
# Cost: ~20-40 min on a single A100 (50 prefixes × 5000 cram steps).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

EXPERIMENT="downstream/smollm2_135m_hellaswag"
OUTPUT_DIR="artifacts/thesis_reproduction/${EXPERIMENT}"

mkdir -p "$OUTPUT_DIR"

uv run python scripts/thesis_reproduction/run_downstream_eval.py \
  --benchmark hellaswag \
  --model_checkpoint HuggingFaceTB/SmolLM2-135M \
  --output_dir "$OUTPUT_DIR" \
  --num_samples 50 \
  --max_sequence_length 256 \
  --max_optimization_steps 5000 \
  --learning_rate 0.1 \
  --embedding_init_method random0.02 \
  --dtype bf16 \
  --only_full_convergence

echo
echo "==============================================================================="
echo "Analysis finished. Comparison vs paper Table 10 (qualitative)..."
echo "==============================================================================="
uv run python scripts/thesis_reproduction/analyze.py --experiment "$EXPERIMENT"
