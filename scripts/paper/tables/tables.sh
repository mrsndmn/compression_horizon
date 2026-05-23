#!/usr/bin/env bash
#
# Render every paper table into paper/tables/<slug>.tex.
#
# Progressive cramming tables live in code inside scripts/paper/tables/progressive.py
# and are addressed by their LaTeX label. Other tables (full vs. progressive,
# prefix tuning, semantic evaluation) have their own generator scripts.
#
# To list available progressive tables:
#   PYTHONPATH=./src:. python scripts/paper/tables/progressive.py --list
# To render a single progressive table:
#   PYTHONPATH=./src:. python scripts/paper/tables/progressive.py --name tab:low_dim_projection_results --save

set -x

export PYTHONPATH="./src:.${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-python}"

# --- Progressive cramming tables ---------------------------------------------
PROGRESSIVE_TABLES=(
  tab:all_learning_rates
  tab:progressive_for_model_scales
  tab:lr_sweep_llama_3.1_8b
  tab:full_llama_3.1_8b
  tab:full_pythia_1.4b
  tab:full_smollm2_1.7b
  tab:full_qwen3_4b
  tab:low_dim_projection_results
  tab:full_activation_alignment_and_low_dim_projections
  tab:all_progressive_modifications
  tab:progressive_no_bos_token
)

for name in "${PROGRESSIVE_TABLES[@]}"; do
  "$PY" scripts/paper/tables/progressive.py --name "$name" --save
done

# --- Full vs. progressive cramming -------------------------------------------
# tab:full_vs_progressive
"$PY" scripts/paper/tables/full_cramming_table.py --save-dir paper/tables
# tab:full_vs_progressive_appendix
"$PY" scripts/paper/tables/full_cramming_table.py --type full_cramming_apendix --save-dir paper/tables
# tab:prefix_tuning_accuracy
"$PY" scripts/paper/tables/full_cramming_table.py --type prefix_tuning --save-dir paper/tables

# --- Semantic benchmark evaluation -------------------------------------------
# tab:semantic_evaluation
"$PY" scripts/paper/tables/bench_semantic_results.py --tablefmt latex --save-dir paper/tables
