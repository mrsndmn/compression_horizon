#!/usr/bin/env bash
#
# All progressive cramming paper tables are now described in code inside
# scripts/paper/tables/progressive.py and addressed by their LaTeX label.
# This wrapper just renders each label in turn.
#
# To list available tables:
#   PYTHONPATH=./src:. python scripts/paper/tables/progressive.py --list
# To render a single table:
#   PYTHONPATH=./src:. python scripts/paper/tables/progressive.py --name tab:low_dim_projection_results

set -x

export PYTHONPATH="./src:.${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-python}"

TABLES=(
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

for name in "${TABLES[@]}"; do
  "$PY" scripts/paper/tables/progressive.py --name "$name" --save
done
