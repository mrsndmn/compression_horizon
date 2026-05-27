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
  tab:layer_ablation
  tab:init_ablation
  tab:added_tokens_ablation
)

for name in "${PROGRESSIVE_TABLES[@]}"; do
  "$PY" scripts/paper/tables/progressive.py --name "$name" --save
done

# --- Surprisal vs. steps-to-converge -----------------------------------------
# tab:surprisal_steps_correlation. Renders from the per-checkpoint
# surprisal_steps_cache.json files (no GPU). Regenerate those caches once with
# `--compute` (needs a GPU + the four base models) before this will reflect new runs.
"$PY" scripts/paper/tables/surprisal_steps_correlation.py --save

# --- Solution diversity across learning rates --------------------------------
# tab:solution_diversity. Renders from artifacts/paper/solution_diversity/<key>.json.
# Regenerate those caches once with `--compute` (CPU only; reads the LR-sweep
# trajectory datasets) before this will reflect new runs.
"$PY" scripts/paper/tables/solution_diversity.py --save

# --- Trajectory cluster-structure: dwelling basins vs heavy-tailed wandering --
# tab:trajectory_cluster_structure (+ _lr) and paper/figures/trajectory_cluster_dwelling_vs_scale.pdf.
# Renders from the per-run summary.json caches written by
# scripts/analyze_trajectory_clusters.py; regenerate those once per run (CPU only;
# reads the saved trajectories) before this will reflect new runs.
"$PY" scripts/paper/tables/trajectory_clusters.py --save --figure

# --- Full vs. progressive cramming -------------------------------------------
# tab:full_vs_progressive
"$PY" scripts/paper/tables/full_cramming_table.py --save-dir paper/tables
# tab:full_vs_progressive_appendix
"$PY" scripts/paper/tables/full_cramming_table.py --type full_cramming_apendix --save-dir paper/tables
# tab:prefix_tuning_accuracy
"$PY" scripts/paper/tables/full_cramming_table.py --type prefix_tuning --save-dir paper/tables

# --- Semantic benchmark evaluation -------------------------------------------
# tab:semantic_evaluation
"$PY" scripts/paper/tables/bench_semantic_results.py --tablefmt latex --save-dir paper/tables --arc-split ARC-Easy

# --- Attention hijacking -----------------------------------------------------
# tab:attn_hijacking (progressive cramming, all model families)
"$PY" scripts/paper/tables/attn_hijacking.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-1B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-3B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_nobos_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-160m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-410m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_nobos_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-135M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-360M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_nobos_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-270m_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-1b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_nobos_lr_0.1/progressive_prefixes \
  --compute --midrule_indicies 3 7 11 --tablefmt latex \
  --save-dir paper/tables --save-name attn_hijacking

# tab:prefix_tuning_attention_hijacking (prefix-tuned baselines)
"$PY" scripts/paper/tables/attn_hijacking.py \
  --checkpoints \
    artifacts/experiments_prefix_tuning/pt_sl_1024_Llama-3.2-3B/prefix_tuning_prefixes \
    artifacts/experiments_prefix_tuning/pt_sl_1024_SmolLM2-135M/prefix_tuning_prefixes \
    artifacts/experiments_prefix_tuning/pt_sl_1024_SmolLM2-360M/prefix_tuning_prefixes \
    artifacts/experiments_prefix_tuning/pt_sl_1024_SmolLM2-1.7B/prefix_tuning_prefixes \
    artifacts/experiments_prefix_tuning/pt_sl_1024_Qwen3-4B/prefix_tuning_prefixes \
  --compute --tablefmt latex \
  --save-dir paper/tables --save-name prefix_tuning_attention_hijacking
