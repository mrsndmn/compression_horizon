#!/bin/bash
cd /mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/compression_horizon
LOG=/tmp/llama_basin_orchestrator.log
echo "=== orchestrator start $(date) ===" > $LOG

EXP=artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1
PY=python

# Wait for any existing plot_basin compute to finish (GPU is single)
while ps aux | grep plot_basin | grep -v grep | grep -v orchestrator >/dev/null 2>&1; do
  echo "[$(date)] waiting for running compute to free GPU..." >> $LOG
  sleep 60
done

for sid in 3 4 5 6 7 8 9; do
  echo "[$(date)] starting sample $sid" >> $LOG
  $PY scripts/paper/plot_basin_area_vs_stage.py \
    --dataset_path $EXP/progressive_prefixes \
    --sample_ids $sid \
    --num_anchors 8 --mesh_resolution 60 --batch_size 32 --recompute \
    --output /dev/null >> $LOG 2>&1
  echo "[$(date)] finished sample $sid" >> $LOG
done

echo "[$(date)] all per-sample done, regenerating multi-model figure" >> $LOG
$PY scripts/paper/plot_basin_area_vs_stage.py --plot_multi \
    --multi_exp_dirs \
        artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1 \
        artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1 \
        artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.1 \
        artifacts/experiments_progressive/sl_4096_SmolLM2-135M_lr_0.1 \
    --multi_model_names "Llama-3.1-8B" "SmolLM2-1.7B" "Pythia-1.4B" "SmolLM2-135M" \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 \
    --output paper/figures/basin_area_vs_stage_multi.png >> $LOG 2>&1
echo "[$(date)] ORCHESTRATOR COMPLETE" >> $LOG
