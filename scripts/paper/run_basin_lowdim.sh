#!/bin/bash
cd /mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/compression_horizon
LOG=/tmp/lowdim_basin_orchestrator.log
echo "=== lowdim orchestrator start $(date) ===" > $LOG
PY=python

# Wait for the non-lowdim Llama orchestrator to FULLY complete (race-free sentinel),
# so two 8B models never co-reside on the single 80GB GPU.
while ! grep -q "ORCHESTRATOR COMPLETE" /tmp/llama_basin_orchestrator.log 2>/dev/null; do
  echo "[$(date)] waiting for non-lowdim Llama orchestrator to complete..." >> $LOG
  sleep 120
done
echo "[$(date)] non-lowdim orchestrator done; GPU free for lowdim" >> $LOG
# extra guard: ensure no compute proc lingering
while ps aux | grep plot_basin | grep -v grep | grep -v orchestrator >/dev/null 2>&1; do
  sleep 30
done

# (key | exp_dir | model_name)  -- fast models first, Llama-8B last
run_one () {
  local key="$1"; local exp="$2"; local name="$3"
  echo "[$(date)] START $key ($exp)" >> $LOG
  $PY scripts/paper/plot_basin_area_vs_stage.py \
    --dataset_path artifacts/experiments_progressive/$exp/progressive_prefixes \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 \
    --num_anchors 8 --mesh_resolution 60 --batch_size 32 --recompute \
    --model_name "$name (lowdim-256)" \
    --output artifacts/analysis/basin_area_lowdim_${key}.png >> $LOG 2>&1
  echo "[$(date)] DONE $key" >> $LOG
}

run_one pythia1p4b   sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.5   "Pythia-1.4B"
run_one smollm2_1p7b sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1  "SmolLM2-1.7B"
run_one llama3p1_8b  sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1 "Llama-3.1-8B"

echo "[$(date)] LOWDIM ORCHESTRATOR COMPLETE" >> $LOG
