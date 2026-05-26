#!/usr/bin/env bash
# Reproduce Table 18 (camera-ready) with 50 samples on mrsndmn/pg19.
#
# Runs 9 trainings (3 models x 3 setups) followed by 9 evals and aggregates the
# results into a Markdown / LaTeX table.
#
# Each training takes 5000 optimization steps per batch on 50 samples and
# requires a GPU. Adjust PER_DEVICE_BATCH if you run out of memory.

set -euo pipefail

LIMIT=${LIMIT:-50}
STEPS=${STEPS:-5000}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-10}
ARTIFACTS=${ARTIFACTS:-artifacts/experiments}

# (model_checkpoint, max_sequence_length)
declare -a MODELS=(
  "unsloth/Llama-3.2-1B:512"
  "unsloth/Llama-3.2-3B:1024"
  "unsloth/Meta-Llama-3.1-8B:1568"
)

SETUPS=(common no_bos 2leading)

train_one() {
  local model="$1"
  local seq_len="$2"
  local setup="$3"

  local extra_flags=()
  case "$setup" in
    common)    : ;;  # defaults
    no_bos)    extra_flags+=(--no_bos_token) ;;
    2leading)  extra_flags+=(--leading_token_loss_weight 3.0 --leading_token_loss_count 2) ;;
    *)         echo "unknown setup: $setup" >&2; exit 1 ;;
  esac

  python scripts/reproduction.py \
    --setup_name "$setup" \
    --model_checkpoint "$model" \
    --max_sequence_length "$seq_len" \
    --max_optimization_steps_per_sample "$STEPS" \
    --limit_dataset_items "$LIMIT" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --dataset_name "LarryLovestein/pg19_1k" \
    --full_cramming_convergence_threshold 0.99 \
    --dtype bfloat16 \
    --learning_rate 0.01 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.9 \
    --number_of_mem_tokens 1 \
    --embedding_init_method random \
    "${extra_flags[@]}"
}

eval_one() {
  local model="$1"
  local seq_len="$2"
  local setup="$3"
  local model_short
  model_short=$(basename "$model")
  local run_dir="${ARTIFACTS}/${model_short}_${seq_len}_${setup}"

  python scripts/eval_table_18.py \
    --compressed_prefixes_path "${run_dir}/compressed_prefixes" \
    --model_checkpoint "$model" \
    --dtype bfloat16
}

for entry in "${MODELS[@]}"; do
  IFS=':' read -r MODEL SEQ_LEN <<<"$entry"
  for SETUP in "${SETUPS[@]}"; do
    echo "==== TRAIN $MODEL  seq=$SEQ_LEN  setup=$SETUP ===="
    train_one "$MODEL" "$SEQ_LEN" "$SETUP"
    echo "==== EVAL  $MODEL  seq=$SEQ_LEN  setup=$SETUP ===="
    eval_one "$MODEL" "$SEQ_LEN" "$SETUP"
  done
done

echo "==== BUILD TABLE ===="
python scripts/build_table_18.py --artifacts_dir "$ARTIFACTS" --format both \
  --output "${ARTIFACTS}/table_18.md"
