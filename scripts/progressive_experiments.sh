
MODEL_NAME=unsloth/Meta-Llama-3.1-8B
PROGRESSIVE_LR=0.1

python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR"
# Low Dim
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --low_dim_projection --low_dim_size 256
# Hybrid alpha
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 4
# Hybrid alpha + LowProj
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 256

# No BOS token
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --no_bos_token

# ==========================

MODEL_NAME=EleutherAI/pythia-1.4b
PROGRESSIVE_LR=0.5

# Simple progressive
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR"
# Low Dim
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --low_dim_projection --low_dim_size 256
# Hybrid alpha
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8
# Hybrid alpha + LowProj
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 256

# No BOS token
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --no_bos_token

# ==========================

MODEL_NAME=HuggingFaceTB/SmolLM2-1.7B
PROGRESSIVE_LR=0.1

python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR"
# Low Dim
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --low_dim_projection --low_dim_size 256
# Hybrid alpha
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8
# Hybrid alpha + LowProj
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 256

# No BOS token
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --no_bos_token

# ==========================

MODEL_NAME=unsloth/gemma-3-4b-pt
PROGRESSIVE_LR=0.1

python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR"
# Low Dim
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --low_dim_projection --low_dim_size 32
# Hybrid alpha
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8
# Hybrid alpha + LowProj
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --hybrid_alpha 1.0 --loss_type cosine --num_alignment_layers 8 --low_dim_projection --low_dim_size 32

# No BOS token
python scripts/jobs/run_jobs_progressive.py --model $MODEL_NAME --limit_dataset_items 50 --dataset_name LarryLovestein/pg19_1k --learning_rate "$PROGRESSIVE_LR" --no_bos_token

