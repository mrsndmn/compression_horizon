
set -x

# tab:all_learning_rates
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_1.0/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_1.0/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_1.0/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_1.0/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 3 7 12 \
  --show_labels --only_stat_table --tablefmt latex


# tab:progressive_for_model_scales
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-1B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-3B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-160m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-410m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-135M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-360M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-270m_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-1b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 2 5 8 \
  --show_labels --only_stat_table --tablefmt latex


PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.01/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_1.0/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_5.0/progressive_prefixes \
  --names_mapping "0.01,0.1,0.5,1.0,5.0" \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex


# Full Llama31-8B
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1*/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex

# Full pythia-1.4b
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b*/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex

# Full SmalLM2
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B*/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex

# Full Qwen3-4B
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B*/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex


-----

# Low dim projection experiments tab:low_dim_projection_results
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_64_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_128_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_512_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_32_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_64_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_128_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_512_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_32_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_64_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_128_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_512_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_32_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_64_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_128_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_512_lowproj/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 5 11 17 \
  --show_labels --only_stat_table --tablefmt latex


# Allignment
# Full experiments list tab:full_activation_alignment_and_low_dim_projections
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_2/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_16/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_24/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_32/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_16/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_20/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_16/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_20/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_loss_cosine_hybrid_1.0_align_16/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_loss_cosine_hybrid_1.0_align_20/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_16/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_20/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 9 17 25 33 \
  --show_labels --only_stat_table --tablefmt latex


# tab:all_progressive_modifications
# tab:rebuttle_all_progressive_modifications
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.5_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lowdim_32_lowproj_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lowdim_32_lowproj_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 3 7 11 \
  --show_labels --only_stat_table --tablefmt latex




# NO BOS token
# tab:progressive_no_bos_token
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_nobos_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_nobos_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_nobos_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_nobos_lr_0.1/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex --short
