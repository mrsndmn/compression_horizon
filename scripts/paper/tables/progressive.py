"""Generate paper-ready statistics tables for progressive cramming experiments.

Every table previously assembled in ``scripts/paper/tables/tables.sh`` is
encoded below as a :class:`TableSpec`, addressable by its LaTeX label
(``tab:...``). Section breaks are written inline by dropping :data:`MIDRULE`
between checkpoint paths. Run::

    python scripts/paper/tables/progressive.py --list
    python scripts/paper/tables/progressive.py --name tab:low_dim_projection_results
    python scripts/paper/tables/progressive.py --name tab:full_llama_3.1_8b --tablefmt grid
"""

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


class _MidRule:
    """Sentinel placed between checkpoints to insert a LaTeX ``\\midrule``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MIDRULE"


MIDRULE = _MidRule()

CheckpointEntry = Union[str, _MidRule]


from tqdm.auto import tqdm  # noqa: E402  (kept below the public sentinel for readability)

from compression_horizon.paper.tables import (  # noqa: E402
    extract_trajectory,
    format_statistics_table,
    print_statistics_table,
)


@dataclass
class TableSpec:
    """Declarative description of one paper statistics table.

    Each entry in ``checkpoints`` is either a path (string, may contain
    shell-style globs ``*?[``) or the :data:`MIDRULE` sentinel — its
    presence inserts a LaTeX midrule between the preceding and following
    rows in the rendered table.
    """

    name: str
    checkpoints: List[CheckpointEntry]
    names_mapping: Optional[str] = None  # see ``parse_names_mapping``
    short: bool = False
    sample_id: int = 0
    tablefmt: str = "latex"
    # stats key for the "Compressed Tokens" column. ``None`` -> the default
    # ``num_embeddings`` (number of converged stages). Tables that ablate the
    # progressive step (Δ tokens added per converged stage) set this to
    # ``"max_prefix_len"`` so the column reports the achieved prefix length n in
    # actual tokens rather than the stage count (which would be deflated ~Δ×).
    compressed_tokens_key: Optional[str] = None


_EXP = "artifacts/experiments_progressive"


TABLES: List[TableSpec] = [
    TableSpec(
        name="tab:all_learning_rates",
        checkpoints=[
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_1.0/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_pythia-1.4b/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lr_1.0/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lr_1.0/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lr_0.5/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lr_1.0/progressive_prefixes",
        ],
    ),
    TableSpec(
        name="tab:progressive_for_model_scales",
        checkpoints=[
            f"{_EXP}/sl_4096_Llama-3.2-1B_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_Llama-3.2-3B_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_pythia-160m_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-410m_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-135M_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-360M_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_gemma-3-270m_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-1b-pt_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes",
        ],
    ),
    # ---- Disabled: not referenced by paper/example_paper.tex ----
    # Re-enable individually if a future revision of the manuscript needs them.
    # TableSpec(
    #     # ``tables.sh`` listed 5 names for 4 checkpoints — the lr_0.5 entry was
    #     # missing from its --checkpoints flag. Restored here so the sweep is
    #     # complete and the names_mapping length matches.
    #     name="tab:lr_sweep_llama_3.1_8b",
    #     checkpoints=[
    #         f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.01/progressive_prefixes",
    #         f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes",
    #         f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.5/progressive_prefixes",
    #         f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_1.0/progressive_prefixes",
    #         f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_5.0/progressive_prefixes",
    #     ],
    #     names_mapping="0.01,0.1,0.5,1.0,5.0",
    # ),
    # TableSpec(
    #     name="tab:full_llama_3.1_8b",
    #     checkpoints=[f"{_EXP}/sl_4096_Meta-Llama-3.1*/progressive_prefixes"],
    # ),
    # TableSpec(
    #     name="tab:full_pythia_1.4b",
    #     checkpoints=[f"{_EXP}/sl_4096_pythia-1.4b*/progressive_prefixes"],
    # ),
    # TableSpec(
    #     name="tab:full_smollm2_1.7b",
    #     checkpoints=[f"{_EXP}/sl_4096_SmolLM2-1.7B*/progressive_prefixes"],
    # ),
    # TableSpec(
    #     name="tab:full_qwen3_4b",
    #     checkpoints=[f"{_EXP}/sl_4096_Qwen3-4B*/progressive_prefixes"],
    # ),
    # ---- end disabled ----
    TableSpec(
        name="tab:low_dim_projection_results",
        checkpoints=[
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lowdim_64_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lowdim_128_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lowdim_256_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lowdim_512_lowproj/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lowdim_32_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lowdim_64_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lowdim_128_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lowdim_256_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lowdim_512_lowproj/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lowdim_32_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lowdim_64_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lowdim_128_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lowdim_512_lowproj/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lowdim_32_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lowdim_64_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lowdim_128_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lowdim_256_lowproj/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lowdim_512_lowproj/progressive_prefixes",
        ],
    ),
    TableSpec(
        name="tab:full_activation_alignment_and_low_dim_projections",
        checkpoints=[
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_2/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_4/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_16/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_24/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_32/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_4/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_16/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_20/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_4/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_16/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_20/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_gemma-3-4b-pt_loss_cosine_hybrid_1.0_align_4/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_loss_cosine_hybrid_1.0_align_16/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_loss_cosine_hybrid_1.0_align_20/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_4/progressive_prefixes",
            f"{_EXP}/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_16/progressive_prefixes",
            f"{_EXP}/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_20/progressive_prefixes",
            f"{_EXP}/sl_4096_Qwen3-4B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_Qwen3-4B_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_Qwen3-4B_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_Qwen3-4B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
        ],
    ),
    TableSpec(
        # tables.sh ships this under two labels (``tab:all_progressive_modifications``
        # and ``tab:rebuttle_all_progressive_modifications``); aliases are added below.
        name="tab:all_progressive_modifications",
        checkpoints=[
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1_loss_cosine_hybrid_1.0_align_4/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.5_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lowdim_32_lowproj_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lowdim_32_lowproj_lr_0.1_loss_cosine_hybrid_1.0_align_8/progressive_prefixes",
        ],
    ),
    TableSpec(
        name="tab:progressive_no_bos_token",
        checkpoints=[
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_nobos_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_pythia-1.4b_nobos_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_nobos_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_gemma-3-4b-pt_nobos_lr_0.1/progressive_prefixes",
        ],
        short=True,
    ),
    TableSpec(
        # Transformer-depth ablation: SmolLM2-1.7B truncated to its first N and
        # last N decoder layers (N in {1,2,4,8} -> 2/4/8/16 of 24 layers), plus
        # the full-depth (24-layer) baseline as the reference row. Each truncated
        # depth is shown twice -- as-is and after causal-LM finetuning on
        # fineweb-edu ("(finetuned)") -- so the before/after recovery is visible.
        # All runs share the same progressive eval config (pg19_1k, 50, 0.1).
        # Un-finetuned checkpoints: make_first_last_layers_ckpt.py +
        # run_jobs_layer_ablation.py. Finetuning: run_jobs_finetune_truncated.py
        # (now the width/CH recipe -> "-ftw" checkpoints; the old "-ft" eval dirs
        # from the previous recipe are left intact); finetuned eval:
        # run_jobs_layer_ablation_ft.py. The "-ftw" finetuned rows fill in once the
        # in-progress depth retrain finishes (see watch_finetune_truncated.py).
        name="tab:layer_ablation",
        checkpoints=[
            f"{_EXP}/sl_4096_SmolLM2-1.7B-firstlast1_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B-firstlast1-ftw_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-1.7B-firstlast2_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B-firstlast2-ftw_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-1.7B-firstlast4_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B-firstlast4-ftw_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-1.7B-firstlast8_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B-firstlast8-ftw_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
        ],
        names_mapping=(
            "2 layers,2 layers (finetuned),"
            "4 layers,4 layers (finetuned),"
            "8 layers,8 layers (finetuned),"
            "16 layers,16 layers (finetuned),"
            "24 layers (full)"
        ),
    ),
    TableSpec(
        # Initialization ablation: SmolLM2-1.7B with exactly one component randomly
        # re-initialized (transformer layers / LM head / input embeddings), against
        # the fully pretrained reference. Checkpoints built by
        # scripts/checkpoints/make_random_init_ckpt.py and trained via
        # scripts/jobs/run_jobs_init_ablation.py. Row labels must match
        # run_jobs_init_ablation.ROW_LABELS (used by the watcher's trend sentence).
        name="tab:init_ablation",
        checkpoints=[
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B-randinit-layers_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B-randinit-lmhead_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B-randinit-embeddings_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
        ],
        names_mapping="Pretrained,Random transformer layers,Random LM head,Random input embeddings",
    ),
    TableSpec(
        # Compression-head -> progressive-cramming layer ablation. Each row is the progressive
        # eval (canonical benchmark: pg19_1k / sl_4096 / lr_0.1) of one trained compression head,
        # using the head to seed every per-sample embedding (--embedding_init_method
        # compression_head_forward). The Q-Former head is shown at truncated depths
        # firstlast{1,2,4,8} (= 2/4/8/16 of SmolLM2-1.7B's 24 decoder layers) and at full depth,
        # plus the simple MLP head at full depth as a head-architecture reference. This mirrors the
        # depth axis of tab:layer_ablation but with a learned init instead of random per-sample init.
        # Heads trained and evals launched by scripts/jobs/run_jobs_compression_head.py
        # (--stage train / --stage eval). The "ds_fineweb-edu_seq_1024" in each path refers to the
        # head's *training* data, not the eval benchmark (which is pg19_1k for every row).
        name="tab:ch_qformer_layer_ablation",
        checkpoints=[
            f"{_EXP}/progeval_chfwd_ch_head_SmolLM2-1.7B-firstlast1_qformer_q1_l3_h8_ds_fineweb-edu_seq_1024_lr_0.001_a_1.0_b_0.0_unfrozen_v3/progressive_prefixes",
            f"{_EXP}/progeval_chfwd_ch_head_SmolLM2-1.7B-firstlast2_qformer_q1_l3_h8_ds_fineweb-edu_seq_1024_lr_0.001_a_1.0_b_0.0_unfrozen_v3/progressive_prefixes",
            f"{_EXP}/progeval_chfwd_ch_head_SmolLM2-1.7B-firstlast4_qformer_q1_l3_h8_ds_fineweb-edu_seq_1024_lr_0.001_a_1.0_b_0.0_unfrozen_v3/progressive_prefixes",
            f"{_EXP}/progeval_chfwd_ch_head_SmolLM2-1.7B-firstlast8_qformer_q1_l3_h8_ds_fineweb-edu_seq_1024_lr_0.001_a_1.0_b_0.0_unfrozen_v3/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/progeval_chfwd_ch_head_SmolLM2-1.7B_qformer_q1_l3_h8_ds_fineweb-edu_seq_1024_lr_0.001_a_1.0_b_0.0_unfrozen_v3/progressive_prefixes",
            f"{_EXP}/progeval_chfwd_ch_head_SmolLM2-1.7B_mlp_ds_fineweb-edu_seq_1024_lr_0.001_a_1.0_b_0.0_unfrozen_v3/progressive_prefixes",
        ],
        names_mapping=(
            "2 layers (Q-Former),4 layers (Q-Former),8 layers (Q-Former),"
            "16 layers (Q-Former),24 layers (Q-Former),24 layers (MLP head)"
        ),
    ),
    TableSpec(
        # Model-width ablation: depth held fixed at first-4 + last-4 = 8 layers while model
        # width varies across the SmolLM2 family (135M / 360M / 1.7B; hidden 576 / 960 / 2048).
        # Each width is recovered two ways and shown as two rows: (a) plain causal-LM finetuning
        # ("-ftw", evaluated with the baseline random0.02 per-sample init, like tab:layer_ablation)
        # and (b) a Q-Former compression head (evaluated with --embedding_init_method
        # compression_head_forward, like tab:ch_qformer_layer_ablation). All rows share the same
        # progressive eval config (pg19_1k / sl_4096 / lr_0.1). Checkpoints built by
        # make_first_last_layers_ckpt.py --keep 4; finetuning + heads + evals launched by
        # run_jobs_finetune_width.py, run_jobs_compression_head_width.py, run_jobs_width_ablation_ft.py
        # and driven by watch_width_ablation.py. The "ds_fineweb-edu_seq_1024" in the Q-Former paths
        # is the head's *training* data, not the eval benchmark (pg19_1k for every row).
        name="tab:width_ablation",
        checkpoints=[
            f"{_EXP}/sl_4096_SmolLM2-135M-firstlast4-ftw_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/progeval_chfwd_ch_head_SmolLM2-135M-firstlast4_qformer_q1_l3_h8_ds_fineweb-edu_seq_1024_lr_0.001_a_1.0_b_0.0_unfrozen_v3/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-360M-firstlast4-ftw_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/progeval_chfwd_ch_head_SmolLM2-360M-firstlast4_qformer_q1_l3_h8_ds_fineweb-edu_seq_1024_lr_0.001_a_1.0_b_0.0_unfrozen_v3/progressive_prefixes",
            MIDRULE,
            f"{_EXP}/sl_4096_SmolLM2-1.7B-firstlast4-ftw_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/progeval_chfwd_ch_head_SmolLM2-1.7B-firstlast4_qformer_q1_l3_h8_ds_fineweb-edu_seq_1024_lr_0.001_a_1.0_b_0.0_unfrozen_v3/progressive_prefixes",
        ],
        names_mapping=(
            "135M (causal-LM),135M (Q-Former)," "360M (causal-LM),360M (Q-Former)," "1.7B (causal-LM),1.7B (Q-Former)"
        ),
    ),
    TableSpec(
        # Tokens-per-stage ablation: vary the progressive step Δ -- the number of
        # target tokens appended to the prefix each time a stage converges (and then
        # re-compressed) -- over Δ in {1,2,4,8,16,32,64,128} on SmolLM2-1.7B. Δ=1 is
        # the main baseline run (grow one token at a time); Δ>1 runs set
        # --progressive_step Δ and --progressive_min_seq_len Δ so the prefix is always
        # a multiple of Δ. All rows share the canonical progressive eval config
        # (pg19_1k / sl_4096 / lr_0.1). Submitted by run_jobs_added_tokens_ablation.py
        # and driven by watch_ablation.py --launcher run_jobs_added_tokens_ablation.
        # "Compressed Tokens" uses max_prefix_len (achieved prefix length n in actual
        # tokens), so Δ=1 matches the baseline exactly while Δ>1 stays comparable
        # instead of being deflated by the stage count.
        name="tab:added_tokens_ablation",
        checkpoints=[
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1_step_2/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1_step_4/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1_step_8/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1_step_16/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1_step_32/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1_step_64/progressive_prefixes",
            f"{_EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1_step_128/progressive_prefixes",
        ],
        names_mapping=(
            "1 token/stage,2 tokens/stage,4 tokens/stage,8 tokens/stage,"
            "16 tokens/stage,32 tokens/stage,64 tokens/stage,128 tokens/stage"
        ),
        compressed_tokens_key="max_prefix_len",
    ),
]


# Label aliases (tables.sh sometimes references a table under multiple LaTeX labels).
_ALIASES: Dict[str, str] = {
    "tab:rebuttle_all_progressive_modifications": "tab:all_progressive_modifications",
}


TABLES_BY_NAME: Dict[str, TableSpec] = {spec.name: spec for spec in TABLES}
for alias, target in _ALIASES.items():
    TABLES_BY_NAME[alias] = TABLES_BY_NAME[target]


def parse_names_mapping(names_str: Optional[str]) -> Tuple[Dict[str, str], Optional[List[str]]]:
    """Parse a ``path:name,...`` mapping or a positional ``name,...`` list."""
    if names_str is None:
        return {}, None
    if ":" in names_str:
        mapping: Dict[str, str] = {}
        for pair in names_str.split(","):
            if ":" in pair:
                key, value = pair.split(":", 1)
                mapping[key.strip()] = value.strip()
        return mapping, None
    names = [name.strip() for name in names_str.split(",") if name.strip()]
    return {}, names if names else None


def expand_checkpoints(entries: List[CheckpointEntry]) -> Tuple[List[str], List[int]]:
    """Walk a spec's checkpoint list, resolving globs and collecting midrule positions.

    Returns ``(checkpoints, midrule_indicies)`` where ``midrule_indicies`` is
    the row-index list expected by :func:`print_statistics_table` — each value
    ``i`` means "draw a midrule after the ``i``-th rendered row". Sentinels at
    the very start of the list (no preceding entry) or repeated back-to-back
    are deduplicated silently.
    """
    checkpoints: List[str] = []
    midrules: List[int] = []
    for entry in entries:
        if isinstance(entry, _MidRule):
            if not checkpoints:
                continue
            position = len(checkpoints) - 1
            if not midrules or midrules[-1] != position:
                midrules.append(position)
            continue
        if any(ch in entry for ch in "*?["):
            matched = sorted(glob.glob(entry))
            if not matched:
                raise FileNotFoundError(f"Glob {entry!r} matched no checkpoint directories")
            checkpoints.extend(matched)
        else:
            checkpoints.append(entry)
    return checkpoints, midrules


def table_slug(name: str) -> str:
    """Filesystem-safe slug derived from a LaTeX label (drops the ``tab:`` prefix)."""
    slug = name[len("tab:") :] if name.startswith("tab:") else name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", slug).strip("_")


def render_table(
    spec: TableSpec,
    *,
    tablefmt_override: Optional[str] = None,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
) -> None:
    """Compute statistics, print the table, and optionally save it as ``.tex``.

    When ``save_dir`` is set the rendered table is also written to
    ``<save_dir>/<save_name or slug>.tex`` using ``tablefmt='latex'``
    regardless of the spec's default format (so the saved file is always
    LaTeX-compatible).
    """
    path_mapping, positional_names = parse_names_mapping(spec.names_mapping)
    checkpoints, midrule_indicies = expand_checkpoints(spec.checkpoints)

    if positional_names is not None and len(positional_names) != len(checkpoints):
        raise ValueError(
            f"Table {spec.name!r}: names_mapping has {len(positional_names)} entries "
            f"but {len(checkpoints)} checkpoints were resolved"
        )

    missing = [c for c in checkpoints if not os.path.isdir(c)]
    if missing:
        raise FileNotFoundError(f"Table {spec.name!r}: missing checkpoints: {missing}")

    statistics_list: List[dict] = []
    checkpoint_names: List[str] = []
    for idx, checkpoint_path in tqdm(enumerate(checkpoints), desc="Checkpoints", total=len(checkpoints)):
        _, _, stats, _ = extract_trajectory(checkpoint_path, sample_id=spec.sample_id)
        statistics_list.append(stats)

        if positional_names is not None:
            checkpoint_names.append(positional_names[idx])
        elif checkpoint_path in path_mapping:

            checkpoint_names.append(path_mapping[checkpoint_path])
        else:
            name = os.path.basename(os.path.dirname(checkpoint_path))
            if not name or name == ".":
                name = os.path.basename(checkpoint_path)
            checkpoint_names.append(name)

        print(f"Loaded trajectory from {checkpoint_path}")

    midrules = midrule_indicies or None
    compressed_tokens_key = spec.compressed_tokens_key or "num_embeddings"

    print_statistics_table(
        checkpoint_names,
        statistics_list,
        midrule_indicies=midrules,
        tablefmt=tablefmt_override or spec.tablefmt,
        short=spec.short,
        compressed_tokens_key=compressed_tokens_key,
    )

    if save_dir is not None:
        tex = format_statistics_table(
            checkpoint_names,
            statistics_list,
            midrule_indicies=midrules,
            tablefmt="latex",
            short=spec.short,
            compressed_tokens_key=compressed_tokens_key,
        )
        os.makedirs(save_dir, exist_ok=True)
        filename = (save_name or table_slug(spec.name)) + ".tex"
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "w") as f:
            f.write(tex)
            if not tex.endswith("\n"):
                f.write("\n")
        print(f"Saved {spec.name!r} to {save_path}")


DEFAULT_SAVE_DIR = "paper/tables"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", help="LaTeX label of the table to render (e.g. tab:low_dim_projection_results)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render every primary table (aliases skipped) in one process. Implies --save.",
    )
    parser.add_argument("--list", action="store_true", help="List available table names and exit.")
    parser.add_argument("--tablefmt", help="Override the spec's tablefmt for the stdout print (default: spec value).")
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Also write the rendered LaTeX table to {DEFAULT_SAVE_DIR}/<slug>.tex.",
    )
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help=f"Directory to write the .tex file into when --save is set (default: {DEFAULT_SAVE_DIR}).",
    )
    parser.add_argument(
        "--save-name",
        default=None,
        help="Override the saved filename stem (defaults to a slug of the table label).",
    )
    args = parser.parse_args()

    if args.list:
        for spec in TABLES:
            print(spec.name)
        for alias, target in _ALIASES.items():
            print(f"{alias}  (alias for {target})")
        return

    if args.all and args.name:
        parser.error("--all and --name are mutually exclusive")
    if not args.all and not args.name:
        parser.error("one of --name, --all, or --list is required")
    if args.all and args.save_name:
        parser.error("--save-name only applies to single-table renders (omit it with --all)")

    if args.all:
        failures: List[Tuple[str, str]] = []
        for spec in TABLES:
            print(f"\n=== Rendering {spec.name} ===")
            try:
                render_table(
                    spec,
                    tablefmt_override=args.tablefmt,
                    save_dir=args.save_dir,
                    save_name=None,
                )
            except (FileNotFoundError, ValueError) as e:
                # Keep going — a single broken spec shouldn't block the rest of
                # the batch when the user just wants whatever can be rendered.
                print(f"!! SKIPPED {spec.name}: {e}")
                failures.append((spec.name, str(e)))
        if failures:
            print(f"\n{len(failures)} table(s) skipped:")
            for name, msg in failures:
                print(f"  - {name}: {msg}")
            sys.exit(1)
        return

    if args.name not in TABLES_BY_NAME:
        available = "\n  ".join(sorted(TABLES_BY_NAME))
        sys.exit(f"Unknown table {args.name!r}. Available:\n  {available}")

    save_dir = args.save_dir if args.save else None
    # If the user requested an alias, save under the alias's slug (matches their intent).
    save_name = args.save_name or (table_slug(args.name) if args.save else None)
    render_table(
        TABLES_BY_NAME[args.name],
        tablefmt_override=args.tablefmt,
        save_dir=save_dir,
        save_name=save_name,
    )


if __name__ == "__main__":
    main()
