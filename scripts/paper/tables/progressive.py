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
    TableSpec(
        # ``tables.sh`` listed 5 names for 4 checkpoints — the lr_0.5 entry was
        # missing from its --checkpoints flag. Restored here so the sweep is
        # complete and the names_mapping length matches.
        name="tab:lr_sweep_llama_3.1_8b",
        checkpoints=[
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.01/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_0.5/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_1.0/progressive_prefixes",
            f"{_EXP}/sl_4096_Meta-Llama-3.1-8B_lr_5.0/progressive_prefixes",
        ],
        names_mapping="0.01,0.1,0.5,1.0,5.0",
    ),
    TableSpec(
        name="tab:full_llama_3.1_8b",
        checkpoints=[f"{_EXP}/sl_4096_Meta-Llama-3.1*/progressive_prefixes"],
    ),
    TableSpec(
        name="tab:full_pythia_1.4b",
        checkpoints=[f"{_EXP}/sl_4096_pythia-1.4b*/progressive_prefixes"],
    ),
    TableSpec(
        name="tab:full_smollm2_1.7b",
        checkpoints=[f"{_EXP}/sl_4096_SmolLM2-1.7B*/progressive_prefixes"],
    ),
    TableSpec(
        name="tab:full_qwen3_4b",
        checkpoints=[f"{_EXP}/sl_4096_Qwen3-4B*/progressive_prefixes"],
    ),
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

    print_statistics_table(
        checkpoint_names,
        statistics_list,
        midrule_indicies=midrules,
        tablefmt=tablefmt_override or spec.tablefmt,
        short=spec.short,
    )

    if save_dir is not None:
        tex = format_statistics_table(
            checkpoint_names,
            statistics_list,
            midrule_indicies=midrules,
            tablefmt="latex",
            short=spec.short,
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
        for spec in TABLES:
            print(f"\n=== Rendering {spec.name} ===")
            render_table(
                spec,
                tablefmt_override=args.tablefmt,
                save_dir=args.save_dir,
                save_name=None,
            )
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
