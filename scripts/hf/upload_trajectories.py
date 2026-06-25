#!/usr/bin/env python
"""Upload progressive-cramming trajectories to the HF Hub.

Publishes the per-stage trajectory datasets behind ``tab:progressive_modifications``
(see ``scripts/paper/tables/progressive.py``) as a single Hub dataset with one
*config* per model family and two *splits* per config:

    config (model)        split=baseline                          split=lowdim
    --------------------  --------------------------------------  -------------------------------------
    Llama-3.1-8B          .../lr_0.1                               .../lowdim_256_lowproj_lr_0.1
    pythia-1.4b           .../lr_0.5                               .../lowdim_256_lowproj_lr_0.5
    SmolLM2-1.7B          .../lr_0.1                               .../lowdim_256_lowproj_lr_0.1
    gemma-3-4b-pt         .../lr_0.1                               .../lowdim_32_lowproj_lr_0.1

Each row is one converged progressive-cramming stage (one trajectory point) for
one document; order a sample's trajectory by ``(sample_id, stage_index)``.

The HF token is read from the ``HF_TOKEN`` environment variable; it is never
written to disk or committed.
"""
import os
import sys

from datasets import load_from_disk
from huggingface_hub import HfApi

REPO_ID = "mrsndmn/progressive_cramming_trajectories"
EXP = "artifacts/experiments_progressive"

# Smallest-first so the end-to-end flow is validated cheaply before the big push.
CONFIGS = [
    (
        "gemma-3-4b-pt",
        {
            "baseline": f"{EXP}/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            "lowdim": f"{EXP}/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lowdim_32_lowproj_lr_0.1/progressive_prefixes",
        },
    ),
    (
        "pythia-1.4b",
        {
            "baseline": f"{EXP}/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5/progressive_prefixes",
            "lowdim": f"{EXP}/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.5/progressive_prefixes",
        },
    ),
    (
        "SmolLM2-1.7B",
        {
            "baseline": f"{EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            "lowdim": f"{EXP}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1/progressive_prefixes",
        },
    ),
    (
        "Llama-3.1-8B",
        {
            "baseline": f"{EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1/progressive_prefixes",
            "lowdim": f"{EXP}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1/progressive_prefixes",
        },
    ),
]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is not set", file=sys.stderr)
        return 1

    HfApi(token=token).create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)

    for config_name, splits in CONFIGS:
        for split, path in splits.items():
            ds = load_from_disk(path)
            print(f">>> push {config_name}/{split}  rows={ds.num_rows}  from {path}", flush=True)
            ds.push_to_hub(
                REPO_ID,
                config_name=config_name,
                split=split,
                token=token,
                max_shard_size="500MB",
            )
            print(f"<<< done {config_name}/{split}", flush=True)

    print("ALL UPLOADS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
