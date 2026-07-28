#!/usr/bin/env python3
"""Turn the keep-set into a tiered, reversible deletion plan.

Deletion is staged as a **move into a trash directory on the same filesystem**, not
an ``rm``: a rename is instantaneous regardless of directory size and is undone by
moving the entry back. Nothing is actually freed until the trash is removed, so the
validation pass in ``verify_paper.py`` runs against a tree that still has every byte
recoverable.

Tiers, in increasing order of risk::

    0  artifacts/cache/packed_lm_datasets   regenerable training-input cache; the
                                            paper never reads it
    1  dead-by-name roots and runs          *_invalid_*, *_old*, *_backup*, dated
                                            pre-cutoff dirs
    2  unreferenced runs                    run dirs in the per-run roots that no
                                            consumer in the keep-set names

Deliberately out of scope:

* Stripping ``.arrow`` payloads out of *kept* runs. It is the single largest saving
  available and it is not taken: it would permanently foreclose cache rebuilds
  (``CACHE_VERSION`` bumps, new metric columns, reviewer re-derivations) that the
  arrow data is the only source for.
* ``artifacts/experiments_in_progress`` -- excluded wholesale.

Usage::

    python scripts/cleanup/cleanup_plan.py --keepset artifacts_keepset.txt --report
    python scripts/cleanup/cleanup_plan.py --keepset ... --tier 0 --apply
    python scripts/cleanup/cleanup_plan.py --restore run/artifacts_cleanup/trash_<stamp>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

PER_RUN_ROOTS = ("experiments_progressive", "experiments", "experiments_prefix_tuning")

# Never considered, at any tier, for any reason.
EXCLUDE_ROOTS = {
    "experiments_in_progress",  # may have live MLS jobs writing; excluded by decision
    "cache",  # handled explicitly by tier 0, not by name matching
}

TIER0_PATHS = ["cache/packed_lm_datasets"]

# Whole roots that are dead by name.
TIER1_ROOTS = [
    "experiments_progressive_invalid_optimization",
    "experiments_before_27.11",
    "experiments_old",
    "hellaswag_evaluation_old",
    "hellaswag_evaluation_old_backup_25.03.26",
    "arc_evaluation_old",
    "arc_evaluation_invalid_ppl_computation",
    "hellaswag_evaluation_batch_64_no_reconstruction_accurecy",
    "hellaswag_evaluation_25.03.26_forward_intervention",
]

# Individual runs inside per-run roots whose name marks them superseded.
TIER1_RUN_MARKERS = ("_old_backup_", "_invalid_", "_deprecated_")

# A run touched this recently is treated as possibly live and is never a candidate,
# whatever the keep-set says. Cheap guard against deleting under a running job.
FRESH_DAYS = 7

# Cache files that a paper table renders from. A directory holding one of these is
# never deleted on the strength of its *name* alone, however dead that name looks.
#
# This guard exists because the name heuristic was caught red-handed: the only four
# surviving surprisal_steps_cache.json files -- the sole backing for
# tab:surprisal_steps_correlation -- live in the four
# sl_4096_*_old_backup_26.06.26 dirs that TIER1_RUN_MARKERS flags for deletion. The
# generator points at the non-backup paths, where the cache is absent, so the table is
# already non-regenerable and those backups are the last copy of that computation.
# Deleting them would have been silent and irreversible.
PAPER_CACHE_FILES = {
    "low_dimensional_cache.json",
    "surprisal_steps_cache.json",
    "full_cramming_table_cache.json",
    "prefix_tuning_table_cache.json",
    "greedy_accuracy_cache.json",
}


def paper_caches_in(path: Path) -> List[str]:
    """Names of paper cache files found anywhere under ``path``."""
    found: Set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda e: None):
        for fn in filenames:
            if fn in PAPER_CACHE_FILES:
                found.add(fn)
    return sorted(found)


# Suffixes that mark a directory as a snapshot of another run rather than a run of
# its own. Stripping one yields the "base" run the snapshot was taken from.
_SNAPSHOT_SUFFIX_RE = re.compile(r"_(?:old_backup|backup|old|archive)(?:_[0-9.]+)?$")


def base_run_of(rel: str) -> str | None:
    """``.../sl_4096_X_lr_0.1_old_backup_26.06.26`` -> ``.../sl_4096_X_lr_0.1``."""
    root, _, name = rel.rpartition("/")
    stripped = _SNAPSHOT_SUFFIX_RE.sub("", name)
    return f"{root}/{stripped}" if stripped != name else None


def last_copy_caches(rel: str, keep: Set[str]) -> List[str]:
    """Paper caches in ``rel`` that are the last copy for a run we are keeping.

    A cache only matters if some *kept* run needs it and does not have it. The
    surprisal case: the generator wants
    ``sl_4096_X_lr_0.1/progressive_prefixes/surprisal_steps_cache.json``; that path is
    empty, and the only copy sits in ``sl_4096_X_lr_0.1_old_backup_26.06.26``. Deleting
    the snapshot loses it for good.

    Caches inside a run that nothing references are not protected -- no generator will
    ever look for them.
    """
    base = base_run_of(rel)
    if base is None or base not in keep:
        return []
    have_in_base = set(paper_caches_in(ROOT / base)) if (ROOT / base).exists() else set()
    return [c for c in paper_caches_in(ROOT / rel) if c not in have_in_base]


def du_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda e: None):
        for fn in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, fn)).st_size
            except OSError:
                pass
    return total


def newest_mtime(path: Path) -> float:
    newest = 0.0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda e: None):
        for fn in filenames:
            try:
                newest = max(newest, os.lstat(os.path.join(dirpath, fn)).st_mtime)
            except OSError:
                pass
    return newest


def load_keepset(path: str) -> Set[str]:
    return {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()}


def build_candidates(
    keep: Set[str], tiers: List[int], allow_cache_drop: bool = False
) -> Tuple[Dict[int, List[str]], List[str], Dict[str, List[str]]]:
    """Return {tier: [repo-relative paths]}, skipped-as-fresh paths, and blocked
    paths mapped to the paper caches they hold."""
    cands: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    fresh_skipped: List[str] = []
    cache_blocked: Dict[str, List[str]] = {}
    cutoff = time.time() - FRESH_DAYS * 86400

    def consider(tier: int, path: Path) -> None:
        if not path.exists():
            return
        rel = str(path.relative_to(ROOT))
        if newest_mtime(path) > cutoff:
            fresh_skipped.append(rel)
            return
        if not allow_cache_drop:
            caches = last_copy_caches(rel, keep)
            if caches:
                cache_blocked[rel] = caches
                return
        cands[tier].append(rel)

    if 0 in tiers:
        for rel in TIER0_PATHS:
            consider(0, ARTIFACTS / rel)

    if 1 in tiers:
        for name in TIER1_ROOTS:
            consider(1, ARTIFACTS / name)
        for root in PER_RUN_ROOTS:
            base = ARTIFACTS / root
            if not base.is_dir():
                continue
            for run in sorted(base.iterdir()):
                if run.is_dir() and any(m in run.name for m in TIER1_RUN_MARKERS):
                    if f"artifacts/{root}/{run.name}" not in keep:
                        consider(1, run)

    if 2 in tiers:
        tier1_set = set(cands[1])
        for root in PER_RUN_ROOTS:
            base = ARTIFACTS / root
            if not base.is_dir() or root in EXCLUDE_ROOTS:
                continue
            for run in sorted(base.iterdir()):
                if not run.is_dir():
                    continue
                rel = f"artifacts/{root}/{run.name}"
                if rel in keep or rel in tier1_set:
                    continue
                consider(2, run)

    return cands, fresh_skipped, cache_blocked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keepset", help="keep-set file from keepset.py")
    ap.add_argument("--tier", type=int, action="append", help="tier(s) to include; repeatable. Default: all of 0,1,2")
    ap.add_argument("--report", action="store_true", help="print the plan with sizes")
    ap.add_argument("--manifest", help="write the plan as JSON here")
    ap.add_argument("--apply", action="store_true", help="move candidates into a trash dir")
    ap.add_argument("--trash", help="trash dir (default run/artifacts_cleanup/trash_<stamp>)")
    ap.add_argument("--restore", help="move everything in this trash dir back and exit")
    ap.add_argument(
        "--allow-cache-drop",
        action="store_true",
        help="delete candidates even if they hold a paper cache file (see PAPER_CACHE_FILES)",
    )
    args = ap.parse_args()

    if args.restore:
        trash = Path(args.restore)
        manifest = json.loads((trash / "manifest.json").read_text())
        for rel in manifest["paths"]:
            src, dst = trash / rel, ROOT / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                print(f"restored {rel}")
        print(f"\nrestored {len(manifest['paths'])} paths from {trash}")
        return 0

    if not args.keepset:
        ap.error("--keepset is required unless --restore is used")

    keep = load_keepset(args.keepset)
    tiers = sorted(set(args.tier)) if args.tier else [0, 1, 2]
    cands, fresh_skipped, cache_blocked = build_candidates(keep, tiers, args.allow_cache_drop)

    sizes = {rel: du_bytes(ROOT / rel) for tier in tiers for rel in cands[tier]}
    grand = 0

    if args.report:
        for tier in tiers:
            entries = cands[tier]
            sub = sum(sizes[r] for r in entries)
            grand += sub
            print(f"\n=== tier {tier}: {len(entries)} paths, {sub / 1e9:.1f} GB ===")
            for rel in sorted(entries, key=lambda r: -sizes[r])[:15]:
                print(f"  {sizes[rel] / 1e9:8.2f} GB  {rel}")
            if len(entries) > 15:
                rest = sum(sizes[r] for r in sorted(entries, key=lambda r: -sizes[r])[15:])
                print(f"  {rest / 1e9:8.2f} GB  ... and {len(entries) - 15} more")
        print(f"\nTOTAL RECLAIMABLE: {grand / 1e9:.1f} GB")
        print(f"keep-set: {len(keep)} run dirs (never touched)")
        if fresh_skipped:
            print(f"\nskipped as modified within {FRESH_DAYS}d (possibly live): {len(fresh_skipped)}")
            for rel in fresh_skipped[:10]:
                print(f"  ~ {rel}")
        if cache_blocked:
            blocked_gb = sum(du_bytes(ROOT / rel) for rel in cache_blocked) / 1e9
            print(f"\nBLOCKED -- holds a paper cache file ({len(cache_blocked)} paths, {blocked_gb:.1f} GB):")
            for rel, caches in sorted(cache_blocked.items())[:15]:
                print(f"  x {rel}\n      {', '.join(caches)}")
            if len(cache_blocked) > 15:
                print(f"  ... and {len(cache_blocked) - 15} more")
            print("  These are excluded regardless of tier. Review each before using --allow-cache-drop.")

    plan = {
        "tiers": tiers,
        "keepset_size": len(keep),
        "paths": [rel for tier in tiers for rel in cands[tier]],
        "sizes_bytes": sizes,
        "fresh_skipped": fresh_skipped,
        "cache_blocked": cache_blocked,
        "total_bytes": sum(sizes.values()),
    }
    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(plan, indent=2))
        print(f"\nmanifest -> {args.manifest}")

    if args.apply:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        trash = Path(args.trash) if args.trash else ROOT / "run" / "artifacts_cleanup" / f"trash_{stamp}"
        trash.mkdir(parents=True, exist_ok=True)
        (trash / "manifest.json").write_text(json.dumps(plan, indent=2))
        moved = 0
        for rel in plan["paths"]:
            src, dst = ROOT / rel, trash / rel
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved += 1
            print(f"moved {rel}")
        print(f"\nmoved {moved} paths into {trash}")
        print(f"restore:  python scripts/cleanup/cleanup_plan.py --restore {trash}")
        print("free it:  rm -rf <trash>   # only after verify_paper.py passes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
