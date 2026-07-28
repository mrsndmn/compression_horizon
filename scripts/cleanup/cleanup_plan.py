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


def build_candidates(keep: Set[str], tiers: List[int]) -> Tuple[Dict[int, List[str]], List[str]]:
    """Return {tier: [repo-relative paths]} plus the list of skipped-as-fresh paths."""
    cands: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    fresh_skipped: List[str] = []
    cutoff = time.time() - FRESH_DAYS * 86400

    def consider(tier: int, path: Path) -> None:
        if not path.exists():
            return
        rel = str(path.relative_to(ROOT))
        if newest_mtime(path) > cutoff:
            fresh_skipped.append(rel)
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

    return cands, fresh_skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keepset", help="keep-set file from keepset.py")
    ap.add_argument("--tier", type=int, action="append", help="tier(s) to include; repeatable. Default: all of 0,1,2")
    ap.add_argument("--report", action="store_true", help="print the plan with sizes")
    ap.add_argument("--manifest", help="write the plan as JSON here")
    ap.add_argument("--apply", action="store_true", help="move candidates into a trash dir")
    ap.add_argument("--trash", help="trash dir (default run/artifacts_cleanup/trash_<stamp>)")
    ap.add_argument("--restore", help="move everything in this trash dir back and exit")
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
    cands, fresh_skipped = build_candidates(keep, tiers)

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

    plan = {
        "tiers": tiers,
        "keepset_size": len(keep),
        "paths": [rel for tier in tiers for rel in cands[tier]],
        "sizes_bytes": sizes,
        "fresh_skipped": fresh_skipped,
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
