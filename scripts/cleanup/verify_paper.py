#!/usr/bin/env python3
"""Prove that an artifact deletion changed nothing the paper or poster renders.

Protocol::

    # 1. before touching anything
    python scripts/cleanup/verify_paper.py --snapshot run/artifacts_cleanup/baseline.json

    # 2. stage a tier into the trash
    python scripts/cleanup/cleanup_plan.py --keepset artifacts_keepset.txt --tier 0 --apply

    # 3. regenerate and compare
    make tables
    python scripts/cleanup/verify_paper.py --compare run/artifacts_cleanup/baseline.json

A byte-identical set of rendered tables and figures means the tier was safe. Any
difference -- including a table that now renders ``nan`` or a generator that dies --
means the deleted data was load-bearing: restore the trash and narrow the tier.

``--compare`` intentionally does not run ``make tables`` itself. Regeneration must be
the caller's explicit step so a failed regeneration is visible as a failed command
rather than being swallowed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]

# Rendered outputs whose content must be invariant under artifact deletion.
WATCHED = [
    ("tables", "paper/tables", ("*.tex",)),
    ("figures", "paper/figures", ("*.pdf", "*.png", "*.tikz")),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for group, rel, patterns in WATCHED:
        base = ROOT / rel
        files: Dict[str, str] = {}
        if base.is_dir():
            for pat in patterns:
                for path in sorted(base.rglob(pat)):
                    if "__pycache__" in path.parts:
                        continue
                    files[str(path.relative_to(ROOT))] = sha256(path)
        out[group] = files
    return out


def nan_scan() -> Dict[str, int]:
    """Count 'nan' occurrences per rendered table -- lint_paper.py fails on these,
    and a stale/absent cache is the usual cause."""
    hits: Dict[str, int] = {}
    base = ROOT / "paper" / "tables"
    for path in sorted(base.rglob("*.tex")):
        n = path.read_text(errors="ignore").lower().count("nan")
        if n:
            hits[str(path.relative_to(ROOT))] = n
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot", help="write a baseline manifest here")
    g.add_argument("--compare", help="compare current state against this baseline")
    args = ap.parse_args()

    state = {"files": collect(), "nan": nan_scan()}

    if args.snapshot:
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
        Path(args.snapshot).write_text(json.dumps(state, indent=2))
        n = sum(len(v) for v in state["files"].values())
        print(f"baseline: {n} rendered files -> {args.snapshot}")
        if state["nan"]:
            print(f"note: {len(state['nan'])} table(s) already contain 'nan' before any deletion:")
            for path, count in state["nan"].items():
                print(f"  {path}: {count}")
        return 0

    base = json.loads(Path(args.compare).read_text())
    failed = False

    for group in sorted(state["files"]):
        before, after = base["files"].get(group, {}), state["files"][group]
        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
        status = "OK" if not (missing or changed) else "FAIL"
        if missing or changed:
            failed = True
        print(f"[{status}] {group}: {len(after)} files, " f"{len(changed)} changed, {len(missing)} missing, {len(added)} new")
        for path in changed:
            print(f"    ~ {path}")
        for path in missing:
            print(f"    - {path}")
        for path in added:
            print(f"    + {path}")

    new_nan = {k: v for k, v in state["nan"].items() if k not in base.get("nan", {})}
    if new_nan:
        failed = True
        print("\n[FAIL] tables that gained 'nan' (lint_paper.py will reject these):")
        for path, count in new_nan.items():
            print(f"    {path}: {count}")

    diff = subprocess.run(
        ["git", "diff", "--stat", "--", "paper/tables", "paper/figures"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    print(f"\ngit diff (paper/tables, paper/figures):\n{diff or '  (clean)'}")

    print(
        "\nRESULT:", "MISMATCH -- restore the trash and narrow the tier" if failed else "identical -- this tier is safe to free"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
