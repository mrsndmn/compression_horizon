#!/usr/bin/env python3
"""Record every ``artifacts/`` path opened while regenerating the paper's tables
and figures.

This is the dynamic half of the keep-set derivation (see ``keepset.py``). It runs
each generator in a subprocess whose interpreter carries a ``sys.addaudithook``
that logs ``open`` events, then unions the per-generator logs.

The audit hook sees every open regardless of which library performs it (``datasets``,
``pyarrow``, ``json``, ``PIL``), which plain static analysis cannot promise.

Note on interpretation: with the ``low_dimensional_cache.json`` files present, the
table generators short-circuit and never touch the ``.arrow`` payloads -- so the
trace names *cache files*, not datasets. That is why the keep-set is defined at
run-directory granularity: an opened cache keeps its whole run dir, arrow included.

Usage::

    python scripts/cleanup/trace_regen.py --out run/artifacts_cleanup/trace.txt
    python scripts/cleanup/trace_regen.py --out ... --only tables
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]

# Injected into each traced subprocess via -c. Writes on interpreter exit so a
# generator that raises still yields the paths it managed to open.
_HOOK = r"""
import atexit, os, sys
_log = os.environ["_TRACE_LOG"]
_seen = set()
def _hook(event, args):
    if event == "open":
        p = args[0]
        if isinstance(p, bytes):
            try: p = p.decode()
            except Exception: return
        if isinstance(p, str) and "artifacts/" in p:
            _seen.add(os.path.abspath(p))
sys.addaudithook(_hook)
@atexit.register
def _dump():
    with open(_log, "a") as f:
        f.write("\n".join(sorted(_seen)) + "\n")
"""


def table_names() -> List[str]:
    """Every TableSpec label, read from the registry rather than copied.

    ``scripts/paper/tables/tables.sh`` hardcodes a list that has drifted -- it still
    names ``tab:lr_sweep_llama_3.1_8b``, ``tab:full_pythia_1.4b`` and others that no
    longer exist in ``TABLES`` (the shell loop has no ``set -e``, so those renders
    fail silently). Reading the registry keeps the trace honest and complete.
    """
    import importlib.util

    path = ROOT / "scripts" / "paper" / "tables" / "progressive.py"
    spec = importlib.util.spec_from_file_location("_trace_progressive", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    argv, cwd = sys.argv, os.getcwd()
    os.chdir(ROOT)
    sys.argv = [str(path)]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
        os.chdir(cwd)
    return [t.name for t in mod.TABLES]


def jobs(only: str | None, scratch: str) -> List[Tuple[str, str, List[str]]]:
    """Generators that must write somewhere are pointed at a scratch dir, never at
    paper/tables/ -- tracing must not mutate committed .tex files."""
    out: List[Tuple[str, str, List[str]]] = []

    for name in table_names():
        out.append(("tables", name, ["scripts/paper/tables/progressive.py", "--name", name]))

    for script, label in [
        ("depth_size_pivot.py", "tab:depth_size_pivot"),
        ("surprisal_steps_correlation.py", "tab:surprisal_steps_correlation"),
        ("solution_diversity.py", "tab:solution_diversity"),
        ("temperature_logit_stats.py", "tab:temperature_logit_stats"),
        ("trajectory_clusters.py", "tab:trajectory_cluster_structure"),
    ]:
        out.append(("tables", label, [f"scripts/paper/tables/{script}"]))

    for t in ["full_cramming", "full_cramming_apendix", "prefix_tuning"]:
        out.append(
            (
                "tables",
                f"full_cramming:{t}",
                ["scripts/paper/tables/full_cramming_table.py", "--type", t, "--save-dir", scratch],
            )
        )

    out.append(
        (
            "tables",
            "tab:progressive_margin_greedy",
            ["scripts/paper/tables/progressive_margin_greedy_table.py", "--save-dir", scratch],
        )
    )
    # Not in tables.sh -- progressive_margin.tex is in the paper but `make tables`
    # never rebuilds it. Traced here so its runs are not mistaken for dead.
    out.append(
        (
            "tables",
            "tab:progressive_margin",
            ["scripts/paper/tables/progressive_margin_table.py", "--save-dir", scratch],
        )
    )
    out.append(
        (
            "tables",
            "tab:semantic_evaluation",
            [
                "scripts/paper/tables/bench_semantic_results.py",
                "--tablefmt",
                "latex",
                "--save-dir",
                scratch,
                "--arc-split",
                "ARC-Easy",
            ],
        )
    )

    if only:
        out = [j for j in out if j[0] == only or j[1] == only]
    return out


def run(argv: List[str], log: Path, env: dict) -> Tuple[int, str]:
    """Run one generator under the audit hook. --save is deliberately omitted so
    tracing never overwrites a committed .tex."""
    code = _HOOK + f"\nimport runpy, sys\nsys.argv = {argv!r}\nrunpy.run_path({argv[0]!r}, run_name='__main__')\n"
    env = {**env, "_TRACE_LOG": str(log)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    return proc.returncode, (proc.stderr or "")[-400:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="union of all observed artifact paths")
    ap.add_argument("--only", help="restrict to one group ('tables') or one label")
    ap.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help="continue after a generator fails (default: on; failures are reported)",
    )
    args = ap.parse_args()

    env = {**os.environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}"}
    scratch = tempfile.mkdtemp(prefix="trace_regen_tex_")
    todo = jobs(args.only, scratch)
    seen: Set[str] = set()
    failures: List[Tuple[str, str]] = []

    for i, (_group, label, argv) in enumerate(todo, 1):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            log = Path(tmp.name)
        print(f"[{i}/{len(todo)}] {label} ... ", end="", flush=True)
        try:
            rc, err = run(argv, log, env)
        except subprocess.TimeoutExpired:
            rc, err = -1, "timeout"
        found = {ln.strip() for ln in log.read_text().splitlines() if ln.strip()} if log.exists() else set()
        log.unlink(missing_ok=True)
        seen |= found
        print(f"rc={rc}  paths={len(found)}")
        if rc != 0:
            failures.append((label, err))
            if not args.keep_going:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sorted(seen)) + "\n")
    print(f"\n{len(seen)} distinct artifact paths -> {out}")

    if failures:
        print(f"\n{len(failures)} generator(s) failed -- their reads are MISSING from this trace.")
        print("Do not delete anything until these are resolved or explicitly accounted for:")
        for label, err in failures:
            print(f"  ! {label}: {err.strip().splitlines()[-1] if err.strip() else 'see stderr'}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
