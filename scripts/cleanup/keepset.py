#!/usr/bin/env python3
"""Derive the set of artifact run directories the paper, poster, blog post and
public release still depend on.

The keep-set is expressed at **run-directory granularity** (``artifacts/<root>/<run>``),
never per-file: a run is either kept whole or removed whole. Stripping ``.arrow``
payloads out of kept runs is deliberately not supported -- it would foreclose any
cache rebuild (``CACHE_VERSION`` bumps, new metric columns, reviewer re-derivations)
for a saving that only exists once.

Two independent derivations feed the final keep-set, and both are needed:

``static``
    Reads each generator's *own* constants (importing the module, never copying
    its path logic) plus a regex sweep over prose sources. Catches run dirs that
    a generator only touches under ``--compute`` -- e.g. ``solution_diversity``
    and ``trajectory_clusters`` render from ``artifacts/paper/`` caches and would
    otherwise look unused.

``trace``
    ``scripts/cleanup/trace_regen.py`` records every ``artifacts/`` path opened
    during a full regeneration. Catches paths no static rule predicted -- e.g.
    the hash-keyed ``artifacts/experiments/*<id>/`` globs in ``full_cramming_table``.

Usage::

    python scripts/cleanup/keepset.py --report
    python scripts/cleanup/keepset.py --out artifacts_keepset.txt
    python scripts/cleanup/keepset.py --trace run/artifacts_cleanup/trace.txt --report
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import io
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]

# Roots analysed per-run. Everything else under artifacts/ is governed by
# WHOLE_ROOT_KEEP / the tier policy in cleanup_plan.py.
PER_RUN_ROOTS = (
    "experiments_progressive",
    "experiments",
    "experiments_prefix_tuning",
)

# Small roots (~2.5 GB combined) kept in their entirety. Per-run analysis here
# would buy almost nothing and every one of them backs a rendered table or figure.
WHOLE_ROOT_KEEP = (
    "paper",
    "analysis",
    "results",
    "visualizations",
    "arc_evaluation",
    "hellaswag_evaluation",
    "mmlu_evaluation",
    "interpolations",
    "tda_analysis",
    "pg19_paraphrases",
    "hellaswag_paraphrases",
    "hellaswag_paraphrases_temps",
    "arc_easy_paraphrases",
    "arc_easy_paraphrases_temps",
    "arc_challenge_paraphrases",
    "arc_challenge_paraphrases_temps",
    "arc_paraphrases",
    "arc_eval_paraphrase_temps",
)

# Prose / config sources swept for literal artifact paths. Relative to ROOT unless
# absolute. The sibling public repo and the blog post are included per the decision
# to fold the public release and habr post into the keep-set.
TEXT_SOURCES = (
    "paper/example_paper.tex",
    "paper/appendix.tex",
    "poster/poster.html",
    "habr",
    "scripts",
    "experiments",
    "../progressive_cramming/scripts",
    "../progressive_cramming/presentation",
)

TEXT_SUFFIXES = {".tex", ".html", ".md", ".py", ".sh", ".js", ".json", ".yaml", ".yml"}

_PATH_RE = re.compile(r"artifacts/[A-Za-z0-9_.=+-]+(?:/[A-Za-z0-9_.=+*-]+)*")

# ``_EXP = "artifacts/experiments_progressive"`` and friends. Generators build almost
# every checkpoint path as f"{_EXP}/sl_4096_...", which a literal search cannot see.
_ROOT_VAR_RE = re.compile(r"""^\s*(_?[A-Z][A-Z0-9_]*)\s*(?::\s*[^=]+)?=\s*["'](artifacts/[^"']*)["']""", re.M)


def _expand_path_vars(text: str) -> str:
    """Substitute ``{VAR}`` with the literal artifact root VAR is assigned in this file.

    Without this, a checkpoint list written as f"{_EXP}/sl_4096_foo" is invisible to
    the regex sweep. That matters most for **commented-out** specs: tab:width_dualmodel_ablation
    is commented out in progressive.py, yet width_dualmodel_ablation.tex is still
    \\input by paper/appendix.tex -- its runs are load-bearing for a table in the paper
    even though no live code names them.
    """
    for var, value in _ROOT_VAR_RE.findall(text):
        text = text.replace("{" + var + "}", value)
    return text


def _load(rel: str):
    """Import a generator module by path, muting its import-time output."""
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(f"_keepset_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    argv, cwd = sys.argv, os.getcwd()
    os.chdir(ROOT)
    sys.argv = [str(path)]
    try:
        with redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.argv, _ = argv, None
        os.chdir(cwd)
    return mod


def _to_run_dir(path: str) -> str | None:
    """Normalise any artifact path to ``artifacts/<root>/<run>``.

    Returns None for paths outside the per-run roots (those are covered by
    WHOLE_ROOT_KEEP) or too shallow to name a run.
    """
    path = str(path).replace(os.sep, "/")
    idx = path.find("artifacts/")
    if idx < 0:
        return None
    parts = [p for p in path[idx:].split("/") if p]
    if len(parts) < 3 or parts[1] not in PER_RUN_ROOTS:
        return None
    return "/".join(parts[:3])


def _expand(entries: Iterable[str], allow_globs: bool = True) -> Set[str]:
    """Resolve entries against the real tree and reduce to run dirs.

    ``allow_globs=False`` accepts literal paths only. Prose and tooling docstrings
    are full of generic sweeps (``artifacts/experiments_progressive/*/progressive_prefixes``,
    ``artifacts/experiments/*``) that expand to the entire tree and would silently
    make the keep-set everything -- a wildcard in a help string is not evidence that
    a *particular* run is load-bearing. Generators whose registries legitimately
    contain globs (TableSpec.checkpoints) keep the default.
    """
    out: Set[str] = set()
    for entry in entries:
        if not isinstance(entry, str) or "artifacts/" not in entry:
            continue
        is_glob = any(c in entry for c in "*?[")
        if is_glob and not allow_globs:
            continue
        abs_entry = entry if os.path.isabs(entry) else str(ROOT / entry)
        hits = glob.glob(abs_entry) if is_glob else [abs_entry]
        for hit in hits or []:
            if not os.path.exists(hit):
                continue
            run = _to_run_dir(hit)
            if run:
                out.add(run)
    return out


# --------------------------------------------------------------------------
# Static sources. Each returns the run dirs one consumer depends on.
# --------------------------------------------------------------------------


def src_progressive_tables() -> Set[str]:
    """The 15 declarative TableSpecs in scripts/paper/tables/progressive.py."""
    mod = _load("scripts/paper/tables/progressive.py")
    entries: List[str] = []
    for table in mod.TABLES:
        entries.extend(c for c in table.checkpoints if isinstance(c, str))
    return _expand(entries)


def src_solution_diversity() -> Set[str]:
    """LR-sweep runs behind tab:solution_diversity (cache-only at render time)."""
    mod = _load("scripts/paper/tables/solution_diversity.py")
    entries: List[str] = []
    for _name, _key, run_dirs in mod.FAMILIES:
        entries.extend(run_dirs)
    return _expand(entries)


def src_trajectory_clusters() -> Set[str]:
    """50-sample runs behind tab:trajectory_cluster_structure (cache-only at render time)."""
    mod = _load("scripts/paper/tables/trajectory_clusters.py")
    names = [n for _label, n in list(mod.SCALE_ROWS) + list(mod.LR_ROWS)]
    return _expand(f"artifacts/experiments_progressive/{n}" for n in names)


def src_surprisal_steps() -> Set[str]:
    mod = _load("scripts/paper/tables/surprisal_steps_correlation.py")
    return _expand(ckpt for ckpt, _label in mod.CHECKPOINTS)


def src_margin_greedy() -> Set[str]:
    mod = _load("scripts/paper/tables/progressive_margin_greedy_table.py")
    entries = []
    for _label, short, low_dim, lr in mod.MODELS:
        for margin in mod.MARGINS:
            base = mod.run_dir_name(short, lr, low_dim)
            entries.append(f"artifacts/experiments_progressive/{base}_cm_{margin}")
            entries.append(f"artifacts/experiments_progressive/{base}")
    return _expand(entries)


def src_temperature_logit_stats() -> Set[str]:
    mod = _load("scripts/paper/tables/temperature_logit_stats.py")
    entries = []
    for short, lr, _ckpt, _prefix in mod.MODELS:
        for temp in mod.TEMPS:
            base = f"artifacts/experiments_progressive/sl_4096_{short}_ds_pg19_1k_limit_50_lr_{lr}_temp_{temp}"
            entries += [base, f"{base}_comp_t2"]
            if (short, temp) in mod.HIGHCAP:
                entries.append(f"{base}{mod.HIGHCAP_SUFFIX}")
    return _expand(entries)


def src_depth_size_pivot() -> Set[str]:
    mod = _load("scripts/paper/tables/depth_size_pivot.py")
    entries = []
    for model in mod.MODELS:
        for n in model.first_ns:
            entries.append(mod._first_dir(model.ckpt_base, n))
        if model.has_full:
            entries.append(mod._full_dir(model.ckpt_base))
    return _expand(entries)


_FULLCRAM_RE = re.compile(r'\{\s*"train"\s*:\s*"(full|progr|prefix)"\s*,\s*"id"\s*:\s*"([^"]+)"')

_FULLCRAM_ROOT = {
    "full": "artifacts/experiments/*{id}",
    "progr": "artifacts/experiments_progressive/{id}",
    "prefix": "artifacts/experiments_prefix_tuning/{id}",
}


def src_full_cramming() -> Set[str]:
    """Runs behind tab:full_vs_progressive{,_appendix} and tab:prefix_tuning_accuracy.

    ``full_cramming_table.py`` holds its row lists as inline ``{"train":..., "id":...}``
    dicts inside the arg-parsing branch, keyed by ``--type`` -- there is no module-level
    constant to import. ``full`` ids are 8-char content hashes resolved by globbing
    ``artifacts/experiments/*<id>``, which no literal path search can predict.

    Commented-out rows are matched deliberately: a row disabled in the generator may
    still back a ``.tex`` that is committed and ``\\input`` by the paper, and the whole
    ``artifacts/experiments`` root is only ~1.6 GB, so erring toward keeping is nearly free.
    """
    text = (ROOT / "scripts/paper/tables/full_cramming_table.py").read_text()
    entries = [_FULLCRAM_ROOT[train].format(id=rid) for train, rid in _FULLCRAM_RE.findall(text)]
    return _expand(entries)


def src_margin_table() -> Set[str]:
    """Convergence-margin runs behind tab:progressive_margin.

    Its generator is *not* invoked by scripts/paper/tables/tables.sh, so `make tables`
    never rebuilds progressive_margin.tex and no trace of a full regeneration will
    reveal these runs -- they must be named statically or the cm_/lm_ runs look dead.
    """
    mod = _load("scripts/paper/tables/progressive_margin_table.py")
    return _expand(f"artifacts/experiments_progressive/{b}" for _model, _variant, b in mod.EXPERIMENTS)


def src_hf_uploader() -> Set[str]:
    """The 8 runs published as mrsndmn/progressive_cramming_trajectories."""
    mod = _load("scripts/hf/upload_trajectories.py")
    entries = []
    for _name, splits in mod.CONFIGS:
        entries.extend(splits.values())
    return _expand(entries)


def src_text_refs() -> Set[str]:
    """Literal artifact paths in paper/poster/habr/scripts and the public repo."""
    entries: List[str] = []
    for rel in TEXT_SOURCES:
        base = (ROOT / rel).resolve()
        if not base.exists():
            continue
        files = (
            [base]
            if base.is_file()
            else [p for p in base.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES and "__pycache__" not in p.parts]
        )
        for path in files:
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            entries.extend(_PATH_RE.findall(_expand_path_vars(text)))
    return _expand(entries, allow_globs=False)


STATIC_SOURCES: Dict[str, Callable[[], Set[str]]] = {
    "progressive_tables": src_progressive_tables,
    "solution_diversity": src_solution_diversity,
    "trajectory_clusters": src_trajectory_clusters,
    "surprisal_steps": src_surprisal_steps,
    "margin_greedy": src_margin_greedy,
    "temperature_logit_stats": src_temperature_logit_stats,
    "depth_size_pivot": src_depth_size_pivot,
    "margin_table": src_margin_table,
    "full_cramming": src_full_cramming,
    "hf_uploader": src_hf_uploader,
    "text_refs": src_text_refs,
}


def load_trace(trace_file: str) -> Set[str]:
    """Run dirs observed by scripts/cleanup/trace_regen.py."""
    out: Set[str] = set()
    for line in Path(trace_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        run = _to_run_dir(line)
        if run and (ROOT / run).exists():
            out.add(run)
    return out


def build(trace_file: str | None = None) -> Tuple[Set[str], Dict[str, Set[str]]]:
    provenance: Dict[str, Set[str]] = {}
    for name, fn in STATIC_SOURCES.items():
        provenance[name] = fn()
    if trace_file:
        provenance["trace"] = load_trace(trace_file)
    keep: Set[str] = set()
    for runs in provenance.values():
        keep |= runs
    return keep, provenance


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", help="trace file from trace_regen.py to union in (and cross-check against)")
    ap.add_argument("--out", help="write the keep-set (one run dir per line) here")
    ap.add_argument("--report", action="store_true", help="print per-source counts and trace/static deltas")
    args = ap.parse_args()

    keep, provenance = build(args.trace)

    if args.report:
        print(f"{'source':<24} {'runs':>6}")
        print("-" * 32)
        for name in list(STATIC_SOURCES) + (["trace"] if args.trace else []):
            print(f"{name:<24} {len(provenance.get(name, set())):>6}")
        print("-" * 32)
        print(f"{'UNION (keep-set)':<24} {len(keep):>6}\n")

        static_union: Set[str] = set()
        for name in STATIC_SOURCES:
            static_union |= provenance[name]
        if args.trace:
            only_trace = provenance["trace"] - static_union
            print(f"opened during regeneration but NOT predicted statically: {len(only_trace)}")
            for run in sorted(only_trace):
                print(f"  + {run}")
            print("  (each one is a gap in a static rule -- fix the rule, do not just union it)\n")

        print("per-root keep vs. total on disk:")
        for root in PER_RUN_ROOTS:
            base = ROOT / "artifacts" / root
            if not base.is_dir():
                continue
            total = sum(1 for p in base.iterdir() if p.is_dir())
            kept = sum(1 for r in keep if r.startswith(f"artifacts/{root}/"))
            print(f"  artifacts/{root:<32} keep {kept:>4} / {total:>4}")
        print("\nwhole-root keeps (not per-run analysed):")
        print("  " + ", ".join(WHOLE_ROOT_KEEP))

    if args.out:
        Path(args.out).write_text("\n".join(sorted(keep)) + "\n")
        print(f"\nwrote {len(keep)} run dirs -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
