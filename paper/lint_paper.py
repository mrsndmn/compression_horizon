#!/usr/bin/env python3
"""Run all automated paper lint checks.

Each check is a callable returning a list of human-readable error strings
(empty list = pass). Add new checks by appending to ``CHECKS``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

PAPER_DIR = Path(__file__).resolve().parent
ATTACHMENT_DIRS = ("figures", "styles")

# Body sources that make up the compiled paper. Raw tabular content is
# prohibited here: every table must live under tables/ (generated) or
# tables/manual/ (hand-edited) and be pulled in with \input. rebuttle.tex is a
# standalone reviewer-response document (not part of the paper) and is exempt.
BODY_TEX = ("example_paper.tex", "appendix.tex")

COMMENT_RE = re.compile(r"(?<!\\)%.*")

# Matches the start of a tabular-like environment (tabular, tabular*,
# tabularx, longtable). ``array`` is intentionally excluded: it is a math
# environment, not a table.
TABLE_ENV_RE = re.compile(r"\\begin\{(tabularx?\*?|longtable)\}")

# Matches a standalone ``nan`` token (any case) so we don't trip on words
# like "nano" or model names that happen to contain the substring.
NAN_RE = re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])", re.IGNORECASE)

REF_PATTERNS = [
    re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}"),
    re.compile(r"\\input\s*\{([^}]+)\}"),
    re.compile(r"\\include\s*\{([^}]+)\}"),
    re.compile(r"\\usepackage(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}"),
    re.compile(r"\\RequirePackage(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}"),
    re.compile(r"\\bibliographystyle\s*\{([^}]+)\}"),
    re.compile(r"\\bibliography\s*\{([^}]+)\}"),
]


def _collect_reference_names() -> set[str]:
    """Scan .tex and .sty sources for file references. Style files are
    included because LaTeX classes pull in other styles via \\RequirePackage."""
    refs: set[str] = set()
    sources = list(PAPER_DIR.rglob("*.tex")) + list(PAPER_DIR.rglob("*.sty"))
    for src in sources:
        text = COMMENT_RE.sub("", src.read_text(encoding="utf-8", errors="ignore"))
        for pat in REF_PATTERNS:
            for match in pat.findall(text):
                for name in match.split(","):
                    name = name.strip().split("/")[-1]
                    if not name:
                        continue
                    refs.add(name)
                    if "." in name:
                        refs.add(name.rsplit(".", 1)[0])
    return refs


def _attachment_files() -> list[Path]:
    files: list[Path] = []
    for sub in ATTACHMENT_DIRS:
        d = PAPER_DIR / sub
        if d.is_dir():
            files.extend(p for p in d.rglob("*") if p.is_file())
    return files


def check_unused_attachments() -> list[str]:
    """Every file under figures/ and styles/ must be referenced from a source."""
    refs = _collect_reference_names()
    return [
        f"{f.relative_to(PAPER_DIR)} is not referenced from any .tex/.sty source"
        for f in sorted(_attachment_files())
        if f.stem not in refs and f.name not in refs
    ]


def check_table_nans() -> list[str]:
    """No table row may carry a NaN value (e.g. from a missing metric).

    Scans every ``.tex`` source for ``nan`` tokens on lines that look like
    table rows (those with an ``&`` column separator), after stripping
    comments so commented-out rows are ignored."""
    errors: list[str] = []
    for src in sorted(PAPER_DIR.rglob("*.tex")):
        text = src.read_text(encoding="utf-8", errors="ignore")
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = COMMENT_RE.sub("", raw)
            if "&" not in line:
                continue
            if NAN_RE.search(line):
                rel = src.relative_to(PAPER_DIR)
                errors.append(f"{rel}:{lineno} table row contains NaN: {line.strip()}")
    return errors


def check_raw_tables() -> list[str]:
    """The paper body may not embed a raw tabular environment.

    Tables must live under ``tables/`` (generated) or ``tables/manual/``
    (hand-edited) and be pulled into the main text or appendix with
    ``\\input``. Scans the body sources (comments stripped) for a
    ``\\begin{tabular...}`` / ``\\begin{longtable}`` and flags any hit."""
    errors: list[str] = []
    for name in BODY_TEX:
        src = PAPER_DIR / name
        if not src.is_file():
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = COMMENT_RE.sub("", raw)
            match = TABLE_ENV_RE.search(line)
            if match:
                errors.append(
                    f"{name}:{lineno} raw table \\begin{{{match.group(1)}}} "
                    "in paper body -- move it to tables/ (or tables/manual/) "
                    "and \\input it instead"
                )
    return errors


# Main-body tables must be built from the canonical 50-sample PG19 run. Each
# table generator stamps "% paper-lint: n_samples=<N>" into its .tex (read from
# the sample count it aggregates); the check below enforces N=50 for every table
# \input by the main text. Entries here are body tables that are intentionally
# exempt, each with a justification.
MAIN_BODY_TEX = "example_paper.tex"
EXPECTED_N_SAMPLES = "50"
SAMPLE_COUNT_ALLOWLIST = {
    # Benchmark table: its "samples" are HellaSwag/ARC-Easy instances, not PG19
    # cramming samples, so the 50-sample rule does not apply.
    "semantic_evaluation": "benchmark instances (HellaSwag/ARC), not PG19 samples",
    # Hand-authored benchmark table (tables/manual/): token-normalized HellaSwag/
    # ARC accuracy under different scoring strategies, not PG19 cramming samples.
    "semantic_benchmarks_token": "benchmark instances (HellaSwag/ARC), not PG19 samples",
    # TODO: remove once the 50-sample full-cramming re-run lands. The "Full"
    # baseline currently exists only at limit_10, so this table is 10-sample.
    "full_vs_progressive": "pending 50-sample full-cramming re-run",
    # Hand-authored compact summary (tables/manual/) of the Llama
    # reconstruction-failure runs in tab:compression_reconstruction_summary;
    # not the 50-sample PG19 cramming protocol.
    "compression_reconstruction_main": "hand-authored summary of Llama reconstruction runs, not PG19 cramming samples",
}
INPUT_TABLE_RE = re.compile(r"\\input\s*\{tables/([^}]+)\}")
SAMPLE_STAMP_RE = re.compile(r"^%\s*paper-lint:\s*n_samples=(\S+)\s*$", re.MULTILINE)


def check_body_table_sample_count() -> list[str]:
    """Every table \\input by the main body must be the canonical 50-sample run.

    Table generators stamp ``% paper-lint: n_samples=<N>`` into each generated
    .tex. This parses that stamp for every table pulled into ``example_paper.tex``
    and fails unless it reads ``n_samples=50``. Tables in
    ``SAMPLE_COUNT_ALLOWLIST`` are exempt with a documented reason."""
    errors: list[str] = []
    body = PAPER_DIR / MAIN_BODY_TEX
    if not body.is_file():
        return errors
    text = COMMENT_RE.sub("", body.read_text(encoding="utf-8", errors="ignore"))
    for raw_name in INPUT_TABLE_RE.findall(text):
        slug = raw_name.strip().split("/")[-1]
        if slug in SAMPLE_COUNT_ALLOWLIST:
            continue
        tex_path = PAPER_DIR / "tables" / f"{slug}.tex"
        if not tex_path.is_file():
            errors.append(f"{MAIN_BODY_TEX}: \\input{{tables/{slug}}} but tables/{slug}.tex is missing")
            continue
        match = SAMPLE_STAMP_RE.search(tex_path.read_text(encoding="utf-8", errors="ignore"))
        if match is None:
            errors.append(
                f"tables/{slug}.tex has no '% paper-lint: n_samples=' provenance stamp -- "
                "regenerate it with its generator (or allowlist it with a reason)"
            )
        elif match.group(1) != EXPECTED_N_SAMPLES:
            errors.append(
                f"tables/{slug}.tex: n_samples={match.group(1)}, but main-body tables must use the "
                f"{EXPECTED_N_SAMPLES}-sample run (fix the generator's source dirs, or allowlist with a reason)"
            )
    return errors


CHECKS: list[tuple[str, Callable[[], list[str]]]] = [
    ("unused-attachments", check_unused_attachments),
    ("table-nans", check_table_nans),
    ("raw-tables", check_raw_tables),
    ("body-table-samples", check_body_table_sample_count),
]


def main() -> int:
    overall_ok = True
    for name, check in CHECKS:
        errors = check()
        if errors:
            overall_ok = False
            print(f"FAIL [{name}]", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
        else:
            print(f"OK   [{name}]")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
