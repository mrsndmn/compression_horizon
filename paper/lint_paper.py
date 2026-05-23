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

COMMENT_RE = re.compile(r"(?<!\\)%.*")

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


CHECKS: list[tuple[str, Callable[[], list[str]]]] = [
    ("unused-attachments", check_unused_attachments),
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
