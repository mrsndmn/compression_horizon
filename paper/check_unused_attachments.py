#!/usr/bin/env python3
"""Fail the pipeline if any file under paper/figures or paper/styles is never
referenced from a .tex source in the paper directory.

Matching is by basename (with and without extension), since LaTeX resolves
includes via TEXINPUTS / \\graphicspath rather than literal paths.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
ATTACHMENT_DIRS = ("figures", "styles")

REF_PATTERNS = [
    re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}"),
    re.compile(r"\\input\s*\{([^}]+)\}"),
    re.compile(r"\\include\s*\{([^}]+)\}"),
    re.compile(r"\\usepackage(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}"),
    re.compile(r"\\RequirePackage(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}"),
    re.compile(r"\\bibliographystyle\s*\{([^}]+)\}"),
    re.compile(r"\\bibliography\s*\{([^}]+)\}"),
]

COMMENT_RE = re.compile(r"(?<!\\)%.*")


def collect_reference_names() -> set[str]:
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


def attachment_files() -> list[Path]:
    files: list[Path] = []
    for sub in ATTACHMENT_DIRS:
        d = PAPER_DIR / sub
        if d.is_dir():
            files.extend(p for p in d.rglob("*") if p.is_file())
    return files


def main() -> int:
    refs = collect_reference_names()
    files = attachment_files()
    unused = [f for f in files if f.stem not in refs and f.name not in refs]

    if unused:
        print("ERROR: unused attachment files in paper/:", file=sys.stderr)
        for f in sorted(unused):
            print(f"  - {f.relative_to(PAPER_DIR)}", file=sys.stderr)
        print(
            "\nEither reference these files from a .tex source or delete them.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: all {len(files)} attachments under {', '.join(ATTACHMENT_DIRS)}/ are referenced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
