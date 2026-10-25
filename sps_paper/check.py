"""Check the standalone SPS manuscript's local references and generated tables."""

from __future__ import annotations

import re
from pathlib import Path

COMMENT = re.compile(r"(?<!\\)%[^\n]*")
INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}")
CITE = re.compile(r"\\cite\w*\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]+)\}")
BIB = re.compile(r"\\bibliography\s*\{([^}]+)\}")
RAW_TABLE = re.compile(r"\\begin\{(?:tabularx?\*?|longtable)\}")
NONFINITE = re.compile(r"(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])", re.I)


def resolve(root: Path, name: str, extensions: tuple[str, ...]) -> bool:
    return any((root / (name + extension)).is_file() for extension in ("", *extensions))


def check(root: Path) -> list[str]:
    root = Path(root)
    if not (root / "main.tex").is_file():
        return ["missing main.tex"]
    errors, citations, bibliography = [], set(), set()
    sources = sorted(p for p in root.rglob("*.tex") if "build" not in p.relative_to(root).parts)
    for path in sources:
        rel = path.relative_to(root)
        source = COMMENT.sub("", path.read_text())
        if rel.name in {"main.tex", "appendix.tex"} and RAW_TABLE.search(source):
            errors.append(f"{rel}: raw table in manuscript body; use a generated tables/ input")
        for number, line in enumerate(source.splitlines(), 1):
            if "&" in line and NONFINITE.search(line):
                errors.append(f"{rel}:{number}: nonfinite value in table")
        for name in INPUT.findall(source):
            if not resolve(root, name, (".tex", ".tikz")):
                errors.append(f"{rel}: missing input {name}")
        for name in GRAPHIC.findall(source):
            candidates = [name, "figures/" + name]
            if not any(resolve(root, candidate, (".pdf", ".png", ".jpg")) for candidate in candidates):
                errors.append(f"{rel}: missing figure {name}")
        for group in CITE.findall(source):
            citations.update(key.strip() for key in group.split(","))
        for group in BIB.findall(source):
            for name in group.split(","):
                name = name.strip()
                if not resolve(root, name, (".bib",)):
                    errors.append(f"{rel}: missing bibliography {name}")
                else:
                    path = root / (name if name.endswith(".bib") else name + ".bib")
                    bibliography.update(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", path.read_text()))
    errors.extend(f"unknown citation {key}" for key in sorted(citations - bibliography))
    return errors


if __name__ == "__main__":
    issues = check(Path(__file__).resolve().parent)
    for issue in issues:
        print(issue)
    if issues:
        raise SystemExit(1)
    print("SPS paper checks passed")
