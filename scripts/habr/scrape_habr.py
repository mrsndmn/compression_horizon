#!/usr/bin/env python
"""Scrape Habr articles to Markdown for style calibration.

Reuses the approach of Ilya Gusev's ``create_habr.py``
(https://github.com/IlyaGusev/rulm/blob/master/data_processing/create_habr.py):
query the public Habr API v2 for an article, pull ``titleHtml`` / ``textHtml`` /
``leadData.textHtml``, and convert the body HTML to Markdown. Gusev uses the
``html2text`` library; that package is not installed here, so we implement an
equivalent BeautifulSoup-based HTML->Markdown pass covering the tags Habr emits.

The scraped Markdown lands under ``habr/refs/`` and is committed as reference
material for matching the AIRI/Habr writing style — it is NOT the article.

Usage:
    python scripts/habr/scrape_habr.py                 # default reference set
    python scripts/habr/scrape_habr.py 906592 804515   # explicit ids/urls
    python scripts/habr/scrape_habr.py --force         # re-scrape existing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

API = "https://habr.com/kek/v2/articles/{}/?fl=ru&hl=ru"
UA = {"User-Agent": "Mozilla/5.0 (compatible; habr-style-scraper/1.0)"}

# Default reference set for the Progressive Cramming Habr article:
#   906592 — Kuratov et al. "cramming" review (must be referenced in the article)
#   804515 — AIRI company blog (style reference)
#   816125 — AIRI company blog (style reference)
DEFAULT_IDS = ["906592", "804515", "816125"]

REFS_DIR = Path(__file__).resolve().parents[2] / "habr" / "refs"


def article_id(token: str) -> str:
    """Accept a raw id or any habr URL and return the numeric article id."""
    m = re.search(r"(\d{5,})", token)
    if not m:
        raise ValueError(f"could not find an article id in {token!r}")
    return m.group(1)


def fetch(post_id: str) -> dict:
    req = urllib.request.Request(API.format(post_id), headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def slugify(text: str, maxlen: int = 60) -> str:
    text = re.sub(r"<[^>]+>", "", text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:maxlen] or "article"


# --------------------------------------------------------------------------- #
# HTML -> Markdown (BeautifulSoup); mirrors what html2text would produce.
# --------------------------------------------------------------------------- #
_INLINE_SKIP = {"script", "style"}


def _inline(node) -> str:
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    if not isinstance(node, Tag) or node.name in _INLINE_SKIP:
        return ""
    name = node.name
    inner = "".join(_inline(c) for c in node.children)
    if name in ("strong", "b"):
        return f"**{inner.strip()}**" if inner.strip() else ""
    if name in ("em", "i"):
        return f"*{inner.strip()}*" if inner.strip() else ""
    if name == "code":
        return f"`{inner.strip()}`"
    if name == "a":
        href = node.get("href", "")
        text = inner.strip() or href
        return f"[{text}]({href})" if href else text
    if name == "br":
        return "\n"
    if name in ("sub", "sup"):
        return inner
    return inner


def _block(node, out: list[str]) -> None:
    if isinstance(node, NavigableString):
        text = re.sub(r"\s+", " ", str(node)).strip()
        if text:
            out.append(text)
        return
    if not isinstance(node, Tag) or node.name in _INLINE_SKIP:
        return
    name = node.name
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        out.append("#" * level + " " + _inline(node).strip())
    elif name == "p":
        text = _inline(node).strip()
        if text:
            out.append(text)
    elif name == "blockquote":
        text = _inline(node).strip()
        if text:
            out.append("\n".join("> " + ln for ln in text.splitlines()))
    elif name in ("ul", "ol"):
        ordered = name == "ol"
        for i, li in enumerate(node.find_all("li", recursive=False), 1):
            bullet = f"{i}." if ordered else "-"
            out.append(f"{bullet} {_inline(li).strip()}")
    elif name == "pre":
        code = node.get_text()
        out.append("```\n" + code.rstrip() + "\n```")
    elif name == "img":
        src = node.get("src") or node.get("data-src") or ""
        alt = node.get("alt", "")
        if src:
            out.append(f"![{alt}]({src})")
    elif name == "figure":
        for c in node.children:
            _block(c, out)
    elif name == "figcaption":
        text = _inline(node).strip()
        if text:
            out.append(f"*{text}*")
    elif name == "table":
        out.append(_table(node))
    elif name in ("div", "section", "article", "span"):
        for c in node.children:
            _block(c, out)
    else:
        text = _inline(node).strip()
        if text:
            out.append(text)


def _table(node) -> str:
    rows = []
    for tr in node.find_all("tr"):
        cells = [_inline(c).strip() for c in tr.find_all(["th", "td"], recursive=False)]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * ncol) + " |"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    out: list[str] = []
    for node in soup.children:
        _block(node, out)
    md = "\n\n".join(b for b in out if b.strip())
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def strip_tags(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text().strip()


def to_markdown_doc(data: dict, post_id: str) -> tuple[str, str]:
    title = strip_tags(data.get("titleHtml", "")) or f"habr-{post_id}"
    author = (data.get("author") or {}).get("alias", "")
    lang = data.get("lang", "")
    published = data.get("timePublished", "")
    hubs = [h.get("title", "") for h in data.get("hubs", [])]
    tags = [t.get("titleHtml", "") for t in data.get("tags", [])]
    lead = html_to_markdown((data.get("leadData") or {}).get("textHtml", ""))
    body = html_to_markdown(data.get("textHtml", ""))

    fm = [
        "---",
        f"habr_id: {post_id}",
        f"url: https://habr.com/ru/articles/{post_id}/",
        f'title: "{title}"',
        f"author: {author}",
        f"lang: {lang}",
        f"published: {published}",
        f"hubs: {hubs}",
        f"tags: {tags}",
        "source: scripts/habr/scrape_habr.py (Habr API v2)",
        "---",
    ]
    parts = ["\n".join(fm), f"# {title}"]
    if lead:
        parts.append(lead)
    parts.append(body)
    return "\n\n".join(parts) + "\n", slugify(title)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="*", help="article ids or habr URLs (default: reference set)")
    ap.add_argument("--force", action="store_true", help="re-scrape even if the file exists")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    tokens = args.ids or DEFAULT_IDS
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for token in tokens:
        pid = article_id(token)
        existing = list(REFS_DIR.glob(f"{pid}-*.md"))
        if existing and not args.force:
            print(f"[skip] {pid} -> {existing[0].name} (exists; --force to redo)")
            continue
        try:
            data = fetch(pid)
            doc, slug = to_markdown_doc(data, pid)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {pid}: {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
            continue
        out = REFS_DIR / f"{pid}-{slug}.md"
        for old in existing:
            old.unlink()
        out.write_text(doc, encoding="utf-8")
        print(f"[ok]   {pid} -> {out.relative_to(REFS_DIR.parents[1])} ({len(doc)} chars)")
        time.sleep(args.sleep)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
