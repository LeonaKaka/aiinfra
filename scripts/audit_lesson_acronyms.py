#!/usr/bin/env python3
"""Report high-confidence acronym-like terms missing from the lesson registry.

Only paragraph/list teaching prose is scanned. Code, headings, diagrams and UI
labels are deliberately excluded: their identifiers are useful locally but
should not force glossary entries or invented acronym expansions.
"""
from __future__ import annotations

import html
import re
from collections import defaultdict
from pathlib import Path

from lesson_terms import MAIN_RE, TERM_SECTION_RE, TERMS

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"

ALLOW = {
    # Product / project / library names.
    "CUDA", "NVIDIA", "PyTorch", "Megatron", "MCore", "vLLM", "NIXL",
    "NVLink", "NVSwitch", "GPUDirect", "InfiniBand", "Ethernet", "CUTLASS",
    "DeepEP", "HybridEP", "LMCache", "EAGLE", "Adam", "AdamW",
    # Method/architecture names whose capitalization is a name rather than a
    # reason to invent a mechanical expansion.
    "RMSNorm", "SwiGLU",
    # Mathematical symbols / harmless prose shorthand.
    "Q", "K", "V", "X", "Y", "H", "N", "S", "B", "W", "ID", "IDs", "OK", "VS",
}
ALLOW.update(TERMS)

STOP_WORDS = {
    "AND", "ARE", "AS", "AT", "BE", "BY", "CAN", "DO", "DOES", "FOR",
    "FROM", "HAS", "HAVE", "HOW", "IN", "IS", "IT", "NOT", "OF", "ON",
    "OR", "THE", "THIS", "TO", "USE", "USES", "WHAT", "WHEN", "WHERE",
    "WHILE", "WHO", "WHY", "WITH", "WITHOUT", "YOU",
}

CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"[A-Z]{2,8}[0-9]{0,2}"
    r"|[A-Z][a-z]{1,3}[A-Z][A-Za-z0-9]{0,8}"
    r"|[0-9]+[A-Z][A-Z0-9]{1,6}"
    r")(?![A-Za-z0-9_])"
)

PROSE_RE = re.compile(r"<(?:p|li)\b[^>]*>(.*?)</(?:p|li)>", re.S | re.I)
NON_PROSE_RE = re.compile(
    r"<(?:script|style|svg|pre|code)\b.*?</(?:script|style|svg|pre|code)>",
    re.S | re.I,
)
QUANTITY_RE = re.compile(r"^(?:\d+(?:GB|MB|KB|B|K|H|D)|GPU\d+|TP\d+|MB\d+)$", re.I)


def article_prose(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    m = MAIN_RE.search(source)
    if not m:
        return ""
    body = TERM_SECTION_RE.sub(" ", m.group("body"))
    chunks = []
    for match in PROSE_RE.finditer(body):
        chunk = NON_PROSE_RE.sub(" ", match.group(1))
        chunk = re.sub(r"<[^>]+>", " ", chunk)
        chunks.append(html.unescape(chunk))
    return "\n".join(chunks)


def main() -> int:
    hits: dict[str, set[str]] = defaultdict(set)
    for path in sorted(LESSONS.glob("**/*.html")):
        rel = str(path.relative_to(ROOT))
        for token in CANDIDATE_RE.findall(article_prose(path)):
            if token in ALLOW or token.upper() in STOP_WORDS:
                continue
            if QUANTITY_RE.fullmatch(token):
                continue
            hits[token].add(rel)

    if not hits:
        print("Acronym candidate audit: 0 high-confidence unregistered candidates.")
        return 0

    print(f"Acronym candidate audit: {len(hits)} high-confidence unregistered candidate(s).")
    for token in sorted(hits, key=str.casefold):
        pages = ", ".join(sorted(hits[token]))
        print(f"{token}: {pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
