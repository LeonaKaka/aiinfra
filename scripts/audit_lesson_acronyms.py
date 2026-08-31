#!/usr/bin/env python3
"""Report high-confidence acronym-like terms missing from the lesson registry.

This audit intentionally scans *natural-language teaching prose*, not code/API
identifiers or diagram labels. It is a discovery guard: findings are reviewed
before becoming registry entries, because product names and implementation
symbols should not be given invented expansions.
"""
from __future__ import annotations

import html
import re
from collections import defaultdict
from pathlib import Path

from lesson_terms import MAIN_RE, TERM_SECTION_RE, TERMS

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"

# Names/words that can legitimately appear in prose but are not glossary
# abbreviations that should be mechanically expanded.
ALLOW = {
    # Product / project / library names.
    "CUDA", "NVIDIA", "PyTorch", "Megatron", "MCore", "vLLM", "NIXL",
    "NVLink", "NVSwitch", "GPUDirect", "InfiniBand", "Ethernet", "CUTLASS",
    "DeepEP", "HybridEP", "LMCache", "EAGLE", "Adam", "AdamW",
    # Common method / architecture names whose capitalization is a name, not
    # necessarily an acronym expansion requirement. These remain reviewable
    # separately if a lesson needs a local explanation.
    "RMSNorm", "SwiGLU",
    # Mathematical shape symbols and prose shorthand.
    "Q", "K", "V", "X", "Y", "H", "N", "S", "B", "W",
    # Common English/UI words that can be capitalized for emphasis.
    "AI",  # already in registry; kept for clarity
    "API", # already in registry; kept for clarity
    "ID", "IDs", "OK", "VS",
}
ALLOW.update(TERMS)

# Common English words that are occasionally written in caps in explanatory
# prose. Keeping this small is deliberate; the scanner should stay sensitive.
STOP_WORDS = {
    "AND", "ARE", "AS", "AT", "BE", "BY", "CAN", "DO", "DOES", "FOR",
    "FROM", "HAS", "HAVE", "HOW", "IN", "IS", "IT", "NOT", "OF", "ON",
    "OR", "THE", "THIS", "TO", "USE", "USES", "WHAT", "WHEN", "WHERE",
    "WHILE", "WHO", "WHY", "WITH", "WITHOUT", "YOU",
}

# High-confidence shapes:
#   ABI / MPI / MTU / INT8 / MFU / VPP
#   RoCE / LoRA / GiB / MiB / ZeRO
#   1F1B
# Avoid arbitrary long ALL-CAPS labels by capping plain initialisms at 8 chars.
CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"[A-Z]{2,8}[0-9]{0,2}"
    r"|[A-Z][a-z]{1,3}[A-Z][A-Za-z0-9]{0,8}"
    r"|[0-9]+[A-Z][A-Z0-9]{1,6}"
    r")(?![A-Za-z0-9_])"
)

# Markup that is useful to learners but should not drive glossary discovery.
# Code/class/env identifiers and display labels are handled by their local
# explanation rather than forced into the global abbreviation registry.
NON_PROSE_RE = re.compile(
    r"<(?:script|style|svg|pre|code|h1|h2|h3|small)\b.*?</(?:script|style|svg|pre|code|h1|h2|h3|small)>",
    re.S | re.I,
)

# Quantities / rank labels / tensor-shape shorthand are not abbreviations.
QUANTITY_RE = re.compile(r"^(?:\d+(?:GB|MB|KB|B|K|H|D)|GPU\d+|TP\d+|MB\d+)$", re.I)


def article_prose(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    m = MAIN_RE.search(source)
    if not m:
        return ""
    body = TERM_SECTION_RE.sub(" ", m.group("body"))
    body = NON_PROSE_RE.sub(" ", body)
    body = re.sub(r"<[^>]+>", " ", body)
    return html.unescape(body)


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
