#!/usr/bin/env python3
"""Keep the curated global Glossary aligned with canonical lesson terminology.

The global Glossary is intentionally a curated subset, not a dump of every term
used in lessons. Canonical English names live in ``lesson_terms.TERMS``; this
check owns only the small selection contract for concepts that should always be
recoverable from the global reference page.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from lesson_terms import TERMS

ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / "glossary" / "index.html"

# Keep this list selective: cross-lesson concepts, common confusion points, and
# version-sensitive names that materially affect source reading.
CURATED_GLOSSARY_TERMS = (
    "GEMM",
    "GQA",
    "DRAM",
    "HBM",
    "GDDR",
    "NCCL",
    "SGD",
    "DP",
    "TP",
    "PP",
    "VPP",
    "SP",
    "CP",
    "EP",
    "1F1B",
    "ZeRO",
    "GTP",
    "TTFT",
    "ITL",
    "TPOT",
    "APC",
    "IOMMU",
    "MTU",
    "RDMA",
    "UCX",
    "NIXL",
)

CARD_RE = re.compile(
    r'<div class="term">\s*<small>(?P<label>.*?)</small>\s*'
    r'<b>(?P<title>.*?)</b>(?P<body>.*?)</div>',
    re.S | re.I,
)
TAG_RE = re.compile(r"<[^>]+>")


def plain(text: str) -> str:
    return " ".join(html.unescape(TAG_RE.sub(" ", text)).split())


def label_has_term(label: str, term: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", label
    ) is not None


def main() -> int:
    source = GLOSSARY.read_text(encoding="utf-8")
    cards = [
        {
            "label": plain(m.group("label")),
            "title": plain(m.group("title")),
            "text": plain(m.group(0)),
        }
        for m in CARD_RE.finditer(source)
    ]

    errors: list[str] = []
    for term in CURATED_GLOSSARY_TERMS:
        if term not in TERMS:
            errors.append(f"{term}: curated but missing from canonical TERMS registry")
            continue

        matches = [card for card in cards if label_has_term(card["label"], term)]
        if not matches:
            errors.append(f"{term}: missing curated Glossary card")
            continue
        if len(matches) > 1:
            labels = ", ".join(card["label"] for card in matches)
            errors.append(f"{term}: appears in multiple Glossary labels: {labels}")
            continue

        english = TERMS[term][0]
        if english.casefold() not in matches[0]["text"].casefold():
            errors.append(
                f"{term}: Glossary card does not contain canonical English name {english!r}"
            )

    if errors:
        print("Global glossary consistency check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Global glossary checked: "
        f"{len(CURATED_GLOSSARY_TERMS)} curated terms; canonical names aligned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
