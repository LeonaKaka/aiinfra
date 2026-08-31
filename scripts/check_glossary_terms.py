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

# Explicit label identity avoids substring ambiguity such as RDMA vs
# GPUDIRECT RDMA while keeping the curated set small and readable.
CURATED_GLOSSARY_CARDS = {
    "GEMM": "GEMM",
    "GQA": "GQA",
    "DRAM": "DRAM",
    "HBM": "HBM",
    "GDDR": "GDDR",
    "NCCL": "NCCL",
    "SGD": "SGD",
    "DP": "DP",
    "TP": "TP",
    "PP": "PP",
    "VPP": "VPP",
    "SP": "SP",
    "CP": "CP",
    "EP": "EP",
    "1F1B": "1F1B",
    "ZeRO": "ZeRO",
    "GTP": "GTP",
    "TTFT": "TTFT",
    "ITL": "ITL",
    "TPOT": "TPOT",
    "APC": "APC / PREFIX CACHE",
    "IOMMU": "IOMMU",
    "MTU": "MTU",
    "RDMA": "RDMA",
    "UCX": "UCX",
    "NIXL": "NIXL",
}

CARD_RE = re.compile(
    r'<div class="term">\s*<small>(?P<label>.*?)</small>\s*'
    r'<b>(?P<title>.*?)</b>(?P<body>.*?)</div>',
    re.S | re.I,
)
TAG_RE = re.compile(r"<[^>]+>")


def plain(text: str) -> str:
    return " ".join(html.unescape(TAG_RE.sub(" ", text)).split())


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
    by_label: dict[str, list[dict[str, str]]] = {}
    for card in cards:
        by_label.setdefault(card["label"], []).append(card)

    errors: list[str] = []
    for term, label in CURATED_GLOSSARY_CARDS.items():
        if term not in TERMS:
            errors.append(f"{term}: curated but missing from canonical TERMS registry")
            continue

        matches = by_label.get(label, [])
        if not matches:
            errors.append(f"{term}: missing Glossary card with label {label!r}")
            continue
        if len(matches) > 1:
            errors.append(f"{term}: duplicate Glossary cards with label {label!r}")
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
        f"{len(CURATED_GLOSSARY_CARDS)} curated terms; canonical names aligned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
