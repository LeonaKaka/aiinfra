#!/usr/bin/env python3
"""Audit first pedagogical use of English/source-facing terms.

During migration this script reports backlog and exits 0 by default.
Pass --strict to make missing first-use explanations fail CI.

Accepted source-term form:
    shape（形状：描述 tensor 每一维的长度）

Accepted acronym forms:
    Tensor Parallel (TP)（张量并行：把单层张量计算切到多个 ranks）
    TP（张量并行：把单层张量计算切到多个 ranks）

Only learner-facing teaching prose is scanned. Code/pre/SVG/tables, headings,
navigation, term tables, exact identifiers and source paths are excluded.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from first_use_terms import SOURCE_TERMS
from lesson_terms import TERMS, MAIN_RE, TERM_SECTION_RE

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"
LABS = ROOT / "labs"

SKIP_BLOCK_RE = re.compile(
    r"<(?:script|style|pre|svg|table|nav|aside)\b.*?</(?:script|style|pre|svg|table|nav|aside)>",
    re.S | re.I,
)
HEADING_RE = re.compile(r"<h[1-3]\b.*?</h[1-3]>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")

ALIASES = {
    "tensor": ("tensor", "tensors"),
    "kernel": ("kernel", "kernels"),
    "process": ("process", "processes"),
    "rank": ("rank", "ranks"),
    "process group": ("process group", "process groups"),
    "collective": ("collective", "collectives"),
    "shard": ("shard", "shards"),
    "replica": ("replica", "replicas"),
    "stage": ("stage", "stages"),
    "microbatch": ("microbatch", "microbatches"),
    "bucket": ("bucket", "buckets"),
    "router": ("router", "routers"),
    "dispatcher": ("dispatcher", "dispatchers"),
    "assignment": ("assignment", "assignments"),
    "request": ("request", "requests"),
    "scheduler": ("scheduler", "schedulers"),
    "iteration": ("iteration", "iterations"),
    "buffer": ("buffer", "buffers"),
    "block": ("block", "blocks"),
    "descriptor": ("descriptor", "descriptors"),
    "region": ("region", "regions"),
    "producer": ("producer", "producers"),
    "consumer": ("consumer", "consumers"),
    "workload": ("workload", "workloads"),
    "parameter": ("parameter", "parameters"),
    "gradient": ("gradient", "gradients"),
    "activation": ("activation", "activations"),
}

def teaching_text(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    match = MAIN_RE.search(source)
    body = match.group("body") if match else source
    body = TERM_SECTION_RE.sub(" ", body)
    body = SKIP_BLOCK_RE.sub(" ", body)
    body = UI_BLOCK_RE.sub(" ", body)
    chunks = []
    for match in PROSE_RE.finditer(body):
        item = TAG_RE.sub(" ", match.group(1))
        chunks.append(html.unescape(item))
    return SPACE_RE.sub(" ", " ".join(chunks)).strip()

def standalone_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", re.I)

def first_source_use(text: str, key: str):
    aliases = sorted(ALIASES.get(key, (key,)), key=len, reverse=True)
    matches = []
    for alias in aliases:
        m = standalone_pattern(alias).search(text)
        if m:
            matches.append((m.start(), m.end(), m.group(0)))
    return min(matches) if matches else None

def has_source_annotation(text: str, start: int, end: int, key: str) -> bool:
    chinese, _ = SOURCE_TERMS[key]
    tail = text[start:min(len(text), end + 120)]
    term = re.escape(text[start:end])
    # Require Chinese name + colon + a non-trivial explanation.
    pat = re.compile(
        rf"^{term}（{re.escape(chinese)}：[^）]{{4,}}）",
        re.I,
    )
    return bool(pat.search(tail))

def acronym_first_use(text: str, abbr: str):
    if abbr == "P/D":
        pat = re.compile(r"(?<![A-Za-z0-9])P/D(?![A-Za-z0-9])")
    else:
        pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(abbr)}(?![A-Za-z0-9])")
    return pat.search(text)

def has_acronym_annotation(text: str, match: re.Match[str], abbr: str) -> bool:
    english, chinese, _, expand = TERMS[abbr]
    start = max(0, match.start() - len(english) - len(abbr) - 12)
    tail = text[start:min(len(text), match.end() + 140)]
    explanation = rf"（{re.escape(chinese)}：[^）]{{4,}}）"
    short = rf"{re.escape(abbr)}{explanation}"
    if re.search(short, tail):
        return True
    if expand:
        long = rf"{re.escape(english)}\s*\({re.escape(abbr)}\){explanation}"
        if re.search(long, tail, re.I):
            return True
    return False

def audit_page(path: Path):
    text = teaching_text(path)
    missing = []

    for key in SOURCE_TERMS:
        use = first_source_use(text, key)
        if not use:
            continue
        start, end, shown = use
        if not has_source_annotation(text, start, end, key):
            chinese, explanation = SOURCE_TERMS[key]
            missing.append(
                f"{shown} -> {shown}（{chinese}：{explanation}）"
            )

    for abbr, (english, chinese, explanation, expand) in TERMS.items():
        m = acronym_first_use(text, abbr)
        if not m:
            continue
        if not has_acronym_annotation(text, m, abbr):
            preferred = (
                f"{english} ({abbr})（{chinese}：{explanation.rstrip('。')}）"
                if expand else
                f"{abbr}（{chinese}：{explanation.rstrip('。')}）"
            )
            missing.append(f"{abbr} -> {preferred}")

    return missing

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    paths = sorted(LESSONS.glob("**/*.html")) + sorted(LABS.glob("*.html"))
    pages = {}
    for path in paths:
        missing = audit_page(path)
        if missing:
            pages[str(path.relative_to(ROOT))] = missing

    if not pages:
        print("First-use terminology audit: all teaching pages pass.")
        return 0

    total = sum(len(v) for v in pages.values())
    print(
        f"First-use terminology audit: {total} missing explanation(s) "
        f"across {len(pages)} page(s)."
    )
    for rel, items in pages.items():
        print(rel)
        for item in items:
            print(f"  - {item}")

    return 1 if args.strict else 0

if __name__ == "__main__":
    raise SystemExit(main())
