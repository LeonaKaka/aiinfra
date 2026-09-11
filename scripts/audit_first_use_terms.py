#!/usr/bin/env python3
"""Normalize and strictly audit the first pedagogical use of technical English.

Course rule:
    shape（形状：表示 tensor 各维度的长度）

The first substantive teaching occurrence keeps the English/source spelling and
adds a short Chinese explanation. Later uses stay in English. Headings, code,
paths, SVGs, navigation and lesson term tables do not consume the first use.

Course order is learn/01-* -> learn/08-*; Labs are scanned afterwards and inherit
terms already introduced by the lessons.

Run:
  python scripts/audit_first_use_terms.py --write
  python scripts/audit_first_use_terms.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from first_use_terms import SOURCE_TERMS
from lesson_terms import TERMS, TERM_ALIASES, MAIN_RE, TERM_SECTION_RE

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"
LABS = ROOT / "labs"

SKIP_TAGS = {"code", "pre", "script", "style", "svg", "table", "h1", "h2", "h3", "nav", "aside"}
SKIP_CLASSES = {
    "lesson-kicker", "section-no", "breadcrumb", "mobile-course-bar",
    "lesson-terms", "toc", "next-lesson",
}
TAG_RE = re.compile(r"(<[^>]+>)")
OPEN_TAG_RE = re.compile(r"<\s*([A-Za-z0-9]+)\b([^>]*)>")
CLOSE_TAG_RE = re.compile(r"</\s*([A-Za-z0-9]+)\s*>")
CLASS_RE = re.compile(r'class\s*=\s*["\']([^"\']*)["\']', re.I)

SOURCE_ALIASES = {
    "tensor": ("tensor", "tensors"),
    "shape": ("shape", "shapes"),
    "dtype": ("dtype", "dtypes"),
    "device": ("device", "devices"),
    "view": ("view", "views"),
    "projection": ("projection", "projections"),
    "attention score": ("attention score", "attention scores"),
    "attention head": ("attention head", "attention heads"),
    "Tensor Core": ("Tensor Core", "Tensor Cores"),
    "kernel": ("kernel", "kernels"),
    "stream": ("stream", "streams"),
    "process": ("process", "processes"),
    "process group": ("process group", "process groups"),
    "collective": ("collective", "collectives"),
    "communicator": ("communicator", "communicators"),
    "shard": ("shard", "shards", "sharded"),
    "replica": ("replica", "replicas", "replicated"),
    "pipeline stage": ("pipeline stage", "pipeline stages"),
    "microbatch": ("microbatch", "microbatches"),
    "bucket": ("bucket", "buckets"),
    "router": ("router", "routers"),
    "dispatcher": ("dispatcher", "dispatchers"),
    "assignment": ("assignment", "assignments"),
    "request": ("request", "requests"),
    "scheduler": ("scheduler", "Scheduler"),
    "iteration": ("iteration", "iterations"),
    "batch": ("batch", "batches"),
    "engine": ("engine", "engines"),
    "workload": ("workload", "workloads"),
    "benchmark": ("benchmark", "benchmarks"),
    "worker": ("worker", "workers"),
    "executor": ("executor", "executors"),
    "block": ("block", "blocks"),
    "block table": ("block table", "block tables"),
    "slot mapping": ("slot mapping", "slot mappings"),
    "eviction": ("eviction", "evict"),
    "producer": ("producer", "producers"),
    "consumer": ("consumer", "consumers"),
    "transfer": ("transfer", "transfers"),
    "transport": ("transport", "transports"),
    "buffer": ("buffer", "buffers"),
    "region": ("region", "regions"),
    "descriptor": ("descriptor", "descriptors"),
    "lease": ("lease", "leases"),
    "heartbeat": ("heartbeat", "heartbeats"),
    "completion": ("completion", "completions"),
    "peer": ("peer", "peers"),
    "parameter": ("parameter", "parameters"),
    "gradient": ("gradient", "gradients"),
    "activation": ("activation", "activations"),
    "optimizer": ("optimizer", "optimizers"),
    "optimizer state": ("optimizer state", "optimizer states"),
}


def short_hint(meaning: str) -> str:
    return re.split(r"[；。]", meaning, maxsplit=1)[0].strip()


def clean_chinese(abbr: str, chinese: str) -> str:
    prefix = abbr + " "
    return chinese[len(prefix):].strip() if chinese.startswith(prefix) else chinese


def boundary_pattern(alias: str, *, case_sensitive: bool = True) -> re.Pattern[str]:
    # Do not split source identifiers such as request-level or gpu_worker.
    flags = 0 if case_sensitive else re.I
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_-])",
        flags,
    )


def tokenize(body: str):
    stack: list[tuple[str, bool]] = []
    for part in TAG_RE.split(body):
        if not part:
            continue
        if not part.startswith("<"):
            yield "text", part, any(blocked for _, blocked in stack)
            continue

        close = CLOSE_TAG_RE.match(part)
        if close:
            tag = close.group(1).lower()
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == tag:
                    del stack[i:]
                    break
            yield "tag", part, any(blocked for _, blocked in stack)
            continue

        op = OPEN_TAG_RE.match(part)
        if op and not part.rstrip().endswith("/>") and not part.startswith("<!--"):
            tag = op.group(1).lower()
            classes: set[str] = set()
            m = CLASS_RE.search(op.group(2))
            if m:
                classes = set(m.group(1).split())
            blocked = tag in SKIP_TAGS or bool(classes & SKIP_CLASSES)
            stack.append((tag, blocked))
        yield "tag", part, any(blocked for _, blocked in stack)


def source_case_sensitive(term: str) -> bool:
    # Capitalized product/model names must not accidentally match ordinary words
    # such as "linear algebra" when we mean the PyTorch Linear layer.
    return any(ch.isupper() for ch in term)


def source_specs():
    specs = []
    for term, (chinese, hint) in SOURCE_TERMS.items():
        aliases = SOURCE_ALIASES.get(term, (term,))
        specs.append({
            "key": f"source:{term}",
            "aliases": aliases,
            "chinese": chinese,
            "hint": hint,
            "case_sensitive": source_case_sensitive(term),
            "kind": "source",
        })
    return specs


def acronym_specs():
    specs = []
    for abbr, (english, chinese, meaning, _) in TERMS.items():
        aliases = TERM_ALIASES.get(abbr, (abbr,))
        specs.append({
            "key": f"abbr:{abbr}",
            "abbr": abbr,
            "english": english,
            "aliases": aliases,
            "chinese": clean_chinese(abbr, chinese),
            "hint": short_hint(meaning),
            "case_sensitive": True,
            "kind": "abbr",
        })
    return specs


SPECS = source_specs() + acronym_specs()


def acceptable_annotation(text: str, end: int, chinese: str) -> tuple[bool, int]:
    """Accept authored explanations using either Chinese colon or comma."""
    if end >= len(text) or text[end] != "（":
        return False, end
    close = text.find("）", end + 1)
    if close < 0:
        return False, end
    inside = text[end + 1:close]
    for sep in ("：", "，"):
        prefix = chinese + sep
        if inside.startswith(prefix) and len(inside[len(prefix):].strip()) >= 4:
            return True, close + 1
    return False, end


def legacy_acronym_candidates(text: str, spec: dict, cursor: int):
    abbr = spec["abbr"]
    chinese = spec["chinese"]
    english = spec["english"]
    candidates = []

    # Old Chinese-first form: 图形处理器（GPU）
    p1 = re.compile(rf"{re.escape(chinese)}（{re.escape(abbr)}）")
    m = p1.search(text, cursor)
    if m:
        candidates.append((m.start(), m.end(), "legacy"))

    # Older English-first generated form: Graphics Processing Unit (GPU，图形处理器)
    p2 = re.compile(
        rf"{re.escape(english)}\s*\({re.escape(abbr)}[，,]\s*{re.escape(chinese)}\)",
        re.I,
    )
    m = p2.search(text, cursor)
    if m:
        candidates.append((m.start(), m.end(), "legacy"))

    return candidates


def next_candidate(text: str, cursor: int, seen: set[str]):
    best = None

    for spec in SPECS:
        if spec["key"] in seen:
            continue

        for alias in spec["aliases"]:
            pat = boundary_pattern(alias, case_sensitive=spec["case_sensitive"])
            m = pat.search(text, cursor)
            if not m:
                continue
            candidate = (
                m.start(),
                -len(m.group(0)),  # longest term wins at the same position
                m.end(),
                spec,
                "raw",
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate

        if spec["kind"] == "abbr":
            for start, end, mode in legacy_acronym_candidates(text, spec, cursor):
                candidate = (start, -(end - start), end, spec, mode)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate

    return best


def annotate_text(text: str, seen: set[str]) -> str:
    cursor = 0
    out: list[str] = []

    while cursor < len(text):
        candidate = next_candidate(text, cursor, seen)
        if candidate is None:
            out.append(text[cursor:])
            break

        start, _, end, spec, mode = candidate
        out.append(text[cursor:start])
        chinese = spec["chinese"]
        hint = spec["hint"]
        canonical = f"（{chinese}：{hint}）"

        if mode == "legacy":
            lexeme = spec["abbr"]
            out.append(lexeme + canonical)
            cursor = end
        else:
            lexeme = text[start:end]
            out.append(lexeme)
            ok, next_pos = acceptable_annotation(text, end, chinese)
            if ok:
                out.append(text[end:next_pos])
                cursor = next_pos
            else:
                # Replace a bare old Chinese-name-only parenthesis if present.
                simple = f"（{chinese}）"
                if text.startswith(simple, end):
                    cursor = end + len(simple)
                else:
                    cursor = end
                out.append(canonical)

        seen.add(spec["key"])

    return "".join(out)


def annotate_body(body: str, seen: set[str]) -> str:
    parts = []
    for kind, part, blocked in tokenize(body):
        if kind == "text" and not blocked:
            parts.append(annotate_text(part, seen))
        else:
            parts.append(part)
    return "".join(parts)


def normalize_page(source: str, seen: set[str]) -> str:
    m = MAIN_RE.search(source)
    if not m:
        return source

    body = m.group("body")
    table = TERM_SECTION_RE.search(body)
    if table:
        before = body[:table.start()]
        term_table = table.group(0)
        after = body[table.end():]
        body = annotate_body(before, seen) + term_table + annotate_body(after, seen)
    else:
        body = annotate_body(body, seen)

    return source[:m.start()] + m.group("open") + body + m.group("close") + source[m.end():]


def teaching_pages() -> list[Path]:
    # Lessons define the course language. Labs inherit those definitions and only
    # introduce terms that never appeared in the lessons.
    return (
        sorted(p for p in LESSONS.glob("**/*.html") if p.is_file())
        + sorted(p for p in LABS.glob("*.html") if p.is_file())
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    seen: set[str] = set()
    changed: list[Path] = []

    for path in teaching_pages():
        old = path.read_text(encoding="utf-8")
        new = normalize_page(old, seen)
        if new != old:
            changed.append(path)
            if args.write:
                path.write_text(new, encoding="utf-8")

    if args.write:
        print(
            f"First-use terminology normalized: {len(changed)} file(s) changed; "
            f"{len(seen)} technical terms introduced."
        )
        return 0

    if changed:
        print(
            f"First-use terminology audit FAILED: {len(changed)} page(s) "
            "need first-use explanations."
        )
        for path in changed:
            print(f"  - {path.relative_to(ROOT)}")
        print("Run: python scripts/audit_first_use_terms.py --write")
        return 1

    print(
        f"First-use terminology audit passed: {len(seen)} technical terms "
        "introduced once in course order."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
