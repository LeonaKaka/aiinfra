#!/usr/bin/env python3
"""Report potentially heavy first-use terminology without failing CI.

This is a reading-quality audit, not a correctness guard. It highlights:
1. many canonical expansions packed near the beginning of a lesson; and
2. a first expansion placed inside compact UI text such as <span>/<small>.

The thresholds are deliberately conservative and the script always returns 0.
Use the report for human review before turning any pattern into a hard rule.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from lesson_terms import MAIN_RE, TERM_SECTION_RE, TERMS, normalized_form

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"

OPENING_CHAR_BUDGET = 1000
OPENING_TERM_WARNING = 6
COMPACT_TAGS = {"span", "small"}
SKIP_TAGS = {"code", "pre", "script", "style", "svg"}
SKIP_CLASSES = {"lesson-kicker", "section-no", "breadcrumb", "mobile-course-bar", "lesson-terms", "toc"}


class ProseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, set[str]]] = []
        self.visible_offset = 0
        self.first_use: dict[str, tuple[int, str, tuple[str, ...]]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes: set[str] = set()
        for key, value in attrs:
            if key == "class" and value:
                classes.update(value.split())
        self.stack.append((tag.lower(), classes))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def excluded(self) -> bool:
        for tag, classes in self.stack:
            if tag in SKIP_TAGS or classes & SKIP_CLASSES:
                return True
        return False

    def handle_data(self, data: str) -> None:
        if self.excluded():
            return
        text = " ".join(data.split())
        if not text:
            return

        tags = tuple(tag for tag, _ in self.stack)
        for term, (_, _, _, expand) in TERMS.items():
            if not expand or term in self.first_use:
                continue
            phrase = normalized_form(term)
            idx = text.find(phrase)
            if idx >= 0:
                self.first_use[term] = (self.visible_offset + idx, phrase, tags)

        # Count visible characters rather than raw HTML bytes. This treats Chinese
        # prose naturally and is stable enough for a warning-only opening window.
        self.visible_offset += len(text) + 1


def scan(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    main = MAIN_RE.search(source)
    if not main:
        return []
    body = TERM_SECTION_RE.sub(" ", main.group("body"))
    parser = ProseParser()
    parser.feed(body)

    warnings: list[str] = []
    opening = sorted(
        (
            (offset, term)
            for term, (offset, _, _) in parser.first_use.items()
            if offset < OPENING_CHAR_BUDGET
        ),
        key=lambda item: item[0],
    )
    if len(opening) >= OPENING_TERM_WARNING:
        terms = ", ".join(term for _, term in opening)
        warnings.append(
            f"opening density: {len(opening)} expansions in first "
            f"~{OPENING_CHAR_BUDGET} visible chars ({terms})"
        )

    compact = []
    for term, (_, _, tags) in sorted(parser.first_use.items()):
        compact_ancestors = [tag for tag in tags if tag in COMPACT_TAGS]
        if compact_ancestors:
            compact.append(f"{term}<{compact_ancestors[-1]}>")
    if compact:
        warnings.append("compact first-use: " + ", ".join(compact))

    return warnings


def main() -> int:
    reports: list[tuple[str, list[str]]] = []
    for path in sorted(LESSONS.glob("**/*.html")):
        warnings = scan(path)
        if warnings:
            reports.append((str(path.relative_to(ROOT)), warnings))

    if not reports:
        print("Term density audit: no reading-density warnings at current thresholds.")
        return 0

    print(f"Term density audit: {len(reports)} lesson(s) worth human review (warning only).")
    for rel, warnings in reports:
        print(rel)
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
