#!/usr/bin/env python3
"""Report potentially heavy first-use terminology without failing CI.

This is a reading-quality audit, not a correctness guard. It highlights only
signals that map reasonably well to visible friction:
1. too many canonical expansions packed into the lesson dek; and
2. a long first expansion placed inside compact <span> card text.

The script always returns 0. Review warnings manually before changing prose or
promoting any pattern to a hard rule.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from lesson_terms import MAIN_RE, TERM_SECTION_RE, TERMS, normalized_form

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"

DEK_TERM_WARNING = 5
LONG_COMPACT_PHRASE = 46
SKIP_TAGS = {"code", "pre", "script", "style", "svg"}
SKIP_CLASSES = {"lesson-kicker", "section-no", "breadcrumb", "mobile-course-bar", "lesson-terms", "toc"}


class ProseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, set[str]]] = []
        self.first_use: dict[str, tuple[str, tuple[tuple[str, frozenset[str]], ...]]] = {}

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

        ancestry = tuple((tag, frozenset(classes)) for tag, classes in self.stack)
        for term, (_, _, _, expand) in TERMS.items():
            if not expand or term in self.first_use:
                continue
            phrase = normalized_form(term)
            if phrase in text:
                self.first_use[term] = (phrase, ancestry)


def has_class(ancestry: tuple[tuple[str, frozenset[str]], ...], class_name: str) -> bool:
    return any(class_name in classes for _, classes in ancestry)


def inside_tag(ancestry: tuple[tuple[str, frozenset[str]], ...], tag_name: str) -> bool:
    return any(tag == tag_name for tag, _ in ancestry)


def scan(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    main = MAIN_RE.search(source)
    if not main:
        return []
    body = TERM_SECTION_RE.sub(" ", main.group("body"))
    parser = ProseParser()
    parser.feed(body)

    warnings: list[str] = []
    dek_terms = sorted(
        term
        for term, (_, ancestry) in parser.first_use.items()
        if has_class(ancestry, "dek")
    )
    if len(dek_terms) >= DEK_TERM_WARNING:
        warnings.append(
            f"dek density: {len(dek_terms)} first-use expansions "
            f"({', '.join(dek_terms)})"
        )

    compact = []
    for term, (phrase, ancestry) in sorted(parser.first_use.items()):
        if inside_tag(ancestry, "span") and len(phrase) >= LONG_COMPACT_PHRASE:
            compact.append(f"{term} ({len(phrase)} chars)")
    if compact:
        warnings.append("long compact first-use: " + ", ".join(compact))

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
