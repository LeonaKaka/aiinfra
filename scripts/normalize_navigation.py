#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "app.js"
LEARN = ROOT / "learn"

LESSON_ROUTE_RE = re.compile(
    r"['\"](?P<key>\d{2}\.\d+)['\"]\s*:\s*['\"](?P<path>[^'\"]+)['\"]"
)
LESSON_KEY_RE = re.compile(r"(?<!\d)(?P<key>\d{2}\.\d+)(?!\d)")
MODULE_RE = re.compile(r'<div class="module-name"><b>(\d{2})</b>')
LOCKED_RE = re.compile(r'<span class="lesson-link locked">(?P<label>.*?)</span>', re.DOTALL)
MUTED_NEXT_RE = re.compile(
    r'<a class="next-lesson muted-next" href="[^"]*">(?P<body>.*?)</a>', re.DOTALL
)
TAG_RE = re.compile(r"<[^>]+>")

CAPSTONES = (
    (re.compile(r"mini\s+megatron", re.IGNORECASE), "labs/mini-megatron.html"),
    (
        re.compile(r"mini\s+kv\s+(?:connector|handoff)", re.IGNORECASE),
        "labs/mini-kv-handoff.html",
    ),
)


def clean_text(fragment: str) -> str:
    return " ".join(html.unescape(TAG_RE.sub(" ", fragment)).split())


def lesson_routes() -> dict[str, str]:
    text = APP_JS.read_text(encoding="utf-8")
    routes = {
        match.group("key"): match.group("path")
        for match in LESSON_ROUTE_RE.finditer(text)
    }
    if not routes:
        raise SystemExit("app.js: no lessonRoutes entries found")
    return routes


def relative_href(page: Path, route: str) -> str:
    target = ROOT / route
    if not target.exists():
        raise SystemExit(f"{page.relative_to(ROOT)}: route target does not exist: {route}")
    rel = os.path.relpath(target, page.parent).replace(os.sep, "/")
    if "/" not in rel and not rel.startswith("."):
        rel = f"./{rel}"
    return rel


def route_from_label(
    label: str,
    routes: dict[str, str],
    module_id: str | None = None,
    allow_module_fallback: bool = True,
) -> str | None:
    text = clean_text(label)
    match = LESSON_KEY_RE.search(text)
    if match and match.group("key") in routes:
        return routes[match.group("key")]

    for pattern, route in CAPSTONES:
        if pattern.search(text):
            return route

    if allow_module_fallback and module_id:
        return routes.get(f"{module_id}.1")
    return None


def normalize_page(page: Path, routes: dict[str, str]) -> tuple[str, int, list[str]]:
    original = page.read_text(encoding="utf-8")
    unresolved: list[str] = []
    edits = 0

    def replace_locked(match: re.Match[str]) -> str:
        nonlocal edits
        prefix = original[: match.start()]
        module_matches = list(MODULE_RE.finditer(prefix))
        module_id = module_matches[-1].group(1) if module_matches else None
        label = match.group("label")
        route = route_from_label(label, routes, module_id=module_id)
        if not route:
            unresolved.append(f"locked: {clean_text(label)!r}")
            return match.group(0)
        edits += 1
        return f'<a class="lesson-link" href="{relative_href(page, route)}">{label}</a>'

    text = LOCKED_RE.sub(replace_locked, original)

    def replace_muted_next(match: re.Match[str]) -> str:
        nonlocal edits
        body = match.group("body")
        route = route_from_label(body, routes, allow_module_fallback=False)
        if not route:
            unresolved.append(f"muted-next: {clean_text(body)!r}")
            return match.group(0)
        edits += 1
        return f'<a class="next-lesson" href="{relative_href(page, route)}">{body}</a>'

    text = MUTED_NEXT_RE.sub(replace_muted_next, text)
    return text, edits, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize published lesson navigation so static HTML does not depend on app.js to unlock links."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite stale navigation in place; without this flag the script is a CI check",
    )
    args = parser.parse_args()

    routes = lesson_routes()
    pages = sorted(LEARN.rglob("*.html"))
    failures: list[str] = []
    changed_pages = 0
    total_edits = 0

    for page in pages:
        original = page.read_text(encoding="utf-8")
        normalized, edits, unresolved = normalize_page(page, routes)
        rel = page.relative_to(ROOT)

        if unresolved:
            failures.extend(f"{rel}: {item}" for item in unresolved)

        if normalized != original:
            if args.write:
                page.write_text(normalized, encoding="utf-8")
                changed_pages += 1
                total_edits += edits
            else:
                failures.append(
                    f"{rel}: static navigation is stale ({edits} published locked/muted link(s)); run scripts/normalize_navigation.py --write"
                )

    if failures:
        print("STATIC NAVIGATION CHECK FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    mode = "normalized" if args.write else "checked"
    print(f"Static navigation {mode}: {len(pages)} lesson pages; {changed_pages} page(s), {total_edits} edit(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
