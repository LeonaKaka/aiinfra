#!/usr/bin/env python3
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP_SCHEMES = {"http", "https", "mailto", "tel", "data", "javascript"}
CSS_URL_RE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
CSS_IMPORT_RE = re.compile(r"@import\s+(['\"])(.*?)\1", re.IGNORECASE)
LESSON_ROUTE_RE = re.compile(r"['\"](?P<key>\d{2}\.\d+)['\"]\s*:\s*['\"](?P<path>[^'\"]+)['\"]")
CAPSTONE_ROUTE_RE = re.compile(r"route\s*:\s*['\"](?P<path>labs/[^'\"]+\.html)['\"]")
STALE_PLACEHOLDERS = (
    "课程正文即将加入",
    "lesson coming soon",
)
# These are deliberately narrow strings that previously encoded a misleading
# teaching shortcut. We do not try to lint technical prose in general; these
# guards only stop known regressions from silently reappearing.
KNOWN_SEMANTIC_REGRESSIONS = (
    ("<b>HBM / VRAM</b>", "do not use HBM as a synonym for generic GPU/VRAM memory"),
    ("<small>ITL / TPOT</small>", "ITL and TPOT are related but distinct serving metrics"),
    ("处理 prompt 的全部 S tokens", "prefill is a logical phase and may be chunked by the scheduler"),
)


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[tuple[str, str]] = []
        self.ids: list[str] = []
        self.titles = 0
        self.h1s = 0
        self.has_viewport = False
        self.html_lang = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = dict(attrs)
        if tag == "html" and data.get("lang"):
            self.html_lang = True
        if tag == "meta" and data.get("name", "").lower() == "viewport":
            self.has_viewport = True
        if tag == "title":
            self.titles += 1
        if tag == "h1":
            self.h1s += 1
        if data.get("id"):
            self.ids.append(data["id"])
        for attr in ("href", "src"):
            if data.get(attr):
                self.refs.append((attr, data[attr]))


def parse_page(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def resolve_local(page: Path, raw: str) -> tuple[Path | None, str | None]:
    parts = urlsplit(raw)
    if parts.scheme in SKIP_SCHEMES or raw.startswith("//"):
        return None, None
    fragment = parts.fragment or None
    target_text = unquote(parts.path)
    if not target_text:
        return page, fragment
    if target_text.startswith("/"):
        target = ROOT / target_text.lstrip("/")
    else:
        target = (page.parent / target_text).resolve()
    try:
        target.relative_to(ROOT)
    except ValueError:
        return target, fragment
    if target.is_dir():
        target = target / "index.html"
    return target, fragment


def css_refs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    refs = [match.group(2).strip() for match in CSS_URL_RE.finditer(text)]
    refs.extend(match.group(2).strip() for match in CSS_IMPORT_RE.finditer(text))
    return list(dict.fromkeys(refs))


def check_dynamic_routes(failures: list[str]) -> tuple[int, int]:
    """Validate lesson/lab paths that only become hrefs at runtime in app.js."""
    app_js = ROOT / "app.js"
    if not app_js.exists():
        failures.append("app.js: missing dynamic route registry")
        return 0, 0

    text = app_js.read_text(encoding="utf-8")
    lesson_routes = list(LESSON_ROUTE_RE.finditer(text))
    capstone_routes = list(CAPSTONE_ROUTE_RE.finditer(text))

    keys = [match.group("key") for match in lesson_routes]
    if len(keys) != len(set(keys)):
        failures.append("app.js: duplicate lessonRoutes key")

    for match in lesson_routes:
        key = match.group("key")
        raw = match.group("path")
        target = ROOT / raw
        if not target.exists():
            failures.append(f"app.js: lesson route {key} points to missing target: {raw}")

    for match in capstone_routes:
        raw = match.group("path")
        target = ROOT / raw
        if not target.exists():
            failures.append(f"app.js: capstone route points to missing target: {raw}")

    return len(lesson_routes), len(capstone_routes)


def main() -> int:
    html_files = sorted(ROOT.rglob("*.html"))
    css_files = sorted(ROOT.rglob("*.css"))
    lab_python_files = sorted((ROOT / "labs" / "code").glob("*.py"))
    failures: list[str] = []
    cache: dict[Path, PageParser] = {}

    for page in html_files:
        text = page.read_text(encoding="utf-8")
        parser = PageParser()
        parser.feed(text)
        cache[page.resolve()] = parser
        rel = page.relative_to(ROOT)

        if not parser.html_lang:
            failures.append(f"{rel}: missing html lang")
        if not parser.has_viewport:
            failures.append(f"{rel}: missing viewport meta")
        if parser.titles != 1:
            failures.append(f"{rel}: expected exactly one <title>, found {parser.titles}")
        if parser.h1s != 1:
            failures.append(f"{rel}: expected exactly one <h1>, found {parser.h1s}")
        if len(parser.ids) != len(set(parser.ids)):
            failures.append(f"{rel}: duplicate id attribute")

        lowered = text.lower()
        for placeholder in STALE_PLACEHOLDERS:
            if placeholder.lower() in lowered:
                failures.append(f"{rel}: stale placeholder text remains: {placeholder!r}")
        for needle, reason in KNOWN_SEMANTIC_REGRESSIONS:
            if needle in text:
                failures.append(f"{rel}: known semantic regression {needle!r}: {reason}")

        for attr, raw in parser.refs:
            target, fragment = resolve_local(page.resolve(), raw)
            if target is None:
                continue
            try:
                target.relative_to(ROOT)
            except ValueError:
                failures.append(f"{rel}: {attr} escapes site root: {raw}")
                continue
            if not target.exists():
                failures.append(f"{rel}: missing local target: {raw}")
                continue
            if fragment and target.suffix.lower() == ".html":
                target_parser = cache.get(target)
                if target_parser is None:
                    target_parser = parse_page(target)
                    cache[target] = target_parser
                if fragment not in set(target_parser.ids):
                    failures.append(
                        f"{rel}: missing fragment #{fragment} in {target.relative_to(ROOT)}"
                    )

    # CSS imports and url(...) references are easy to break when lesson-specific
    # stylesheets are split. Validate local dependencies as part of every push.
    for css in css_files:
        rel = css.relative_to(ROOT)
        for raw in css_refs(css):
            if not raw or raw.startswith("#"):
                continue
            target, _ = resolve_local(css.resolve(), raw)
            if target is None:
                continue
            try:
                target.relative_to(ROOT)
            except ValueError:
                failures.append(f"{rel}: CSS reference escapes site root: {raw}")
                continue
            if not target.exists():
                failures.append(f"{rel}: missing local CSS asset/import: {raw}")

    lesson_route_count, capstone_route_count = check_dynamic_routes(failures)

    # The lab scripts depend on PyTorch at runtime, which is intentionally not
    # installed by this lightweight site workflow. Still compile every script so
    # syntax regressions never reach the published learning site unnoticed.
    for script in lab_python_files:
        rel = script.relative_to(ROOT)
        try:
            compile(script.read_text(encoding="utf-8"), str(rel), "exec")
        except SyntaxError as exc:
            failures.append(
                f"{rel}: Python syntax error at line {exc.lineno}: {exc.msg}"
            )

    if failures:
        print("Site checks failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print(
        f"Site checks passed: {len(html_files)} HTML pages, {len(css_files)} CSS files, "
        f"{len(lab_python_files)} lab scripts, {lesson_route_count} lesson routes, "
        f"{capstone_route_count} capstone routes; local refs/imports, reading metadata, "
        "dynamic routes, stale placeholders, known semantic regressions, and lab Python syntax "
        "are valid."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
