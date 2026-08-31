#!/usr/bin/env python3
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP_SCHEMES = {"http", "https", "mailto", "tel", "data", "javascript"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
CSS_URL_RE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
CSS_IMPORT_RE = re.compile(r"@import\s+(['\"])(.*?)\1", re.IGNORECASE)
LESSON_ROUTE_RE = re.compile(r"['\"](?P<key>\d{2}\.\d+)['\"]\s*:\s*['\"](?P<path>[^'\"]+)['\"]")
CAPSTONE_ROUTE_RE = re.compile(r"route\s*:\s*['\"](?P<path>labs/[^'\"]+\.html)['\"]")
LESSON_KEY_RE = re.compile(r"\b(?P<key>\d{2}\.\d+)\b")
CAPSTONE_LABEL_RE = re.compile(r"mini\s+(?:megatron|kv\s+(?:connector|handoff))", re.IGNORECASE)
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
    (
        "gather_from_sequence_parallel_region</code><span>Forward: all-gather；Backward: reduce-scatter。",
        "sequence-parallel gather backward depends on tensor_parallel_output_grad",
    ),
    (
        "Prefill 一次建立 prompt 的 KV",
        "a logical prefill may be split across multiple scheduler execution chunks",
    ),
)

# Some current-source facts are important enough that silently losing them would
# make an otherwise well-formed page misleading. These checks are intentionally
# page-specific rather than pretending to be a general technical-prose linter.
REQUIRED_PAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "learn/07-vllm/architecture.html": (
        "vllm/v1/worker/gpu_worker.py",
        "vllm/v1/worker/gpu/model_runner.py",
        "vllm/v1/worker/gpu/block_table.py",
    ),
    "learn/07-vllm/model-runner-paged-attention.html": (
        "vllm/v1/worker/gpu_worker.py",
        "vllm/v1/worker/gpu/input_batch.py",
        "vllm/v1/worker/gpu/block_table.py",
        "vllm/v1/worker/gpu_model_runner.py",
    ),
}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[tuple[str, str]] = []
        self.ids: list[str] = []
        self.titles = 0
        self.h1s = 0
        self.has_viewport = False
        self.html_lang = False
        self.locked_items: list[tuple[str, str | None]] = []
        self.muted_next_labels: list[str] = []
        self.stack: list[dict[str, object]] = []

    def _record_tag(self, tag: str, attrs: list[tuple[str, str | None]], push: bool) -> None:
        data = dict(attrs)
        classes = set((data.get("class") or "").split())

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

        if not push:
            return

        frame: dict[str, object] = {
            "tag": tag,
            "parts": [],
            "module_block": tag == "div" and "module-block" in classes,
            "module_name": tag == "div" and "module-name" in classes,
            "capture_module": tag == "b" and any(bool(item.get("module_name")) for item in self.stack),
            "locked": tag == "span" and {"lesson-link", "locked"}.issubset(classes),
            "muted_next": {"next-lesson", "muted-next"}.issubset(classes),
        }
        if frame["module_block"]:
            frame["module_id"] = None
        self.stack.append(frame)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._record_tag(tag, attrs, tag not in VOID_TAGS)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._record_tag(tag, attrs, False)

    def handle_data(self, data: str) -> None:
        for frame in self.stack:
            if frame.get("capture_module") or frame.get("locked") or frame.get("muted_next"):
                parts = frame.get("parts")
                if isinstance(parts, list):
                    parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            return

        # Pages are expected to be well formed; search backward only to avoid a
        # malformed tag making all later navigation-state checks meaningless.
        index = next(
            (idx for idx in range(len(self.stack) - 1, -1, -1) if self.stack[idx].get("tag") == tag),
            None,
        )
        if index is None:
            return
        frame = self.stack.pop(index)
        parts = frame.get("parts")
        text = " ".join("".join(parts).split()) if isinstance(parts, list) else ""

        if frame.get("capture_module"):
            match = re.search(r"\d{2}", text)
            if match:
                for parent in reversed(self.stack):
                    if parent.get("module_block"):
                        parent["module_id"] = match.group(0)
                        break

        if frame.get("locked"):
            module_id = next(
                (parent.get("module_id") for parent in reversed(self.stack) if parent.get("module_block")),
                None,
            )
            self.locked_items.append((text, module_id if isinstance(module_id, str) else None))

        if frame.get("muted_next"):
            self.muted_next_labels.append(text)


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


def check_dynamic_routes(failures: list[str]) -> tuple[dict[str, str], int]:
    """Validate lesson/lab paths that only become hrefs at runtime in app.js."""
    app_js = ROOT / "app.js"
    if not app_js.exists():
        failures.append("app.js: missing dynamic route registry")
        return {}, 0

    text = app_js.read_text(encoding="utf-8")
    lesson_matches = list(LESSON_ROUTE_RE.finditer(text))
    capstone_routes = list(CAPSTONE_ROUTE_RE.finditer(text))
    lesson_routes = {match.group("key"): match.group("path") for match in lesson_matches}

    keys = [match.group("key") for match in lesson_matches]
    if len(keys) != len(set(keys)):
        failures.append("app.js: duplicate lessonRoutes key")

    for key, raw in lesson_routes.items():
        target = ROOT / raw
        if not target.exists():
            failures.append(f"app.js: lesson route {key} points to missing target: {raw}")

    for match in capstone_routes:
        raw = match.group("path")
        target = ROOT / raw
        if not target.exists():
            failures.append(f"app.js: capstone route points to missing target: {raw}")

    # Every published course module needs a canonical entry route so legacy
    # generic sidebar placeholders can safely resolve to the first real lesson.
    for module_dir in sorted((ROOT / "learn").glob("[0-9][0-9]-*")):
        if not module_dir.is_dir():
            continue
        module_id = module_dir.name[:2]
        entry_key = f"{module_id}.1"
        if entry_key not in lesson_routes:
            failures.append(
                f"app.js: published module {module_dir.name} has no canonical {entry_key} route"
            )

    for marker in ("firstLessonRouteByModule", "resolveLockedLessonRoute", "resolveRouteFromText"):
        if marker not in text:
            failures.append(f"app.js: missing navigation fallback helper {marker}")

    return lesson_routes, len(capstone_routes)


def check_navigation_state(
    rel: Path,
    parser: PageParser,
    lesson_routes: dict[str, str],
    failures: list[str],
) -> None:
    """Make sure old locked/muted markup can resolve to a real published destination."""
    if not rel.as_posix().startswith("learn/"):
        return

    for label, module_id in parser.locked_items:
        match = LESSON_KEY_RE.search(label)
        if match:
            key = match.group("key")
            if key not in lesson_routes:
                failures.append(f"{rel}: locked lesson {label!r} has no dynamic route for {key}")
            continue

        if CAPSTONE_LABEL_RE.search(label):
            continue

        if module_id and f"{module_id}.1" in lesson_routes:
            continue

        failures.append(
            f"{rel}: locked sidebar item {label!r} cannot resolve by lesson key, capstone, or module entry"
        )

    for label in parser.muted_next_labels:
        match = LESSON_KEY_RE.search(label)
        if match and match.group("key") in lesson_routes:
            continue
        if CAPSTONE_LABEL_RE.search(label):
            continue
        failures.append(f"{rel}: muted next-lesson {label!r} has no resolvable destination")


def main() -> int:
    html_files = sorted(ROOT.rglob("*.html"))
    css_files = sorted(ROOT.rglob("*.css"))
    lab_python_files = sorted((ROOT / "labs" / "code").glob("*.py"))
    failures: list[str] = []
    cache: dict[Path, PageParser] = {}

    lesson_routes, capstone_route_count = check_dynamic_routes(failures)

    for page in html_files:
        text = page.read_text(encoding="utf-8")
        parser = PageParser()
        parser.feed(text)
        cache[page.resolve()] = parser
        rel = page.relative_to(ROOT)
        rel_key = rel.as_posix()

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
        for marker in REQUIRED_PAGE_MARKERS.get(rel_key, ()):
            if marker not in text:
                failures.append(
                    f"{rel}: missing required current-source marker {marker!r}"
                )

        check_navigation_state(rel, parser, lesson_routes, failures)

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
        f"{len(lab_python_files)} lab scripts, {len(lesson_routes)} lesson routes, "
        f"{capstone_route_count} capstone routes; local refs/imports, reading metadata, "
        "dynamic routes, locked/muted navigation fallbacks, stale placeholders, semantic guards, "
        "current-source markers, and lab Python syntax are valid."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
