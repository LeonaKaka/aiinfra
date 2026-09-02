#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "labs"
NAV_RE = re.compile(r'<nav\b[^>]*class=["\']site-nav["\'][^>]*>.*?</nav>', re.DOTALL)
LAB_CARD_RE = re.compile(
    r'<a\b[^>]*class=["\'][^"\']*\blab-card\b[^"\']*["\'][^>]*href=["\']\./([^"\']+\.html)["\'][^>]*>'
    r'.*?<small>\s*LAB\s+([AB])(\d+)\s*·.*?</small>\s*<h2>(.*?)</h2>',
    re.DOTALL | re.IGNORECASE,
)
SECTION_RE = re.compile(
    r'<section\b[^>]*class=["\'][^"\']*\blab-section\b[^"\']*["\'][^>]*\bid=["\']([^"\']+)["\'][^>]*>.*?<h2>(.*?)</h2>',
    re.DOTALL | re.IGNORECASE,
)
LAB_MAIN_OPEN = '<main class="lab-article">'
LAB_MAIN_WITH_NAV = '<main class="lab-article" style="width:auto;margin:0;padding:0">'
LAB_SHELL_OPEN = '<div class="lesson-shell lab-lesson-shell">'
APP_SCRIPT_RE = re.compile(r'</main>\s*(<script src="\.\./app\.js"></script>)')


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(fragment: str) -> str:
    parser = TextExtractor()
    parser.feed(fragment)
    return " ".join("".join(parser.parts).split())


def root_prefix(path: Path) -> str:
    rel = path.relative_to(ROOT)
    depth = len(rel.parts) - 1
    return "./" if depth == 0 else "../" * depth


def canonical_nav(path: Path) -> str:
    prefix = root_prefix(path)
    roadmap = "#roadmap" if path == ROOT / "index.html" else f"{prefix}index.html#roadmap"
    links = (
        ("课程地图", roadmap),
        ("Labs", f"{prefix}labs/index.html"),
        ("Source Map", f"{prefix}source-map/index.html"),
        ("Glossary", f"{prefix}glossary/index.html"),
        ("GitHub", "https://github.com/LeonaKaka/aiinfra"),
    )
    return '<nav class="site-nav" id="siteNav">' + "".join(
        f'<a href="{href}">{label}</a>' for label, href in links
    ) + '</nav>'


def wants_mobile_toggle(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    # Lesson/Lab CSS deliberately uses a horizontally scrollable compact nav on
    # mobile. Standalone reference pages use the shared dropdown button.
    return not (rel.startswith("learn/") or rel.startswith("labs/"))


def normalize_header(path: Path, text: str) -> str:
    match = NAV_RE.search(text)
    if not match:
        return text
    replacement = canonical_nav(path)
    if wants_mobile_toggle(path) and 'class="nav-toggle"' not in text:
        replacement = (
            '<button class="nav-toggle" id="navToggle" aria-expanded="false" '
            'aria-controls="siteNav">菜单</button>\n    ' + replacement
        )
    return text[: match.start()] + replacement + text[match.end() :]


def lab_catalog() -> dict[str, list[tuple[int, str, str, str]]]:
    """Read the canonical Lab order/names from the published Labs index."""
    index_text = (LAB_DIR / "index.html").read_text(encoding="utf-8")
    tracks: dict[str, list[tuple[int, str, str, str]]] = {"A": [], "B": []}
    for filename, track, number, heading in LAB_CARD_RE.findall(index_text):
        code = f"{track}{int(number)}"
        title = plain_text(heading)
        target = LAB_DIR / filename
        if not target.exists():
            raise RuntimeError(f"labs/index.html: {code} points to missing {filename}")
        tracks[track].append((int(number), code, title, filename))
    for values in tracks.values():
        values.sort(key=lambda item: item[0])
    if [item[0] for item in tracks["A"]] != list(range(1, 13)):
        raise RuntimeError("labs/index.html: expected complete Training sequence A1-A12")
    if [item[0] for item in tracks["B"]] != list(range(1, 9)):
        raise RuntimeError("labs/index.html: expected complete Inference sequence B1-B8")
    return tracks


def lab_sidebar(current: Path, tracks: dict[str, list[tuple[int, str, str, str]]]) -> str:
    chunks = [
        '<aside class="course-sidebar lab-sidebar" aria-label="Labs 目录">',
        '  <div class="side-label">LAB MAP</div>',
        '  <a class="side-home" href="./index.html">← Hands-on Labs</a>',
    ]
    for track, name in (("A", "Training"), ("B", "Inference")):
        chunks.append(f'  <div class="module-block lab-side-group"><div class="module-name lab-side-name"><b>{track}</b> {name}</div>')
        for _, code, title, filename in tracks[track]:
            classes = ["lesson-link", "lab-side-link"]
            if filename == current.name:
                classes.extend(("current", "infer-current" if track == "B" else "train-current"))
            chunks.append(
                f'    <a class="{" ".join(classes)}" href="./{filename}">{code} · {html.escape(title)}</a>'
            )
        chunks.append("  </div>")
    chunks.append("</aside>")
    return "\n".join(chunks)


def lab_toc_entries(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for section_id, heading in SECTION_RE.findall(text):
        label = plain_text(heading)
        if label:
            entries.append((section_id, label))
    if not entries:
        raise RuntimeError("lab page has no .lab-section[id] headings for ON THIS PAGE")
    return entries


def lab_toc(text: str) -> str:
    links = "".join(
        f'<a href="#{section_id}">{html.escape(label)}</a>'
        for section_id, label in lab_toc_entries(text)
    )
    return f'<aside class="toc lab-toc" aria-label="本页目录"><strong>ON THIS PAGE</strong>{links}</aside>'


def validate_existing_lab_shell(
    path: Path,
    text: str,
    tracks: dict[str, list[tuple[int, str, str, str]]],
) -> None:
    if 'class="course-sidebar lab-sidebar"' not in text or 'class="toc lab-toc"' not in text:
        raise RuntimeError(f"{path.relative_to(ROOT)}: incomplete Lab side navigation shell")
    for values in tracks.values():
        for _, _, _, filename in values:
            if f'href="./{filename}"' not in text:
                raise RuntimeError(f"{path.relative_to(ROOT)}: Lab sidebar missing {filename}")
    for section_id, _ in lab_toc_entries(text):
        if f'href="#{section_id}"' not in text:
            raise RuntimeError(f"{path.relative_to(ROOT)}: Lab TOC missing #{section_id}")
    if text.count(" lab-side-link current") != 1:
        raise RuntimeError(f"{path.relative_to(ROOT)}: expected exactly one current Lab sidebar entry")


def normalize_lab(path: Path, text: str, tracks: dict[str, list[tuple[int, str, str, str]]]) -> str:
    if path.parent != LAB_DIR or path.name == "index.html":
        return text

    if LAB_SHELL_OPEN in text:
        validate_existing_lab_shell(path, text, tracks)
        return text
    if LAB_MAIN_OPEN not in text:
        raise RuntimeError(f"{path.relative_to(ROOT)}: missing {LAB_MAIN_OPEN}")

    sidebar = lab_sidebar(path, tracks)
    toc = lab_toc(text)
    text = text.replace(
        LAB_MAIN_OPEN,
        f'{LAB_SHELL_OPEN}\n  {sidebar}\n  {LAB_MAIN_WITH_NAV}',
        1,
    )
    match = APP_SCRIPT_RE.search(text)
    if not match:
        raise RuntimeError(f"{path.relative_to(ROOT)}: cannot find Lab main closing tag before app.js")
    replacement = f'</main>\n  {toc}\n</div>\n  {match.group(1)}'
    return text[: match.start()] + replacement + text[match.end() :]


def normalize(path: Path, tracks: dict[str, list[tuple[int, str, str, str]]]) -> str:
    text = path.read_text(encoding="utf-8")
    text = normalize_header(path, text)
    text = normalize_lab(path, text, tracks)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write canonical navigation into HTML files")
    args = parser.parse_args()

    tracks = lab_catalog()
    changed: list[Path] = []
    html_files = sorted(ROOT.rglob("*.html"))
    for path in html_files:
        try:
            expected = normalize(path, tracks)
        except RuntimeError as exc:
            print(exc)
            return 1
        current = path.read_text(encoding="utf-8")
        if expected == current:
            continue
        changed.append(path)
        if args.write:
            path.write_text(expected, encoding="utf-8")

    if changed and not args.write:
        for path in changed:
            print(f"navigation drift: {path.relative_to(ROOT)}")
        return 1

    verb = "normalized" if args.write else "checked"
    print(
        f"Site navigation {verb}: {len(html_files)} HTML pages; "
        f"{len(tracks['A']) + len(tracks['B'])} Lab detail pages use canonical sidebars/TOCs."
    )
    if args.write and changed:
        print(f"Updated {len(changed)} HTML pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
