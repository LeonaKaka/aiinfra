#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"
LINK = '<link rel="stylesheet" href="../../lesson-terms.css" />'
ANCHOR_RE = re.compile(r'<link\s+rel="stylesheet"\s+href="\.\./\.\./lesson\.css"\s*/?>', re.I)

changed = 0
for path in sorted(LESSONS.glob("**/*.html")):
    text = path.read_text(encoding="utf-8")
    if '../../lesson-terms.css' in text:
        continue
    match = ANCHOR_RE.search(text)
    if not match:
        raise SystemExit(f"{path.relative_to(ROOT)}: missing lesson.css anchor")
    insertion = match.group(0) + LINK
    text = text[:match.start()] + insertion + text[match.end():]
    path.write_text(text, encoding="utf-8")
    changed += 1

print(f"Lesson terminology stylesheet linked: {changed} file(s) changed.")
