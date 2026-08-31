#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"
ANCHOR = '<link rel="stylesheet" href="../../lesson.css" />'
LINK = '<link rel="stylesheet" href="../../lesson-terms.css" />'

changed = 0
for path in sorted(LESSONS.glob("**/*.html")):
    text = path.read_text(encoding="utf-8")
    if LINK in text:
        continue
    if ANCHOR not in text:
        raise SystemExit(f"{path.relative_to(ROOT)}: missing lesson.css anchor")
    text = text.replace(ANCHOR, ANCHOR + "\n  " + LINK, 1)
    path.write_text(text, encoding="utf-8")
    changed += 1

print(f"Lesson terminology stylesheet linked: {changed} file(s) changed.")
