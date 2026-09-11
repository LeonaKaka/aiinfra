#!/usr/bin/env python3
"""Audit learner-visible prose for avoidable English engineering vocabulary.

This is intentionally warning-only while the site is being migrated module by
module. Proper names, source identifiers and code stay English; ordinary prose
should prefer Chinese. Once the backlog reaches zero, this audit can become a
hard CI guard without changing the policy.
"""
from __future__ import annotations

import html
import re
from collections import defaultdict
from pathlib import Path

from lesson_terms import MAIN_RE, TERM_SECTION_RE

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"

# These are concepts whose English spelling is useful in source code, but whose
# bare use in Chinese teaching prose usually makes a sentence harder to read.
CHINESE_DEFAULT = {
    "request": "请求",
    "requests": "请求",
    "scheduler": "调度器",
    "scheduling": "调度",
    "workload": "工作负载",
    "workloads": "工作负载",
    "buffer": "缓冲区",
    "buffers": "缓冲区",
    "metadata": "元数据",
    "shard": "分片",
    "shards": "分片",
    "replica": "副本",
    "replicas": "副本",
    "stage": "阶段",
    "stages": "阶段",
    "worker": "工作进程",
    "workers": "工作进程",
    "backend": "后端",
    "allocation": "分配",
    "ownership": "归属关系",
    "throughput": "吞吐量",
    "latency": "延迟",
    "handoff": "交接",
    "producer": "生产端",
    "consumer": "消费端",
    "descriptor": "描述符",
    "lifecycle": "生命周期",
    "lease": "租约",
    "heartbeat": "心跳",
    "layout": "布局",
    "storage": "存储",
}

# Code/source identifiers are removed before scanning. Product names and core
# source-facing concepts such as PyTorch, CUDA, token, Tensor, Transformer,
# Attention, Prefill, Decode, KV Cache and rank are deliberately not listed.
SKIP_BLOCK_RE = re.compile(
    r"<(?:script|style|pre|code|svg)\b.*?</(?:script|style|pre|code|svg)>",
    re.S | re.I,
)
TEXT_TAG_RE = re.compile(
    r"<(?:p|li|h1|h2|h3|span|b|strong|small)\b[^>]*>(.*?)</(?:p|li|h1|h2|h3|span|b|strong|small)>",
    re.S | re.I,
)


def visible_prose(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    match = MAIN_RE.search(source)
    if not match:
        return ""
    body = TERM_SECTION_RE.sub(" ", match.group("body"))
    body = SKIP_BLOCK_RE.sub(" ", body)
    chunks = []
    for item in TEXT_TAG_RE.finditer(body):
        text = re.sub(r"<[^>]+>", " ", item.group(1))
        chunks.append(html.unescape(text))
    return "\n".join(chunks)


def main() -> int:
    hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path in sorted(LESSONS.glob("**/*.html")):
        prose = visible_prose(path)
        rel = str(path.relative_to(ROOT))
        for english in CHINESE_DEFAULT:
            count = len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(english)}(?![A-Za-z0-9_])", prose, re.I))
            if count:
                hits[rel][english] += count

    if not hits:
        print("Language-mix audit: no discouraged bare English engineering terms.")
        return 0

    total = sum(sum(terms.values()) for terms in hits.values())
    print(f"Language-mix audit: {total} occurrence(s) across {len(hits)} lesson(s) (warning only during migration).")
    for rel, terms in hits.items():
        summary = ", ".join(
            f"{term}×{count}→{CHINESE_DEFAULT[term]}"
            for term, count in sorted(terms.items())
        )
        print(f"{rel}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
