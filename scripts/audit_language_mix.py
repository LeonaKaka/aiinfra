#!/usr/bin/env python3
"""Audit learner-visible prose for avoidable English engineering vocabulary.

Proper names, source identifiers and code stay English; ordinary teaching prose
should prefer Chinese. Glossary term cards are an explicit bilingual lookup
surface and are excluded, while their navigation and explanatory framing remain
audited. Any discouraged bare engineering term outside these intentional
surfaces is a CI failure.
"""
from __future__ import annotations

import html
import re
from collections import defaultdict
from pathlib import Path

from lesson_terms import MAIN_RE, TERM_SECTION_RE

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"
LABS = ROOT / "labs"
GLOBAL_PAGES = (
    ROOT / "index.html",
    ROOT / "source-map" / "index.html",
    ROOT / "glossary" / "index.html",
)
BODY_RE = re.compile(r"<body\b[^>]*>(?P<body>.*?)</body>", re.S | re.I)

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
    "compute": "计算",
    "transfer": "传输",
    "memory": "内存/显存",
    "queue": "队列",
    "failure": "失败",
    "recovery": "恢复",
    "preemption": "抢占",
    "routing": "路由",
    "router": "路由器",
    "dispatcher": "分发器",
    "iteration": "执行轮次",
    "iterations": "执行轮次",
    "batch": "批次",
    "phase": "阶段",
    "state": "状态",
    "states": "状态",
    "step": "步骤",
    "steps": "步骤",
    "engine": "引擎",
    "engines": "引擎",
    "policy": "策略",
    "capacity": "容量",
    "pressure": "压力",
    "planning": "规划",
    "bookkeeping": "维护开销",
    "identity": "标识",
    "info": "信息",
    "compatibility": "兼容性",
    "layer": "层",
    "layers": "层",
    "host": "主机",
    "network": "网络",
    "link": "链路",
    "ratio": "比例",
    "overhead": "开销",
    "placement": "放置策略",
    "feature": "功能",
    "features": "功能",
    "server": "服务器",
    "node": "节点",
    "nodes": "节点",
    "bytes": "字节数",
    "path": "路径",
    "pool": "池",
    "region": "区域",
    "regions": "区域",
    "block": "块",
    "blocks": "块",
}

# Code/source identifiers are removed before scanning. Product names and core
# source-facing concepts such as PyTorch, CUDA, token, Tensor, Transformer,
# Attention, Prefill, Decode, KV Cache and rank are deliberately not listed.
SKIP_BLOCK_RE = re.compile(
    r"<(?:script|style|pre|code|svg)\b.*?</(?:script|style|pre|code|svg)>",
    re.S | re.I,
)
TEXT_TAG_RE = re.compile(
    r"<(?:p|li|h1|h2|h3|span|b|strong|small|a)\b[^>]*>(.*?)</(?:p|li|h1|h2|h3|span|b|strong|small|a)>",
    re.S | re.I,
)


def visible_prose(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    match = MAIN_RE.search(source)
    if match:
        body = match.group("body")
    else:
        body_match = BODY_RE.search(source)
        body = body_match.group("body") if body_match else source
    body = TERM_SECTION_RE.sub(" ", body)
    body = SKIP_BLOCK_RE.sub(" ", body)
    # Glossary/source tables intentionally keep English lookup anchors; prose/UI
    # around them is the readability target.
    body = re.sub(r"<table\b.*?</table>", " ", body, flags=re.S | re.I)
    if path == ROOT / "glossary" / "index.html":
        body = re.sub(r'<div class="term">.*?</div>', " ", body, flags=re.S | re.I)
    chunks = []
    for item in TEXT_TAG_RE.finditer(body):
        text = re.sub(r"<[^>]+>", " ", item.group(1))
        chunks.append(html.unescape(text))
    return "\n".join(chunks)


def main() -> int:
    hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    paths = list(LESSONS.glob("**/*.html"))
    paths += list(LABS.glob("*.html"))
    paths += [path for path in GLOBAL_PAGES if path.exists()]
    for path in sorted(set(paths)):
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
    print(f"Language-mix audit FAILED: {total} occurrence(s) across {len(hits)} page(s).")
    for rel, terms in hits.items():
        summary = ", ".join(
            f"{term}×{count}→{CHINESE_DEFAULT[term]}"
            for term, count in sorted(terms.items())
        )
        print(f"{rel}: {summary}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
