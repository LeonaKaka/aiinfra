#!/usr/bin/env python3
"""Audit learner-visible prose without forcing source-facing terms into Chinese.

Chinese should carry the explanation, but identifiers and source-facing anchors
must remain recognizable when they map to code, APIs, classes, fields or common
repository terminology. English engineering words are therefore review signals,
not automatic failures. Hard failures are reserved for malformed hybrid text
that is usually created by mechanical replacement.
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

# Context-sensitive source-facing words are useful review signals but not hard
# failures. They may be natural anchors in terms such as Transformer Block,
# ModelRunner path or KV block, while prose should still avoid gratuitous use.
REVIEW_ONLY = set(CHINESE_DEFAULT)

# Mechanical replacement artifacts are never acceptable. Keep this list small
# and concrete: it protects readability without dictating whether a legitimate
# source term should be English or Chinese in context.
MALFORMED_PATTERNS = {
    r"分块ed\b": "mixed Chinese/English suffix",
    r"规划ning\b": "mixed Chinese/English suffix",
    r"批次ing\b": "mixed Chinese/English suffix",
    r"句柄s\b": "mixed Chinese/English plural",
    r"后端s\b": "mixed Chinese/English plural",
    r"传输s\b": "mixed Chinese/English plural",
    r"块表S\b": "mechanical plural residue",
    r"槽位映射PING\b": "mechanical suffix residue",
    r"\bX\.形状\b": "translated code attribute",
    r"\bre形状\b": "broken reshape token",
    r"直接直接内存访问": "duplicated translation",
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
    review_hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    malformed_hits: dict[str, list[str]] = defaultdict(list)
    paths = list(LESSONS.glob("**/*.html"))
    paths += list(LABS.glob("*.html"))
    paths += [path for path in GLOBAL_PAGES if path.exists()]

    for path in sorted(set(paths)):
        prose = visible_prose(path)
        rel = str(path.relative_to(ROOT))

        for english in CHINESE_DEFAULT:
            count = len(re.findall(
                rf"(?<![A-Za-z0-9_]){re.escape(english)}(?![A-Za-z0-9_])",
                prose,
                re.I,
            ))
            if count:
                review_hits[rel][english] += count

        # Scan the full source too, because malformed hybrids often occur in UI
        # labels or code-adjacent prose that the visible-prose extractor skips.
        source = path.read_text(encoding="utf-8")
        for pattern, reason in MALFORMED_PATTERNS.items():
            if re.search(pattern, source, re.I):
                malformed_hits[rel].append(reason)

    if review_hits:
        total_review = sum(sum(terms.values()) for terms in review_hits.values())
        print(
            f"Language/source-anchor review: {total_review} occurrence(s) "
            f"across {len(review_hits)} page(s) (warning only)."
        )
        for rel, terms in review_hits.items():
            summary = ", ".join(
                f"{term}×{count}↔{CHINESE_DEFAULT[term]}"
                for term, count in sorted(terms.items())
            )
            print(f"review {rel}: {summary}")

    if malformed_hits:
        print(
            f"Language audit FAILED: malformed hybrid text found in "
            f"{len(malformed_hits)} page(s)."
        )
        for rel, reasons in malformed_hits.items():
            print(f"{rel}: {', '.join(sorted(set(reasons)))}")
        return 1

    print("Language audit: no malformed hybrid replacements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
