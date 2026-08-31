#!/usr/bin/env python3
"""Report acronym-like tokens in lesson articles that are not in the term registry.

This is intentionally a candidate audit, not yet a hard CI failure: the first run
is used to separate real technical abbreviations from UI labels, variables and
product names that should be explicitly allowlisted.
"""
from __future__ import annotations

import html
import re
from collections import defaultdict
from pathlib import Path

from lesson_terms import MAIN_RE, TERM_SECTION_RE, TERMS

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"

# Visual/control words and mathematical symbols are not glossary acronyms.
ALLOW = {
    "A", "B", "C", "D", "E", "H", "K", "N", "Q", "S", "V", "W", "X", "Y",
    "AI", "API", "CPU", "GPU", "RAM", "VRAM", "HBM", "GDDR", "SM", "ALU", "CUDA",
    "GEMM", "MLP", "LLM", "MHA", "GQA", "MQA", "MLA", "QKV", "KV", "GELU", "MSE",
    "SGD", "FFN", "FP32", "FP16", "BF16", "FP8", "TF32", "PCI", "PCIE", "DMA", "NUMA",
    "DP", "DDP", "TP", "PP", "SP", "CP", "EP", "FSDP", "P2P", "A2A", "AG", "RS", "NCCL",
    "NVLS", "OOM", "TTFT", "ITL", "TPOT", "TPS", "QPS", "SLO", "SLA", "APC", "LRU", "TTL",
    "RPC", "IPC", "NIC", "IB", "TCP", "RDMA", "UCX", "NIXL", "DCP", "LBHNC", "LBNHC",
    # Common UI / diagram words.
    "LESSON", "FOUNDATION", "FOUNDATIONS", "INPUT", "OUTPUT", "FORWARD", "BACKWARD", "LOSS",
    "STEP", "REPEAT", "BEFORE", "AFTER", "START", "END", "READ", "WRITE", "WAIT", "READY",
    "COMM", "COMPUTE", "MEMORY", "MODEL", "TOKEN", "TOKENS", "ATTENTION", "LINEAR", "PYTHON",
    "CURRENT", "LOCAL", "GLOBAL", "RANK", "RANKS", "WORLD", "GROUP", "GROUPS", "PROCESS", "DEVICE",
    "HOST", "NODE", "NODES", "NETWORK", "CONTROL", "DATA", "STATE", "SCHEDULER", "ENGINE", "CACHE",
    "PREFILL", "DECODE", "CONNECTOR", "ROUTER", "EXPERT", "EXPERTS", "RING", "TREE", "SOURCE",
    "TARGET", "SHARD", "SHARDS", "FULL", "PARTIAL", "SEND", "RECV", "SUM", "MAX", "NONE", "PASS",
    "FAIL", "INFO", "TRUE", "FALSE", "READING", "COURSE", "MAP", "TERMS", "CHECKPOINT", "NOTE",
    "WHAT", "WHY", "HOW", "WHO", "THE", "ONE", "TWO", "THREE", "FOUR", "FIRST", "NEXT", "VS",
    # Product/proper names that are not useful expansions.
    "NVIDIA", "PYTORCH", "MEGATRON", "VLLM", "NVLINK", "NVSWITCH", "GPUDIRECT", "ETHERNET",
    # Shell/env/code identifiers whose expansion is not the teaching need.
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "NCCL_DEBUG", "NCCL_ALGO",
}
ALLOW.update(TERMS)

# All-caps initialisms, mixed-case abbreviation shapes (RoCE/ReLU/MiB/GiB), and digit-leading forms (1F1B).
CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Z]{2,}[A-Z0-9_]*|[A-Z][a-z]{0,3}[A-Z][A-Za-z0-9]*|[0-9]+[A-Z][A-Z0-9]*)(?![A-Za-z0-9_])"
)


def article_text(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    m = MAIN_RE.search(source)
    if not m:
        return ""
    body = TERM_SECTION_RE.sub(" ", m.group("body"))
    body = re.sub(r"<script\b.*?</script>|<style\b.*?</style>|<svg\b.*?</svg>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    return html.unescape(body)


def main() -> int:
    hits: dict[str, set[str]] = defaultdict(set)
    for path in sorted(LESSONS.glob("**/*.html")):
        rel = str(path.relative_to(ROOT))
        for token in CANDIDATE_RE.findall(article_text(path)):
            if token not in ALLOW:
                hits[token].add(rel)

    if not hits:
        print("Acronym candidate audit: 0 unregistered candidates.")
        return 0

    print(f"Acronym candidate audit: {len(hits)} unregistered candidate(s).")
    for token in sorted(hits, key=str.casefold):
        pages = ", ".join(sorted(hits[token]))
        print(f"{token}: {pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
