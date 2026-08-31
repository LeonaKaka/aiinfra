#!/usr/bin/env python3
"""One-shot finalizer for the 2026-09-01 terminology audit."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def harden_audit() -> None:
    path = ROOT / "scripts" / "audit_lesson_acronyms.py"
    replace_once(
        path,
        '"""Report acronym-like tokens in lesson articles that are not in the term registry.\n\nThis is intentionally a candidate audit, not yet a hard CI failure: the first run\nis used to separate real technical abbreviations from UI labels, variables and\nproduct names that should be explicitly allowlisted.\n"""',
        '"""Fail on high-confidence acronym-like tokens missing from the term registry.\n\nThe scan is intentionally limited to teaching prose (<p>/<li>) and excludes\ncode, compact UI labels, product/class spellings and explicit narrow allowlists.\nA new unexplained acronym in learner-facing prose is therefore a CI regression.\n"""',
    )
    replace_once(
        path,
        '    for token in sorted(hits, key=str.casefold):\n        pages = ", ".join(sorted(hits[token]))\n        print(f"{token}: {pages}")\n    return 0\n',
        '    for token in sorted(hits, key=str.casefold):\n        pages = ", ".join(sorted(hits[token]))\n        print(f"{token}: {pages}")\n    return 1\n',
    )


def move_dram_expansion_into_prose() -> None:
    path = ROOT / "learn" / "08-kv-connector" / "nixl-rdma.html"
    replace_once(
        path,
        '<p>RDMA/zero-copy 风格的数据移动需要底层知道地址范围、memory type、设备、访问权限以及可用于 remote operation 的描述信息。这个准备过程可以有显著固定开销，所以长期存在的 KV cache arena 很适合<strong>一次注册、反复传输其中不同 blocks</strong>。</p><div class="memory-registration"><div><small>STARTUP / INIT</small><b>register KV cache regions</b><span>告诉 data-movement layer：这些 VRAM/Dynamic Random-Access Memory (DRAM，动态随机存取存储器) regions 是可传输资源。</span></div>',
        '<p>RDMA/zero-copy 风格的数据移动需要底层知道地址范围、memory type、设备、访问权限以及可用于 remote operation 的描述信息。这个准备过程可以有显著固定开销，所以长期存在的 KV cache arena 很适合<strong>一次注册、反复传输其中不同 blocks</strong>。这里的 host-side backing memory 常见就是 DRAM；它和 GPU 侧 device memory 是不同层级，注册时都需要被底层正确描述。</p><div class="memory-registration"><div><small>STARTUP / INIT</small><b>register KV cache regions</b><span>告诉 data-movement layer：这些 VRAM/DRAM regions 是可传输资源。</span></div>',
    )


def main() -> None:
    harden_audit()
    move_dram_expansion_into_prose()
    print("Finalized acronym guard and DRAM prose placement.")


if __name__ == "__main__":
    main()
