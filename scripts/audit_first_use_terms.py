#!/usr/bin/env python3
"""Normalize and strictly audit first pedagogical use of technical English.

Course rule:
    shape（形状：表示 tensor 各维度的长度）

The first substantive teaching occurrence keeps the English/source spelling and
puts both the Chinese term and one concise explanation in parentheses. Later
uses stay in English without repeating that parenthesis. Headings, navigation, preformatted code, SVGs and lesson term
tables do not consume first use. Inline code such as <code>dtype</code> may be
the first teaching occurrence; its explanation is appended outside the code tag.

Course order is explicit (01.1 -> 08.5), not filename alphabetical order.
Labs are scanned after lessons and inherit terms already introduced by lessons.

Run:
  python scripts/audit_first_use_terms.py --write
  python scripts/audit_first_use_terms.py
"""
from __future__ import annotations

import argparse
import difflib
import re
from functools import lru_cache
from pathlib import Path

from first_use_terms import SOURCE_TERMS
from lesson_terms import TERMS, TERM_ALIASES, MAIN_RE, TERM_SECTION_RE

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "learn"
LABS = ROOT / "labs"

COURSE_ORDER = [
    "learn/01-foundations/tensor.html",
    "learn/01-foundations/linear.html",
    "learn/01-foundations/training-loop.html",
    "learn/01-foundations/autograd-optimizer.html",
    "learn/02-transformer/attention.html",
    "learn/02-transformer/mha-gqa.html",
    "learn/02-transformer/transformer-block.html",
    "learn/03-gpu-systems/gpu-mental-model.html",
    "learn/03-gpu-systems/gpu-memory.html",
    "learn/03-gpu-systems/gpu-bottlenecks.html",
    "learn/04-distributed/process-rank.html",
    "learn/04-distributed/collectives.html",
    "learn/04-distributed/nccl-topology.html",
    "learn/05-megatron/why-model-parallel.html",
    "learn/05-megatron/tensor-parallel.html",
    "learn/05-megatron/sequence-parallel.html",
    "learn/05-megatron/pipeline-parallel.html",
    "learn/05-megatron/distributed-optimizer.html",
    "learn/05-megatron/context-parallel.html",
    "learn/05-megatron/expert-parallel.html",
    "learn/05-megatron/communication-overlap.html",
    "learn/06-llm-inference/autoregressive-generation.html",
    "learn/06-llm-inference/prefill-decode.html",
    "learn/06-llm-inference/kv-cache.html",
    "learn/06-llm-inference/inference-performance.html",
    "learn/07-vllm/architecture.html",
    "learn/07-vllm/scheduler-continuous-batching.html",
    "learn/07-vllm/kv-cache-manager.html",
    "learn/07-vllm/model-runner-paged-attention.html",
    "learn/07-vllm/prefix-cache-preemption.html",
    "learn/08-kv-connector/why-move-kv.html",
    "learn/08-kv-connector/connector-architecture.html",
    "learn/08-kv-connector/transfer-lifecycle.html",
    "learn/08-kv-connector/nixl-rdma.html",
    "learn/08-kv-connector/production-pd.html",
]

# Tags whose contents are not prose introductions. Inline <code> is deliberately
# NOT here; exact one-word/one-term inline code can be introduced and explained.
BLOCK_TAGS = {"pre", "script", "style", "svg", "table", "h1", "h2", "h3", "nav", "aside"}
SKIP_CLASSES = {
    "lesson-kicker", "section-no", "breadcrumb", "mobile-course-bar",
    "lesson-terms", "toc", "next-lesson",
}
TAG_RE = re.compile(r"(<[^>]+>)")
OPEN_TAG_RE = re.compile(r"<\s*([A-Za-z0-9]+)\b([^>]*)>")
CLOSE_TAG_RE = re.compile(r"</\s*([A-Za-z0-9]+)\s*>")
CLASS_RE = re.compile(r'class\s*=\s*["\']([^"\']*)["\']', re.I)

SOURCE_ALIASES = {
    "tensor": ("tensor", "tensors"),
    "memory": ("memory",),
    "shape": ("shape", "shapes"),
    "dtype": ("dtype", "dtypes"),
    "device": ("device", "devices"),
    "view": ("view", "views"),
    "transpose": ("transpose",),
    "copy": ("copy",),
    "projection": ("projection", "projections"),
    "attention score": ("attention score", "attention scores"),
    "attention head": ("attention head", "attention heads"),
    "Tensor Core": ("Tensor Core", "Tensor Cores"),
    "kernel": ("kernel", "kernels"),
    "stream": ("stream", "streams"),
    "process": ("process", "processes"),
    "rank": ("rank", "ranks"),
    "world size": ("world size", "world_size"),
    "process group": ("process group", "process groups"),
    "collective": ("collective", "collectives"),
    "all-reduce": ("all-reduce", "all_reduce"),
    "all-gather": ("all-gather", "all_gather"),
    "reduce-scatter": ("reduce-scatter", "reduce_scatter"),
    "communicator": ("communicator", "communicators"),
    "shard": ("shard", "shards", "sharded"),
    "replica": ("replica", "replicas", "replicated"),
    "pipeline stage": ("pipeline stage", "pipeline stages"),
    "microbatch": ("microbatch", "microbatches"),
    "bucket": ("bucket", "buckets"),
    "expert": ("expert", "experts"),
    "router": ("router", "routers"),
    "dispatcher": ("dispatcher", "dispatchers"),
    "assignment": ("assignment", "assignments"),
    "request": ("request", "requests"),
    "scheduler": ("scheduler", "Scheduler"),
    "iteration": ("iteration", "iterations"),
    "batch": ("batch", "batches"),
    "engine": ("engine", "engines"),
    "workload": ("workload", "workloads"),
    "benchmark": ("benchmark", "benchmarks"),
    "worker": ("worker", "workers"),
    "executor": ("executor", "executors"),
    "block": ("block", "blocks"),
    "block table": ("block table", "block tables"),
    "slot mapping": ("slot mapping", "slot mappings"),
    "eviction": ("eviction", "evict"),
    "producer": ("producer", "producers"),
    "consumer": ("consumer", "consumers"),
    "transfer": ("transfer", "transfers"),
    "transport": ("transport", "transports"),
    "buffer": ("buffer", "buffers"),
    "region": ("region", "regions"),
    "descriptor": ("descriptor", "descriptors"),
    "lease": ("lease", "leases"),
    "heartbeat": ("heartbeat", "heartbeats"),
    "completion": ("completion", "completions"),
    "peer": ("peer", "peers"),
    "parameter": ("parameter", "parameters"),
    "gradient": ("gradient", "gradients"),
    "activation": ("activation", "activations"),
    "optimizer": ("optimizer", "optimizers"),
    "optimizer state": ("optimizer state", "optimizer states"),
}


def short_hint(meaning: str) -> str:
    return re.split(r"[；。]", meaning, maxsplit=1)[0].strip()


def clean_chinese(abbr: str, chinese: str) -> str:
    prefix = abbr + " "
    return chinese[len(prefix):].strip() if chinese.startswith(prefix) else chinese


@lru_cache(maxsize=None)
def boundary_pattern(alias: str, *, case_sensitive: bool = True) -> re.Pattern[str]:
    # Avoid rewriting inside identifiers such as request-level or gpu_worker.
    flags = 0 if case_sensitive else re.I
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_-])",
        flags,
    )


def source_case_sensitive(term: str) -> bool:
    # Product/model/API names with capitals are exact; lower-case engineering
    # vocabulary is matched case-insensitively.
    return any(ch.isupper() for ch in term)


def source_specs():
    out = []
    for term, (chinese, hint) in SOURCE_TERMS.items():
        out.append({
            "key": f"source:{term}",
            "aliases": SOURCE_ALIASES.get(term, (term,)),
            "chinese": chinese,
            "hint": hint,
            "case_sensitive": source_case_sensitive(term),
            "kind": "source",
        })
    return out


def acronym_specs():
    out = []
    for abbr, (english, chinese, meaning, _) in TERMS.items():
        out.append({
            "key": f"abbr:{abbr}",
            "abbr": abbr,
            "english": english,
            "aliases": TERM_ALIASES.get(abbr, (abbr,)),
            "chinese": clean_chinese(abbr, chinese),
            "hint": short_hint(meaning),
            "case_sensitive": True,
            "kind": "abbr",
        })
    return out


SPECS = source_specs() + acronym_specs()


def canonical_annotation(spec: dict) -> str:
    return f"（{spec['chinese']}：{spec['hint']}）"


def matching_paren_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != "（":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "（":
            depth += 1
        elif text[i] == "）":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def terminology_paren(text: str, end: int, spec: dict) -> tuple[str, int] | None:
    """Classify immediate Chinese parentheses as valid/simple/malformed term note."""
    close = matching_paren_end(text, end)
    if close is None:
        return None
    inside = text[end + 1:close - 1].strip()
    chinese = spec["chinese"]

    for sep in ("：", "，"):
        prefix = chinese + sep
        if inside.startswith(prefix) and len(inside[len(prefix):].strip()) >= 4:
            return "valid", close

    if inside == chinese:
        return "simple", close

    # Migration guard for nested/legacy term annotations, e.g.
    # BF16（BF16（16 位浮点：...） 16 位浮点：...）
    if chinese in inside and len(inside) <= 220:
        return "malformed", close

    return None


def legacy_acronym_matches(text: str, spec: dict, cursor: int):
    abbr = spec["abbr"]
    chinese = spec["chinese"]
    english = spec["english"]
    patterns = [
        # Chinese-first: 图形处理器（GPU）
        re.compile(rf"{re.escape(chinese)}（{re.escape(abbr)}）"),
        # Old generated English-first with Chinese in the same parentheses.
        re.compile(
            rf"{re.escape(english)}\s*\({re.escape(abbr)}[，,]\s*{re.escape(chinese)}\)",
            re.I,
        ),
        # English full name + abbreviation only.
        re.compile(rf"{re.escape(english)}\s*\(\s*{re.escape(abbr)}\s*\)", re.I),
    ]
    matches = []
    for pat in patterns:
        m = pat.search(text, cursor)
        if m:
            matches.append(m)
    return matches


def raw_matches(text: str, spec: dict, cursor: int):
    matches = []
    for alias in spec["aliases"]:
        m = boundary_pattern(alias, case_sensitive=spec["case_sensitive"]).search(text, cursor)
        if m:
            matches.append(m)
    return matches


def next_action(text: str, cursor: int, seen: set[str]):
    """Return earliest introduction or redundant-annotation cleanup action."""
    best = None

    for spec in SPECS:
        key = spec["key"]

        if key not in seen:
            for m in raw_matches(text, spec, cursor):
                candidate = (m.start(), -len(m.group(0)), "introduce", m.end(), spec, m.group(0))
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            if spec["kind"] == "abbr":
                for m in legacy_acronym_matches(text, spec, cursor):
                    candidate = (m.start(), -(m.end() - m.start()), "legacy-introduce", m.end(), spec, spec["abbr"])
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
            continue

        # Term already introduced: only act when a later occurrence still repeats
        # a terminology parenthesis or an old generated expansion.
        for m in raw_matches(text, spec, cursor):
            note = terminology_paren(text, m.end(), spec)
            if note is None:
                continue
            _, close = note
            candidate = (m.start(), -len(m.group(0)), "strip", close, spec, m.group(0))
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        if spec["kind"] == "abbr":
            for m in legacy_acronym_matches(text, spec, cursor):
                candidate = (m.start(), -(m.end() - m.start()), "legacy-strip", m.end(), spec, spec["abbr"])
                if best is None or candidate[:2] < best[:2]:
                    best = candidate

    return best


def normalize_text(text: str, seen: set[str]) -> str:
    cursor = 0
    out: list[str] = []

    while cursor < len(text):
        action = next_action(text, cursor, seen)
        if action is None:
            out.append(text[cursor:])
            break

        start, _, mode, end, spec, lexeme = action
        out.append(text[cursor:start])

        if mode in {"legacy-introduce", "legacy-strip"}:
            out.append(spec["abbr"])
            if mode == "legacy-introduce":
                out.append(canonical_annotation(spec))
                seen.add(spec["key"])
            cursor = end
            continue

        if mode == "strip":
            out.append(lexeme)
            cursor = end
            continue

        # First raw occurrence.
        out.append(lexeme)
        note = terminology_paren(text, end, spec)
        if note is None:
            out.append(canonical_annotation(spec))
            cursor = end
        else:
            kind, close = note
            if kind == "valid":
                out.append(text[end:close])
            else:
                out.append(canonical_annotation(spec))
            cursor = close
        seen.add(spec["key"])

    return "".join(out)


def lex_body(body: str):
    """Return HTML tokens with blocked/code context."""
    tokens = []
    stack: list[tuple[str, bool]] = []

    for part in TAG_RE.split(body):
        if not part:
            continue

        if not part.startswith("<"):
            tokens.append({
                "kind": "text",
                "raw": part,
                "blocked": any(blocked for _, blocked in stack),
                "in_code": any(tag == "code" for tag, _ in stack),
            })
            continue

        close = CLOSE_TAG_RE.match(part)
        if close:
            tag = close.group(1).lower()
            tokens.append({
                "kind": "tag",
                "raw": part,
                "tag": tag,
                "close": True,
                "blocked": any(blocked for _, blocked in stack),
            })
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == tag:
                    del stack[i:]
                    break
            continue

        op = OPEN_TAG_RE.match(part)
        if op and not part.rstrip().endswith("/>") and not part.startswith("<!--"):
            tag = op.group(1).lower()
            classes: set[str] = set()
            m = CLASS_RE.search(op.group(2))
            if m:
                classes = set(m.group(1).split())
            blocked_here = tag in BLOCK_TAGS or bool(classes & SKIP_CLASSES)
            tokens.append({
                "kind": "tag",
                "raw": part,
                "tag": tag,
                "close": False,
                "blocked": any(blocked for _, blocked in stack) or blocked_here,
            })
            stack.append((tag, blocked_here))
        else:
            tokens.append({
                "kind": "tag",
                "raw": part,
                "tag": None,
                "close": False,
                "blocked": any(blocked for _, blocked in stack),
            })

    return tokens


def exact_inline_code_spec(text: str):
    candidate = text.strip()
    if not candidate:
        return None
    matches = []
    for spec in SPECS:
        for alias in spec["aliases"]:
            flags = 0 if spec["case_sensitive"] else re.I
            if re.fullmatch(re.escape(alias), candidate, flags):
                matches.append((len(alias), spec))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def consume_annotation_prefix(text: str, spec: dict):
    """Return (prefix_to_emit, remainder, has_term_note)."""
    m = re.match(r"(\s*)", text)
    lead = m.group(1)
    start = len(lead)
    note = terminology_paren(text, start, spec)
    if note is None:
        return lead, text[start:], False
    kind, close = note
    if kind == "valid":
        return lead + text[start:close], text[close:], True
    # simple/malformed is replaced by canonical by caller
    return lead, text[close:], True


def normalize_body(body: str, seen: set[str]) -> str:
    tokens = lex_body(body)
    out: list[str] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        # Handle simple inline <code>term</code> without changing code contents.
        if (
            tok["kind"] == "tag"
            and tok.get("tag") == "code"
            and not tok.get("close")
            and not tok["blocked"]
            and i + 2 < len(tokens)
            and tokens[i + 1]["kind"] == "text"
            and tokens[i + 2]["kind"] == "tag"
            and tokens[i + 2].get("tag") == "code"
            and tokens[i + 2].get("close")
        ):
            code_text = tokens[i + 1]["raw"]
            spec = exact_inline_code_spec(code_text)
            out.extend([tok["raw"], code_text, tokens[i + 2]["raw"]])

            if spec is not None:
                # An authored parenthesis after </code> is represented by the
                # immediately following text token.
                j = i + 3
                following = tokens[j]["raw"] if j < len(tokens) and tokens[j]["kind"] == "text" else ""
                emitted, remainder, has_note = consume_annotation_prefix(following, spec)

                if spec["key"] not in seen:
                    if has_note and spec["chinese"] in emitted:
                        # Preserve the canonical Chinese-name parenthesis;
                        # replace legacy/malformed forms with it.
                        note = terminology_paren(emitted, len(re.match(r"(\s*)", emitted).group(1)), spec)
                        if note and note[0] == "valid":
                            out.append(emitted)
                        else:
                            out.append(re.match(r"(\s*)", emitted).group(1) + canonical_annotation(spec))
                    else:
                        out.append(canonical_annotation(spec))
                        if emitted:
                            out.append(emitted)
                    seen.add(spec["key"])
                else:
                    # Later duplicate explanation is removed.
                    if emitted and not has_note:
                        out.append(emitted)

                if j < len(tokens) and tokens[j]["kind"] == "text":
                    tokens[j]["raw"] = remainder

            i += 3
            continue

        if tok["kind"] == "text":
            if tok["blocked"] or tok["in_code"]:
                out.append(tok["raw"])
            else:
                out.append(normalize_text(tok["raw"], seen))
        else:
            out.append(tok["raw"])
        i += 1

    return "".join(out)


def normalize_page(source: str, seen: set[str]) -> str:
    m = MAIN_RE.search(source)
    if not m:
        return source

    body = m.group("body")
    table = TERM_SECTION_RE.search(body)
    if table:
        before = body[:table.start()]
        term_table = table.group(0)
        after = body[table.end():]
        body = normalize_body(before, seen) + term_table + normalize_body(after, seen)
    else:
        body = normalize_body(body, seen)

    return source[:m.start()] + m.group("open") + body + m.group("close") + source[m.end():]


def teaching_pages() -> list[Path]:
    pages = []
    missing = []
    for rel in COURSE_ORDER:
        path = ROOT / rel
        if path.exists():
            pages.append(path)
        else:
            missing.append(rel)
    if missing:
        raise SystemExit("Missing course page(s): " + ", ".join(missing))

    known = {p.resolve() for p in pages}
    unexpected = sorted(
        p for p in LESSONS.glob("**/*.html")
        if p.is_file() and p.resolve() not in known
    )
    if unexpected:
        raise SystemExit(
            "Lesson exists but is absent from COURSE_ORDER: "
            + ", ".join(str(p.relative_to(ROOT)) for p in unexpected)
        )

    return pages + sorted(p for p in LABS.glob("*.html") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    seen: set[str] = set()
    changed: list[Path] = []

    for path in teaching_pages():
        old = path.read_text(encoding="utf-8")
        new = normalize_page(old, seen)
        if new != old:
            changed.append(path)
            if args.write:
                path.write_text(new, encoding="utf-8")

    if args.write:
        print(
            f"First-use terminology normalized: {len(changed)} file(s) changed; "
            f"{len(seen)} technical terms introduced."
        )
        return 0

    if changed:
        print(
            f"First-use terminology audit FAILED: {len(changed)} page(s) "
            "need first-use normalization or duplicate-explanation cleanup."
        )
        for path in changed:
            print(f"  - {path.relative_to(ROOT)}")
        print("\nFirst few normalization diffs:")
        preview_seen: set[str] = set()
        shown = 0
        for path in teaching_pages():
            old = path.read_text(encoding="utf-8")
            new = normalize_page(old, preview_seen)
            if new == old:
                continue
            print(f"\n--- {path.relative_to(ROOT)}")
            diff = difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=str(path.relative_to(ROOT)),
                tofile=str(path.relative_to(ROOT)) + " (normalized)",
                lineterm="",
                n=1,
            )
            for line in list(diff)[:60]:
                print(line)
            shown += 1
            if shown >= 5:
                break
        print("Run: python scripts/audit_first_use_terms.py --write")
        return 1

    print(
        f"First-use terminology audit passed: {len(seen)} technical terms "
        "introduced once in pedagogical course order."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
