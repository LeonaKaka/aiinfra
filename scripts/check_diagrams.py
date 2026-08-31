#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

# These standalone core diagrams have been manually re-rendered and accepted at
# the real lesson-column scale. Keep this list narrow: intentionally wide
# timelines / inline architecture canvases are allowed elsewhere when their
# wrapper provides local horizontal scrolling.
LESSON_NATIVE_SVGS = (
    "learn/04-distributed/process-rank-groups.svg",
    "learn/04-distributed/collectives-state-map.svg",
    "learn/04-distributed/nccl-topology-path.svg",
    "learn/05-megatron/distributed-optimizer-flow.svg",
    "learn/05-megatron/context-parallel-ring.svg",
    "learn/05-megatron/expert-parallel-flow.svg",
    "learn/05-megatron/communication-overlap-critical-path.svg",
    "learn/06-llm-inference/autoregressive-loop.svg",
    "learn/06-llm-inference/prefill-decode-scheduler.svg",
    "learn/07-vllm/vllm-architecture-request-loop.svg",
    "learn/07-vllm/scheduler-continuous-batching.svg",
    "learn/07-vllm/prefix-cache-preemption.svg",
)

VIEWBOX_RE = re.compile(
    r"viewBox\s*=\s*['\"]\s*[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*['\"]"
)
MAX_NATIVE_WIDTH = 800.0


def main() -> int:
    failures: list[str] = []

    for rel in LESSON_NATIVE_SVGS:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"{rel}: accepted core diagram is missing")
            continue

        text = path.read_text(encoding="utf-8")
        match = VIEWBOX_RE.search(text)
        if not match:
            failures.append(f"{rel}: missing parseable SVG viewBox")
            continue

        width = float(match.group(1))
        height = float(match.group(2))
        if width <= 0 or height <= 0:
            failures.append(f"{rel}: invalid viewBox dimensions {width:g}×{height:g}")
            continue
        if width > MAX_NATIVE_WIDTH:
            failures.append(
                f"{rel}: accepted lesson-native diagram regressed to {width:g}px wide; "
                f"keep it <= {MAX_NATIVE_WIDTH:g}px or explicitly redesign it as a local-scroll wide diagram"
            )

    if failures:
        print("Diagram checks failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print(
        f"Diagram checks passed: {len(LESSON_NATIVE_SVGS)} accepted standalone core SVGs "
        f"remain <= {MAX_NATIVE_WIDTH:g}px native viewBox width."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
