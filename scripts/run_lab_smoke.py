#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / ".lab-smoke-traces"

LABS: tuple[tuple[str, int, str, tuple[str, ...]], ...] = (
    ("A1 Mini Megatron TP", 2, "labs/code/mini_megatron_tp.py", ()),
    ("A2 Mini TP Backward", 2, "labs/code/mini_tp_backward.py", ()),
    ("A3 DP + Gradient Accumulation", 2, "labs/code/mini_dp_grad_accum.py", ()),
    ("A4 Mini Distributed Optimizer", 2, "labs/code/mini_distributed_optimizer.py", ()),
    ("A5 Bucketed Async Reduce-Scatter", 2, "labs/code/mini_bucketed_reduce_scatter.py", ()),
    ("A6 TP x DP 2D Topology", 4, "labs/code/mini_tp_dp_2d.py", ()),
    (
        "A7 Profiler-ready Overlap",
        2,
        "labs/code/mini_profiler_overlap.py",
        ("--trace-dir", str(TRACE_DIR)),
    ),
    ("B1 Mini KV Handoff", 2, "labs/code/mini_kv_handoff.py", ()),
    ("B2 Mini Block KV Handoff", 2, "labs/code/mini_block_kv_handoff.py", ()),
    ("B3 Mini Async KV Transfer", 2, "labs/code/mini_async_kv_transfer.py", ()),
    ("B4 Layer-wise KV Streaming", 2, "labs/code/mini_layerwise_kv_stream.py", ()),
    ("B5 KV Handshake + Lifetime", 2, "labs/code/mini_kv_handshake_lifetime.py", ()),
    (
        "B6 Registered Region Descriptor",
        2,
        "labs/code/mini_registered_region_descriptor.py",
        (),
    ),
    ("B7 KV Lease / Expiry", 2, "labs/code/mini_kv_lease_expiry.py", ()),
)


def main() -> int:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    # These labs are correctness/state-machine checks on CI. They must not turn
    # CPU/Gloo timings into claims about CUDA, NCCL, NIXL, RDMA, or GPUDirect.
    env.setdefault("CUDA_VISIBLE_DEVICES", "")

    TRACE_DIR.mkdir(exist_ok=True)
    failures: list[str] = []

    for name, nproc, script, extra in LABS:
        path = ROOT / script
        if not path.exists():
            failures.append(f"{name}: missing {script}")
            continue

        command = (
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            str(path),
            *extra,
        )
        print(f"\n=== {name} ===", flush=True)
        print("$ " + " ".join(command), flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if exc.stdout:
                output = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
                print(output, end="" if output.endswith("\n") else "\n", flush=True)
            failures.append(f"{name}: timed out after 180s")
            continue

        output = completed.stdout or ""
        if output:
            print(output, end="" if output.endswith("\n") else "\n", flush=True)

        reported_pass = any(line.strip() == "PASS" for line in output.splitlines())
        if completed.returncode != 0:
            failures.append(f"{name}: exited with status {completed.returncode}")
        elif not reported_pass:
            failures.append(f"{name}: process exited 0 but did not report a standalone PASS")
        else:
            print(f"SMOKE PASS: {name}", flush=True)

    if failures:
        print("\nLab smoke checks failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"\nLab smoke checks passed: {len(LABS)} CPU/Gloo correctness labs.")
    print("No CPU timing from this workflow is a GPU/NCCL/NIXL/RDMA benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
