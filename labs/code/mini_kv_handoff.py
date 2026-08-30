#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Prefill -> KV handoff -> Decode experiment."
    )
    parser.add_argument("--seq", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()


def init_dist() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Launch with: torchrun --standalone --nproc-per-node=2 mini_kv_handoff.py"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("This lab intentionally uses exactly 2 ranks.")

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)

    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def main() -> None:
    args = parse_args()
    rank, _, device = init_dist()

    torch.manual_seed(11)
    prompt = torch.randn(args.seq, args.hidden, device=device)
    decode_token = torch.randn(1, args.hidden, device=device)
    wq = torch.randn(args.hidden, args.hidden, device=device) / args.hidden**0.5
    wk = torch.randn(args.hidden, args.hidden, device=device) / args.hidden**0.5
    wv = torch.randn(args.hidden, args.hidden, device=device) / args.hidden**0.5

    # Rank 0 is the prefiller. Rank 1 is the decoder.
    if rank == 0:
        k_prompt = prompt @ wk
        v_prompt = prompt @ wv
        sync(device)
        t0 = time.perf_counter()
        dist.send(k_prompt.contiguous(), dst=1)
        dist.send(v_prompt.contiguous(), dst=1)
        sync(device)
        transfer_ms = (time.perf_counter() - t0) * 1000

        timing = torch.tensor([transfer_ms], device=device, dtype=torch.float64)
        dist.send(timing, dst=1)

    else:
        recv_k = torch.empty(args.seq, args.hidden, device=device)
        recv_v = torch.empty(args.seq, args.hidden, device=device)

        sync(device)
        t0 = time.perf_counter()
        dist.recv(recv_k, src=0)
        dist.recv(recv_v, src=0)
        sync(device)
        consumer_wait_ms = (time.perf_counter() - t0) * 1000

        timing = torch.empty(1, device=device, dtype=torch.float64)
        dist.recv(timing, src=0)

        q_step = decode_token @ wq
        k_step = decode_token @ wk
        v_step = decode_token @ wv

        handed_k = torch.cat([recv_k, k_step], dim=0)
        handed_v = torch.cat([recv_v, v_step], dim=0)
        handed_out = attention(q_step, handed_k, handed_v)

        # Dense baseline: recompute prompt KV on the decoder.
        ref_k = torch.cat([prompt @ wk, k_step], dim=0)
        ref_v = torch.cat([prompt @ wv, v_step], dim=0)
        ref_out = attention(q_step, ref_k, ref_v)
        max_err = (handed_out - ref_out).abs().max().item()

        kv_bytes = (recv_k.numel() + recv_v.numel()) * recv_k.element_size()
        print("=== Mini KV Handoff ===")
        print(f"backend={dist.get_backend()} device={device.type}")
        print(f"prompt KV shape      : K={tuple(recv_k.shape)} V={tuple(recv_v.shape)}")
        print(f"payload              : {kv_bytes} bytes")
        print(f"producer send time   : {timing.item():.3f} ms")
        print(f"consumer wait time   : {consumer_wait_ms:.3f} ms")
        print(f"max |handoff-ref|    : {max_err:.3e}")
        print("PASS" if max_err < 1e-5 else "FAIL")

    dist.barrier()

    # Repeated P2P loop: useful for seeing the handoff path, not a fabric benchmark.
    if args.steps > 0:
        if rank == 0:
            k_prompt = prompt @ wk
            v_prompt = prompt @ wv
        else:
            recv_k = torch.empty(args.seq, args.hidden, device=device)
            recv_v = torch.empty(args.seq, args.hidden, device=device)

        dist.barrier()
        sync(device)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            if rank == 0:
                dist.send(k_prompt, dst=1)
                dist.send(v_prompt, dst=1)
            else:
                dist.recv(recv_k, src=0)
                dist.recv(recv_v, src=0)
        sync(device)
        dist.barrier()
        avg_ms = (time.perf_counter() - t0) * 1000 / args.steps
        if rank == 1:
            print(f"avg handoff loop     : {avg_ms:.3f} ms / iteration")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
