#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async block KV transfer with nonblocking P2P handles.")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=8)
    return p.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=2 ...")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("This lab intentionally uses exactly 2 ranks: prefill and decode")
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, device


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("hd,thd->ht", q, k) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", probs, v)


def main():
    args = parse_args()
    rank, device = init_dist()
    if args.tokens % args.block_size != 0:
        raise ValueError("--tokens must be divisible by --block-size")
    nblocks = args.tokens // args.block_size

    torch.manual_seed(777)
    x_prompt = torch.randn(args.tokens, args.heads, args.head_dim, device=device)
    wk = torch.randn(args.head_dim, args.head_dim, device=device) / args.head_dim**0.5
    wv = torch.randn(args.head_dim, args.head_dim, device=device) / args.head_dim**0.5
    q_decode = torch.randn(args.heads, args.head_dim, device=device)

    # Both sides know the model weights in this toy; only rank 0 computes prompt KV.
    k_full = torch.einsum("thd,df->thf", x_prompt, wk)
    v_full = torch.einsum("thd,df->thf", x_prompt, wv)
    ref = attention(q_decode, k_full, v_full)

    block_shape = (args.block_size, args.heads, args.head_dim)
    recv_k = [torch.empty(block_shape, device=device) for _ in range(nblocks)]
    recv_v = [torch.empty(block_shape, device=device) for _ in range(nblocks)]

    dist.barrier()
    sync_device(device)
    t0 = time.perf_counter()

    works = []
    if rank == 1:
        # Decode posts all destination receives first: this is the "registered /
        # preallocated destination" mental model, but still ordinary torch.distributed P2P.
        for i in range(nblocks):
            works.append(dist.irecv(recv_k[i], src=0))
            works.append(dist.irecv(recv_v[i], src=0))
    else:
        for i in range(nblocks):
            ks = k_full[i * args.block_size : (i + 1) * args.block_size].contiguous()
            vs = v_full[i * args.block_size : (i + 1) * args.block_size].contiguous()
            works.append(dist.isend(ks, dst=1))
            works.append(dist.isend(vs, dst=1))

    t_post = time.perf_counter()

    # Do work that does not depend on transferred KV before waiting.
    # This is intentionally control/metadata work, not a performance benchmark.
    checksum = 0
    for i in range(20000):
        checksum = (checksum + i * 17) % 1000003

    t_before_wait = time.perf_counter()
    for work in works:
        work.wait()
    sync_device(device)
    t_done = time.perf_counter()

    if rank == 1:
        k_remote = torch.cat(recv_k, dim=0)
        v_remote = torch.cat(recv_v, dim=0)
        out = attention(q_decode, k_remote, v_remote)
        err = (out - ref).abs().max().detach()
    else:
        err = torch.tensor(0.0, device=device)

    dist.broadcast(err, src=1)

    stats = torch.tensor(
        [
            (t_post - t0) * 1000,
            (t_before_wait - t_post) * 1000,
            (t_done - t_before_wait) * 1000,
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(stats) for _ in range(2)] if rank == 0 else None
    dist.gather(stats, gather_list=gathered, dst=0)

    if rank == 0:
        print("=== Mini Async Block KV Transfer ===")
        print(f"backend={dist.get_backend()} device={device.type} blocks={nblocks}")
        print("nonblocking handles       : isend / irecv -> Work.wait()")
        print(f"max attention error       : {err.item():.3e}")
        if gathered is not None:
            print(
                "prefill post/work/wait ms  : "
                + " / ".join(f"{x:.3f}" for x in gathered[0].tolist())
            )
            print(
                "decode  post/work/wait ms  : "
                + " / ".join(f"{x:.3f}" for x in gathered[1].tolist())
            )
        print("NOTE: timings show ordering only; they are not a network benchmark.")
        print("PASS" if err.item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
