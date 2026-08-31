#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ring-style context-parallel attention correctness reference.")
    p.add_argument("--seq", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=8)
    return p.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=2 ...")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def exchange_chunk(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return x
    nxt = (rank + 1) % world_size
    prv = (rank - 1) % world_size
    recv = torch.empty_like(x)
    ops = [
        dist.P2POp(dist.isend, x.contiguous(), nxt),
        dist.P2POp(dist.irecv, recv, prv),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return recv


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_dist()
    if args.seq % world_size != 0:
        raise ValueError("--seq must be divisible by WORLD_SIZE")

    local_seq = args.seq // world_size
    start = rank * local_seq
    end = start + local_seq

    torch.manual_seed(2701)
    q = torch.randn(args.seq, args.head_dim, device=device)
    k = torch.randn(args.seq, args.head_dim, device=device)
    v = torch.randn(args.seq, args.head_dim, device=device)

    q_local = q[start:end].contiguous()
    k_chunk = k[start:end].contiguous()
    v_chunk = v[start:end].contiguous()
    owner = rank
    scale = 1.0 / math.sqrt(args.head_dim)

    score_chunks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []
    owner_order: list[int] = []
    q_pos = torch.arange(start, end, device=device)

    for round_idx in range(world_size):
        key_start = owner * local_seq
        key_pos = torch.arange(key_start, key_start + local_seq, device=device)
        scores = (q_local @ k_chunk.transpose(0, 1)) * scale
        causal_mask = key_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        score_chunks.append(scores)
        value_chunks.append(v_chunk)
        owner_order.append(owner)

        if round_idx != world_size - 1:
            k_chunk = exchange_chunk(k_chunk, rank, world_size)
            v_chunk = exchange_chunk(v_chunk, rank, world_size)
            owner = (owner - 1) % world_size

    scores_all = torch.cat(score_chunks, dim=1)
    values_all = torch.cat(value_chunks, dim=0)
    probs = torch.softmax(scores_all, dim=-1)
    out_local = probs @ values_all

    dense_scores = (q @ k.transpose(0, 1)) * scale
    positions = torch.arange(args.seq, device=device)
    dense_mask = positions.unsqueeze(0) > positions.unsqueeze(1)
    dense_scores = dense_scores.masked_fill(dense_mask, float("-inf"))
    ref = torch.softmax(dense_scores, dim=-1) @ v
    err = (out_local - ref[start:end]).abs().max()
    dist.all_reduce(err, op=dist.ReduceOp.MAX)

    if rank == 0:
        print("=== Mini Context Parallel ===")
        print(f"backend={dist.get_backend()} world_size={world_size}")
        print(f"global sequence   : {args.seq} tokens")
        print(f"local Q shard     : {local_seq} tokens (Q never leaves its owner in this lab)")
        print(f"KV owner rounds   : {owner_order}")
        print("causal semantics  : global query/key positions are used for masking")
        print("teaching model    : ring-style KV rotation; production CP may use other partitions/algorithms")
        print(f"max reference error: {err.item():.3e}")
        print("PASS" if err.item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
