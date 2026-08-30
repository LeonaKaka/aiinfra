#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal tensor-parallel MLP: ColumnParallelLinear + RowParallelLinear."
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=12)
    parser.add_argument("--out", type=int, default=6)
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()


def init_dist() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Launch with torchrun, for example: "
            "torchrun --standalone --nproc-per-node=2 mini_megatron_tp.py"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

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


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_dist()

    if args.ffn % world_size != 0:
        raise ValueError("--ffn must be divisible by WORLD_SIZE")

    torch.manual_seed(7)
    x = torch.randn(args.batch, args.hidden, device=device)
    w1 = torch.randn(args.hidden, args.ffn, device=device) / args.hidden**0.5
    w2 = torch.randn(args.ffn, args.out, device=device) / args.ffn**0.5

    # Dense reference: the result our sharded computation must reproduce.
    dense_y = torch.relu(x @ w1) @ w2

    # Column-parallel W1: split output features across ranks.
    w1_shard = torch.chunk(w1, world_size, dim=1)[rank].contiguous()
    local_h = torch.relu(x @ w1_shard)

    # Row-parallel W2: each rank owns rows matching its local hidden shard.
    w2_shard = torch.chunk(w2, world_size, dim=0)[rank].contiguous()
    local_y = local_h @ w2_shard

    # Full output = sum of partial outputs.
    dist.all_reduce(local_y, op=dist.ReduceOp.SUM)

    max_err = (local_y - dense_y).abs().max()
    dist.all_reduce(max_err, op=dist.ReduceOp.MAX)

    # Tiny timing loop: useful for execution order, not a performance claim.
    dist.barrier()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        h = torch.relu(x @ w1_shard)
        y = h @ w2_shard
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    sync(device)
    dist.barrier()
    elapsed_ms = (time.perf_counter() - t0) * 1000 / args.steps

    if rank == 0:
        shard_width = args.ffn // world_size
        collective_buffer_bytes = args.batch * args.out * x.element_size()
        print("=== Mini Megatron TP ===")
        print(f"backend={dist.get_backend()} world_size={world_size} device={device.type}")
        print(f"W1 dense shape      : [{args.hidden}, {args.ffn}]")
        print(f"W1 shard / rank     : [{args.hidden}, {shard_width}]")
        print(f"W2 shard / rank     : [{shard_width}, {args.out}]")
        print(f"all-reduce buffer   : {collective_buffer_bytes} bytes / rank")
        print(f"max |TP - dense|    : {max_err.item():.3e}")
        print(f"avg step time       : {elapsed_ms:.3f} ms")
        print("PASS" if max_err.item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
