#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TP row-parallel partials -> reduce-scatter into sequence-parallel activations."
    )
    p.add_argument("--seq", type=int, default=8)
    p.add_argument("--hidden", type=int, default=8)
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


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_dist()

    if args.seq % world_size != 0:
        raise ValueError("--seq must be divisible by WORLD_SIZE")
    if args.hidden % world_size != 0:
        raise ValueError("--hidden must be divisible by WORLD_SIZE")

    torch.manual_seed(2501)
    x = torch.randn(args.seq, args.hidden, device=device)
    w = torch.randn(args.hidden, args.hidden, device=device) / args.hidden**0.5

    # Model a row-parallel linear: each TP rank owns a hidden-input shard and
    # produces a full-[S,H] partial contribution.
    x_shard = torch.chunk(x, world_size, dim=1)[rank].contiguous()
    w_shard = torch.chunk(w, world_size, dim=0)[rank].contiguous()
    y_partial = x_shard @ w_shard

    # Instead of all-reducing y onto every TP rank, reduce + scatter along the
    # sequence axis in one collective. Each rank keeps only S / TP tokens.
    local_seq = args.seq // world_size
    y_seq = torch.empty(local_seq, args.hidden, device=device)
    dist.reduce_scatter_single(y_seq, y_partial.contiguous(), op=dist.ReduceOp.SUM)

    # Sequence-local work can now happen without re-replicating the activation.
    local_act = torch.tanh(y_seq)

    # A later consumer that needs the full sequence can gather it back.
    gathered = torch.empty(args.seq, args.hidden, device=device)
    dist.all_gather_single(gathered, local_act.contiguous())

    ref = torch.tanh(x @ w)
    err = (gathered - ref).abs().max()
    dist.all_reduce(err, op=dist.ReduceOp.MAX)

    if rank == 0:
        print("=== Mini Sequence Parallel ===")
        print(f"backend={dist.get_backend()} world_size={world_size}")
        print(f"full activation    : [{args.seq}, {args.hidden}]")
        print(f"per-rank SP shard  : [{local_seq}, {args.hidden}]")
        print("transition         : TP partials -> reduce-scatter(sequence) -> local op")
        print("reconstruction     : all-gather(sequence) only when a full view is needed")
        print(f"max reference error: {err.item():.3e}")
        print("PASS" if err.item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
