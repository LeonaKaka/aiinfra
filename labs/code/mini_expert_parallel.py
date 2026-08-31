#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Balanced MoE expert-parallel dispatch/combine reference.")
    p.add_argument("--tokens-per-rank", type=int, default=4)
    p.add_argument("--hidden", type=int, default=6)
    return p.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=2 ...")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("This teaching lab expects exactly 2 expert-parallel ranks")
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def expert_fn(x: torch.Tensor, expert_id: torch.Tensor) -> torch.Tensor:
    # Deterministic stand-in for different expert MLPs.
    scale = expert_id.to(x.dtype).unsqueeze(1) + 1.0
    bias = expert_id.to(x.dtype).unsqueeze(1) * 0.125
    return x * scale + bias


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_dist()
    if args.tokens_per_rank % world_size != 0:
        raise ValueError("--tokens-per-rank must be divisible by WORLD_SIZE")

    num_experts = 4
    experts_per_rank = num_experts // world_size
    rows_per_dest = args.tokens_per_rank // world_size

    torch.manual_seed(2801 + rank)
    x_local = torch.randn(args.tokens_per_rank, args.hidden, device=device)
    token_ids = rank * args.tokens_per_rank + torch.arange(args.tokens_per_rank, device=device)
    expert_ids = token_ids % num_experts
    owners = expert_ids // experts_per_rank

    # Pack metadata with features so the receiving rank knows which local expert
    # to execute and can later restore original token order.
    rows: list[torch.Tensor] = []
    for dest in range(world_size):
        idx = torch.nonzero(owners == dest, as_tuple=False).flatten()
        if idx.numel() != rows_per_dest:
            raise RuntimeError("Teaching router expected balanced equal splits")
        meta = torch.stack([token_ids[idx], expert_ids[idx]], dim=1).to(x_local.dtype)
        rows.append(torch.cat([meta, x_local[idx]], dim=1))
    send = torch.cat(rows, dim=0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)

    recv_token_ids = recv[:, 0].to(torch.long)
    recv_expert_ids = recv[:, 1].to(torch.long)
    recv_x = recv[:, 2:]
    local_min = rank * experts_per_rank
    local_max = local_min + experts_per_rank
    if not torch.all((recv_expert_ids >= local_min) & (recv_expert_ids < local_max)):
        raise RuntimeError("Received a token for an expert not owned by this rank")

    recv_y = expert_fn(recv_x, recv_expert_ids)
    processed = torch.cat(
        [recv_token_ids.to(recv_y.dtype).unsqueeze(1), recv_y], dim=1
    ).contiguous()

    # Equal routing splits make the reverse all-to-all structurally symmetric:
    # rows are grouped by source rank after the first all-to-all.
    returned = torch.empty_like(processed)
    dist.all_to_all_single(returned, processed)

    returned_ids = returned[:, 0].to(torch.long)
    returned_y = returned[:, 1:]
    order = torch.argsort(returned_ids)
    returned_ids = returned_ids[order]
    returned_y = returned_y[order]

    expected_ids = token_ids
    ref = expert_fn(x_local, expert_ids)
    id_ok = torch.equal(returned_ids, expected_ids)
    err = (returned_y - ref).abs().max()
    ok = torch.tensor(1 if id_ok else 0, device=device, dtype=torch.int32)
    dist.all_reduce(err, op=dist.ReduceOp.MAX)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)

    if rank == 0:
        print("=== Mini Expert Parallel ===")
        print(f"backend={dist.get_backend()} EP={world_size} experts={num_experts}")
        print(f"expert ownership : rank0 -> [0,1], rank1 -> [2,3]")
        print("dataflow         : router -> all-to-all dispatch -> local experts -> reverse all-to-all")
        print("teaching router  : deterministic and perfectly balanced; no capacity/drop/load-balancing model")
        print(f"token order restored: {bool(ok.item())}")
        print(f"max reference error : {err.item():.3e}")
        print("PASS" if ok.item() == 1 and err.item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
