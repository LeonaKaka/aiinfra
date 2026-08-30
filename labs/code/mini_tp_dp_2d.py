#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4-rank TP=2 x DP=2 topology with separate collectives.")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--samples-per-replica", type=int, default=4)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--ffn", type=int, default=12)
    p.add_argument("--out", type=int, default=6)
    return p.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=4 ...")
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


def make_2d_groups(rank: int, world_size: int, tp_size: int):
    if world_size % tp_size != 0:
        raise ValueError("WORLD_SIZE must be divisible by --tp")
    dp_size = world_size // tp_size
    own_tp_group = None
    own_dp_group = None

    # Every rank creates groups in exactly the same order.
    tp_groups = []
    for dp_rank in range(dp_size):
        ranks = [dp_rank * tp_size + t for t in range(tp_size)]
        group = dist.new_group(ranks=ranks)
        tp_groups.append(ranks)
        if rank in ranks:
            own_tp_group = group

    dp_groups = []
    for tp_rank in range(tp_size):
        ranks = [d * tp_size + tp_rank for d in range(dp_size)]
        group = dist.new_group(ranks=ranks)
        dp_groups.append(ranks)
        if rank in ranks:
            own_dp_group = group

    assert own_tp_group is not None and own_dp_group is not None
    return dp_size, own_tp_group, own_dp_group, tp_groups, dp_groups


def main():
    args = parse_args()
    rank, world_size, device = init_dist()
    dp_size, tp_group, dp_group, tp_groups, dp_groups = make_2d_groups(rank, world_size, args.tp)
    tp_rank = rank % args.tp
    dp_rank = rank // args.tp

    if args.ffn % args.tp != 0:
        raise ValueError("--ffn must be divisible by --tp")

    torch.manual_seed(1701)
    global_n = args.samples_per_replica * dp_size
    x_all = torch.randn(global_n, args.hidden, device=device)
    target_all = torch.randn(global_n, args.out, device=device)

    start = dp_rank * args.samples_per_replica
    end = start + args.samples_per_replica
    x = x_all[start:end]
    target = target_all[start:end]

    torch.manual_seed(1702)
    w1 = torch.randn(args.hidden, args.ffn, device=device) / args.hidden**0.5
    w2 = torch.randn(args.ffn, args.out, device=device) / args.ffn**0.5

    # Each DP replica owns the same TP shards. TP rank selects the shard.
    w1_shard = torch.chunk(w1, args.tp, dim=1)[tp_rank].contiguous()
    w2_shard = torch.chunk(w2, args.tp, dim=0)[tp_rank].contiguous()

    # TP forward: same samples within a DP replica, split layer weights.
    pre = x @ w1_shard
    h = torch.relu(pre)
    y_partial = h @ w2_shard
    y = y_partial.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM, group=tp_group)

    # Local loss gradient for this DP replica.
    dy = (y - target) / args.samples_per_replica

    # TP-local backward math.
    grad_w2 = h.transpose(0, 1) @ dy
    dh = dy @ w2_shard.transpose(0, 1)
    dpre = dh * (pre > 0)
    grad_w1 = x.transpose(0, 1) @ dpre
    grad_x = dpre @ w1_shard.transpose(0, 1)

    # Input gradient belongs to a replicated tensor within the TP group.
    dist.all_reduce(grad_x, op=dist.ReduceOp.SUM, group=tp_group)

    # Parameter-gradient shards belong to a fixed TP shard. Synchronize only
    # with the same TP shard in other DP replicas.
    dist.all_reduce(grad_w1, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(grad_w2, op=dist.ReduceOp.SUM, group=dp_group)
    grad_w1.div_(dp_size)
    grad_w2.div_(dp_size)

    # Dense global-batch reference.
    w1r = w1.clone().requires_grad_(True)
    w2r = w2.clone().requires_grad_(True)
    yr = torch.relu(x_all @ w1r) @ w2r
    loss_ref = 0.5 * (yr - target_all).square().sum() / global_n
    loss_ref.backward()

    ref_w1 = torch.chunk(w1r.grad, args.tp, dim=1)[tp_rank]
    ref_w2 = torch.chunk(w2r.grad, args.tp, dim=0)[tp_rank]
    errs = torch.tensor([
        (grad_w1 - ref_w1).abs().max(),
        (grad_w2 - ref_w2).abs().max(),
    ], device=device)
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)

    if rank == 0:
        print("=== Mini TP x DP 2D Topology ===")
        print(f"backend={dist.get_backend()} world_size={world_size} TP={args.tp} DP={dp_size}")
        print(f"TP groups: {tp_groups}")
        print(f"DP groups: {dp_groups}")
        print("TP collective : y partials + dX contributions")
        print("DP collective : same TP shard's dW1 / dW2 across data replicas")
        print(f"max dW1 shard error : {errs[0].item():.3e}")
        print(f"max dW2 shard error : {errs[1].item():.3e}")
        print("PASS" if errs.max().item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
