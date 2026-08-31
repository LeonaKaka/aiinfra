#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TP x DP training step with sharded optimizer state and bucketed async reduce-scatter."
    )
    p.add_argument("--batch-per-replica", type=int, default=4)
    p.add_argument("--in-dim", type=int, default=6)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--out-dim", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    return p.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=4 ...")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError("This capstone expects exactly 4 ranks: TP=2 x DP=2")
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, device


def make_groups():
    tp_groups = [[0, 1], [2, 3]]
    dp_groups = [[0, 2], [1, 3]]
    tp_handles = [dist.new_group(ranks=g) for g in tp_groups]
    dp_handles = [dist.new_group(ranks=g) for g in dp_groups]
    return tp_groups, dp_groups, tp_handles, dp_handles


def main() -> None:
    args = parse_args()
    rank, device = init_dist()
    tp_groups, dp_groups, tp_handles, dp_handles = make_groups()

    tp_idx = rank % 2
    dp_idx = rank // 2
    tp_group = tp_handles[dp_idx]
    dp_group = dp_handles[tp_idx]
    global_batch = args.batch_per_replica * 2

    if args.hidden % 2 != 0:
        raise ValueError("--hidden must be divisible by TP=2")

    # Every rank can build the deterministic dense reference, while the actual
    # distributed step only consumes its DP replica's data and TP weight shard.
    torch.manual_seed(2901)
    x_global = torch.randn(global_batch, args.in_dim, device=device)
    target_global = torch.randn(global_batch, args.out_dim, device=device)
    torch.manual_seed(2902)
    w1_dense = torch.randn(args.in_dim, args.hidden, device=device) / args.in_dim**0.5
    w2_dense = torch.randn(args.hidden, args.out_dim, device=device) / args.hidden**0.5

    b0 = dp_idx * args.batch_per_replica
    b1 = b0 + args.batch_per_replica
    x = x_global[b0:b1]
    target = target_global[b0:b1]

    # Megatron-style MLP TP shape: W1 column split, W2 row split.
    w1_local = torch.chunk(w1_dense, 2, dim=1)[tp_idx].contiguous()
    w2_local = torch.chunk(w2_dense, 2, dim=0)[tp_idx].contiguous()

    pre = x @ w1_local
    h = torch.tanh(pre)
    y_partial = h @ w2_local
    y = y_partial.clone()
    dist.all_reduce(y, group=tp_group)

    # Local loss is normalized by global batch size so DP SUM yields the dense
    # global-batch gradient without a second divide.
    dy = (y - target) / global_batch

    # W2 is ready first. Start DP reduce-scatter immediately, then keep doing
    # independent backward math for W1 before waiting on W2 communication.
    dw2 = h.transpose(0, 1) @ dy
    flat_w2_grad = dw2.flatten().contiguous()
    if flat_w2_grad.numel() % 2 != 0:
        raise RuntimeError("W2 TP shard must split evenly across DP ranks")
    w2_grad_shard = torch.empty(flat_w2_grad.numel() // 2, device=device)
    w2_work = dist.reduce_scatter_single(
        w2_grad_shard, flat_w2_grad, group=dp_group, async_op=True
    )

    dh = dy @ w2_local.transpose(0, 1)
    dpre = dh * (1.0 - h.square())
    dw1 = x.transpose(0, 1) @ dpre

    flat_w1_grad = dw1.flatten().contiguous()
    if flat_w1_grad.numel() % 2 != 0:
        raise RuntimeError("W1 TP shard must split evenly across DP ranks")
    w1_grad_shard = torch.empty(flat_w1_grad.numel() // 2, device=device)
    w1_work = dist.reduce_scatter_single(
        w1_grad_shard, flat_w1_grad, group=dp_group, async_op=True
    )

    # Long-lived optimizer/master state can remain DP-sharded. This toy uses
    # SGD, so the only sharded state is the owned parameter/update chunk.
    w2_work.wait()
    flat_w2 = w2_local.flatten()
    w2_param_shard = torch.chunk(flat_w2, 2)[dp_idx].contiguous()
    w2_param_shard = w2_param_shard - args.lr * w2_grad_shard
    w2_updated_flat = torch.empty_like(flat_w2)
    dist.all_gather_single(w2_updated_flat, w2_param_shard, group=dp_group)
    w2_updated = w2_updated_flat.view_as(w2_local)

    w1_work.wait()
    flat_w1 = w1_local.flatten()
    w1_param_shard = torch.chunk(flat_w1, 2)[dp_idx].contiguous()
    w1_param_shard = w1_param_shard - args.lr * w1_grad_shard
    w1_updated_flat = torch.empty_like(flat_w1)
    dist.all_gather_single(w1_updated_flat, w1_param_shard, group=dp_group)
    w1_updated = w1_updated_flat.view_as(w1_local)

    # Dense reference for the same global batch and one SGD step.
    w1_ref = w1_dense.clone().requires_grad_(True)
    w2_ref = w2_dense.clone().requires_grad_(True)
    y_ref = torch.tanh(x_global @ w1_ref) @ w2_ref
    loss_ref = 0.5 * (y_ref - target_global).square().sum() / global_batch
    loss_ref.backward()
    with torch.no_grad():
        w1_ref_updated = w1_ref - args.lr * w1_ref.grad
        w2_ref_updated = w2_ref - args.lr * w2_ref.grad

    expected_w1 = torch.chunk(w1_ref_updated, 2, dim=1)[tp_idx]
    expected_w2 = torch.chunk(w2_ref_updated, 2, dim=0)[tp_idx]
    err_w1 = (w1_updated - expected_w1).abs().max()
    err_w2 = (w2_updated - expected_w2).abs().max()
    errors = torch.stack([err_w1, err_w2])
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)

    if rank == 0:
        print("=== Mini Training System Capstone ===")
        print(f"backend={dist.get_backend()} world_size=4 TP=2 DP=2")
        print(f"TP groups: {tp_groups}")
        print(f"DP groups: {dp_groups}")
        print("forward       : TP-sharded W1/W2 + TP all-reduce output")
        print("backward      : W2 bucket ready -> async DP reduce-scatter -> compute W1 -> async DP reduce-scatter")
        print("optimizer     : each DP rank updates one parameter chunk; all-gather rematerializes the TP shard")
        print("teaching note : async ordering is validated; CPU/Gloo timing is not proof of hardware overlap")
        print(f"max updated W1 TP-shard error: {errors[0].item():.3e}")
        print(f"max updated W2 TP-shard error: {errors[1].item():.3e}")
        print("PASS" if errors.max().item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
