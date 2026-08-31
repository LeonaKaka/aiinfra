#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two gradient buckets with async reduce-scatter launched as each bucket becomes ready."
    )
    p.add_argument("--samples-per-rank", type=int, default=8)
    p.add_argument("--in-dim", type=int, default=6)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--out-dim", type=int, default=4)
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


def async_reduce_scatter_avg(full_grad: torch.Tensor, world_size: int):
    flat = full_grad.contiguous().view(-1)
    if flat.numel() % world_size != 0:
        raise ValueError("bucket size must be divisible by WORLD_SIZE")
    out = torch.empty(flat.numel() // world_size, device=flat.device, dtype=flat.dtype)
    work = dist.reduce_scatter_single(
        out, flat, op=dist.ReduceOp.SUM, async_op=True
    )
    return out, work


def main():
    args = parse_args()
    rank, world_size, device = init_dist()
    if (args.hidden * args.out_dim) % world_size != 0:
        raise ValueError("W2 bucket size must be divisible by WORLD_SIZE")
    if (args.in_dim * args.hidden) % world_size != 0:
        raise ValueError("W1 bucket size must be divisible by WORLD_SIZE")

    torch.manual_seed(901)
    global_n = args.samples_per_rank * world_size
    x_all = torch.randn(global_n, args.in_dim, device=device)
    target_all = torch.randn(global_n, args.out_dim, device=device)
    start = rank * args.samples_per_rank
    end = start + args.samples_per_rank
    x = x_all[start:end]
    target = target_all[start:end]

    torch.manual_seed(902)
    w1 = torch.randn(args.in_dim, args.hidden, device=device) / args.in_dim**0.5
    w2 = torch.randn(args.hidden, args.out_dim, device=device) / args.hidden**0.5

    # Local forward. Manual backward lets us make bucket-ready order explicit.
    pre = x @ w1
    h = torch.relu(pre)
    y = h @ w2
    dy = (y - target) / args.samples_per_rank

    dist.barrier()
    t0 = time.perf_counter()

    # Bucket 0 (W2) becomes ready first.
    grad_w2_local = h.transpose(0, 1) @ dy
    grad_w2_shard, work_w2 = async_reduce_scatter_avg(grad_w2_local, world_size)
    t_w2_post = time.perf_counter()

    # This backward compute is independent of the already-launched W2 reduction.
    dh = dy @ w2.transpose(0, 1)
    dpre = dh * (pre > 0)
    grad_w1_local = x.transpose(0, 1) @ dpre
    t_w1_ready = time.perf_counter()

    # Bucket 1 (W1) launches later, once its gradient is ready.
    grad_w1_shard, work_w1 = async_reduce_scatter_avg(grad_w1_local, world_size)
    t_w1_post = time.perf_counter()

    work_w2.wait()
    work_w1.wait()
    grad_w2_shard.div_(world_size)
    grad_w1_shard.div_(world_size)
    t_done = time.perf_counter()

    # Dense global-batch reference.
    w1r = w1.clone().requires_grad_(True)
    w2r = w2.clone().requires_grad_(True)
    yr = torch.relu(x_all @ w1r) @ w2r
    loss = 0.5 * (yr - target_all).square().sum() / global_n
    loss.backward()

    ref_w2_shard = torch.chunk(w2r.grad.contiguous().view(-1), world_size)[rank]
    ref_w1_shard = torch.chunk(w1r.grad.contiguous().view(-1), world_size)[rank]
    errors = torch.tensor(
        [
            (grad_w2_shard - ref_w2_shard).abs().max(),
            (grad_w1_shard - ref_w1_shard).abs().max(),
        ],
        device=device,
    )
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)

    timings = torch.tensor(
        [
            (t_w2_post - t0) * 1000,
            (t_w1_ready - t_w2_post) * 1000,
            (t_w1_post - t_w1_ready) * 1000,
            (t_done - t_w1_post) * 1000,
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(timings) for _ in range(world_size)] if rank == 0 else None
    dist.gather(timings, gather_list=gathered, dst=0)

    if rank == 0:
        print("=== Mini Bucketed Async Reduce-Scatter ===")
        print(f"backend={dist.get_backend()} world_size={world_size} device={device.type}")
        print("bucket ready order          : W2 -> W1")
        print("W2 collective              : async reduce-scatter")
        print("compute between post/wait   : dH + dPre + dW1")
        print(f"max W2 shard error          : {errors[0].item():.3e}")
        print(f"max W1 shard error          : {errors[1].item():.3e}")
        if gathered is not None:
            print(
                "rank0 postW2/compute/postW1/wait ms: "
                + " / ".join(f"{x:.3f}" for x in gathered[0].tolist())
            )
        print("NOTE: timing demonstrates ordering only, not proven hardware overlap.")
        print("PASS" if errors.max().item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
