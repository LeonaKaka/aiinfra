#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data parallel gradient accumulation with one all-reduce.")
    p.add_argument("--samples-per-rank", type=int, default=8)
    p.add_argument("--microbatches", type=int, default=2)
    p.add_argument("--features", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.2)
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


def main():
    args = parse_args()
    rank, world_size, device = init_dist()
    if args.samples_per_rank % args.microbatches != 0:
        raise ValueError("--samples-per-rank must be divisible by --microbatches")

    torch.manual_seed(101)
    global_n = args.samples_per_rank * world_size
    x_all = torch.randn(global_n, args.features, device=device)
    true_w = torch.randn(args.features, 1, device=device)
    true_b = torch.randn(1, device=device)
    y_all = x_all @ true_w + true_b + 0.05 * torch.randn(global_n, 1, device=device)

    start = rank * args.samples_per_rank
    end = start + args.samples_per_rank
    x_local, y_local = x_all[start:end], y_all[start:end]

    torch.manual_seed(202)
    w0 = torch.randn(args.features, 1, device=device)
    b0 = torch.randn(1, device=device)

    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)

    micro = args.samples_per_rank // args.microbatches
    for i in range(args.microbatches):
        xb = x_local[i * micro : (i + 1) * micro]
        yb = y_local[i * micro : (i + 1) * micro]
        pred = xb @ w + b
        # SUM locally so accumulation is exactly additive. We normalize once,
        # after the cross-rank reduction, by the global number of samples.
        loss_sum = 0.5 * (pred - yb).square().sum()
        loss_sum.backward()

    # One communication after all local microbatches.
    dist.all_reduce(w.grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(b.grad, op=dist.ReduceOp.SUM)
    w.grad.div_(global_n)
    b.grad.div_(global_n)

    with torch.no_grad():
        w_new = w - args.lr * w.grad
        b_new = b - args.lr * b.grad

    # Dense reference: same initial params, one full global batch.
    wr = w0.clone().requires_grad_(True)
    br = b0.clone().requires_grad_(True)
    loss_ref = 0.5 * ((x_all @ wr + br) - y_all).square().mean()
    loss_ref.backward()
    with torch.no_grad():
        wr_new = wr - args.lr * wr.grad
        br_new = br - args.lr * br.grad

    errs = torch.tensor([
        (w.grad - wr.grad).abs().max(),
        (b.grad - br.grad).abs().max(),
        (w_new - wr_new).abs().max(),
        (b_new - br_new).abs().max(),
    ], device=device)
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)

    if rank == 0:
        print("=== Mini Data Parallel + Gradient Accumulation ===")
        print(f"backend={dist.get_backend()} world_size={world_size} device={device.type}")
        print(f"microbatches per rank   : {args.microbatches}")
        print("gradient syncs per step : 1")
        print(f"max dW error            : {errs[0].item():.3e}")
        print(f"max db error            : {errs[1].item():.3e}")
        print(f"max updated W error     : {errs[2].item():.3e}")
        print(f"max updated b error     : {errs[3].item():.3e}")
        print("PASS" if errs.max().item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
