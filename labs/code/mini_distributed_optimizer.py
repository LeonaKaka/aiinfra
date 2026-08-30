#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A tiny reduce-scatter + sharded optimizer + all-gather training loop."
    )
    p.add_argument("--features", type=int, default=8)
    p.add_argument("--samples-per-rank", type=int, default=6)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--momentum", type=float, default=0.9)
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


def reduce_scatter_average(full_grad: torch.Tensor, world_size: int) -> torch.Tensor:
    if full_grad.numel() % world_size != 0:
        raise ValueError("parameter count must be divisible by WORLD_SIZE")
    shard = torch.empty(full_grad.numel() // world_size, device=full_grad.device, dtype=full_grad.dtype)
    dist.reduce_scatter_tensor(shard, full_grad.contiguous(), op=dist.ReduceOp.SUM)
    shard.div_(world_size)
    return shard


def all_gather_flat(shard: torch.Tensor, world_size: int) -> torch.Tensor:
    full = torch.empty(shard.numel() * world_size, device=shard.device, dtype=shard.dtype)
    dist.all_gather_into_tensor(full, shard.contiguous())
    return full


def main():
    args = parse_args()
    rank, world_size, device = init_dist()
    if args.features % world_size != 0:
        raise ValueError("--features must be divisible by WORLD_SIZE")

    torch.manual_seed(301)
    true_w = torch.randn(args.features, device=device)
    torch.manual_seed(302)
    full_param = torch.randn(args.features, device=device)
    ref_param = full_param.clone()
    ref_momentum = torch.zeros_like(ref_param)

    shard_size = args.features // world_size
    local_momentum = torch.zeros(shard_size, device=device)

    max_param_err = 0.0
    max_state_err = 0.0

    for step in range(args.steps):
        torch.manual_seed(400 + step)
        global_n = args.samples_per_rank * world_size
        x_all = torch.randn(global_n, args.features, device=device)
        y_all = x_all @ true_w + 0.03 * torch.randn(global_n, device=device)

        start = rank * args.samples_per_rank
        end = start + args.samples_per_rank
        x_local = x_all[start:end]
        y_local = y_all[start:end]

        # Every DP rank has the full parameters for forward/backward.
        p = full_param.detach().clone().requires_grad_(True)
        local_loss = 0.5 * (x_local @ p - y_local).square().mean()
        local_loss.backward()
        full_local_grad = p.grad.detach()

        # But gradient reduction lands directly into a local shard.
        grad_shard = reduce_scatter_average(full_local_grad, world_size)

        # Only this rank's optimizer state and parameter shard are updated.
        param_shard = torch.chunk(full_param, world_size)[rank].contiguous()
        local_momentum.mul_(args.momentum).add_(grad_shard)
        param_shard = param_shard - args.lr * local_momentum

        # Re-materialize full parameters for the next forward.
        full_param = all_gather_flat(param_shard, world_size)

        # Dense global-batch reference.
        rp = ref_param.detach().clone().requires_grad_(True)
        ref_loss = 0.5 * (x_all @ rp - y_all).square().mean()
        ref_loss.backward()
        ref_momentum.mul_(args.momentum).add_(rp.grad)
        ref_param = rp.detach() - args.lr * ref_momentum

        state_full = all_gather_flat(local_momentum, world_size)
        param_err = (full_param - ref_param).abs().max()
        state_err = (state_full - ref_momentum).abs().max()
        max_param_err = max(max_param_err, float(param_err))
        max_state_err = max(max_state_err, float(state_err))

    errors = torch.tensor([max_param_err, max_state_err], device=device)
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)

    if rank == 0:
        full_state_elems = args.features
        local_state_elems = args.features // world_size
        print("=== Mini Distributed Optimizer ===")
        print(f"backend={dist.get_backend()} world_size={world_size} device={device.type}")
        print(f"steps                         : {args.steps}")
        print(f"optimizer-state elems / rank  : {local_state_elems} (vs {full_state_elems} replicated)")
        print("gradient collective           : reduce-scatter")
        print("parameter materialization     : all-gather")
        print(f"max parameter error           : {errors[0].item():.3e}")
        print(f"max optimizer-state error     : {errors[1].item():.3e}")
        print("PASS" if errors.max().item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
