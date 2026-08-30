#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual backward pass for a 2-layer tensor-parallel MLP."
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=12)
    parser.add_argument("--out", type=int, default=6)
    return parser.parse_args()


def init_dist() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Launch with torchrun, for example: "
            "torchrun --standalone --nproc-per-node=2 mini_tp_backward.py"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")

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
    if args.ffn % world_size != 0:
        raise ValueError("--ffn must be divisible by WORLD_SIZE")

    torch.manual_seed(17)
    x = torch.randn(args.batch, args.hidden, device=device)
    w1 = torch.randn(args.hidden, args.ffn, device=device) / args.hidden**0.5
    w2 = torch.randn(args.ffn, args.out, device=device) / args.ffn**0.5
    grad_y = torch.randn(args.batch, args.out, device=device)

    # Dense autograd reference. We compare every important TP gradient against it.
    x_ref = x.detach().clone().requires_grad_(True)
    w1_ref = w1.detach().clone().requires_grad_(True)
    w2_ref = w2.detach().clone().requires_grad_(True)
    y_ref = torch.relu(x_ref @ w1_ref) @ w2_ref
    y_ref.backward(grad_y)

    # Forward: Column Parallel W1 followed by Row Parallel W2.
    w1_shard = torch.chunk(w1, world_size, dim=1)[rank].contiguous()
    w2_shard = torch.chunk(w2, world_size, dim=0)[rank].contiguous()
    pre = x @ w1_shard
    hidden = torch.relu(pre)
    partial_y = hidden @ w2_shard
    y = partial_y.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)

    # Manual backward, starting from a replicated upstream gradient dY.
    # W2 gradients are already sharded: each rank updates only its local rows.
    grad_w2_shard = hidden.transpose(0, 1) @ grad_y
    grad_hidden = grad_y @ w2_shard.transpose(0, 1)
    grad_pre = grad_hidden * (pre > 0)

    # W1 gradients are also already sharded: each rank owns its output columns.
    grad_w1_shard = x.transpose(0, 1) @ grad_pre

    # dX receives contributions from every W1 column shard, so the previous
    # replicated layer needs their sum.
    grad_x = grad_pre @ w1_shard.transpose(0, 1)
    dist.all_reduce(grad_x, op=dist.ReduceOp.SUM)

    # Reconstruct full weight gradients only for the correctness check.
    gathered_w1 = [torch.empty_like(grad_w1_shard) for _ in range(world_size)]
    gathered_w2 = [torch.empty_like(grad_w2_shard) for _ in range(world_size)]
    dist.all_gather(gathered_w1, grad_w1_shard)
    dist.all_gather(gathered_w2, grad_w2_shard)
    grad_w1 = torch.cat(gathered_w1, dim=1)
    grad_w2 = torch.cat(gathered_w2, dim=0)

    errors = torch.tensor(
        [
            (y - y_ref.detach()).abs().max(),
            (grad_x - x_ref.grad).abs().max(),
            (grad_w1 - w1_ref.grad).abs().max(),
            (grad_w2 - w2_ref.grad).abs().max(),
        ],
        device=device,
    )
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)

    if rank == 0:
        print("=== Mini TP Backward ===")
        print(f"backend={dist.get_backend()} world_size={world_size} device={device.type}")
        print(f"max forward error : {errors[0].item():.3e}")
        print(f"max dX error      : {errors[1].item():.3e}")
        print(f"max dW1 error     : {errors[2].item():.3e}")
        print(f"max dW2 error     : {errors[3].item():.3e}")
        print("PASS" if errors.max().item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
