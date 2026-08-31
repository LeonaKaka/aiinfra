#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage pipeline with explicit activation/gradient P2P.")
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--microbatches", type=int, default=3)
    p.add_argument("--in-dim", type=int, default=6)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--out-dim", type=int, default=4)
    return p.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=2 ...")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("This teaching lab expects exactly 2 ranks / pipeline stages")
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, device


def main() -> None:
    args = parse_args()
    rank, device = init_dist()
    if args.samples % args.microbatches != 0:
        raise ValueError("--samples must be divisible by --microbatches")
    mb = args.samples // args.microbatches

    torch.manual_seed(2601)
    x_all = torch.randn(args.samples, args.in_dim, device=device)
    target_all = torch.randn(args.samples, args.out_dim, device=device)
    torch.manual_seed(2602)
    w0 = torch.randn(args.in_dim, args.hidden, device=device) / args.in_dim**0.5
    w1 = torch.randn(args.hidden, args.out_dim, device=device) / args.hidden**0.5

    grad_w0 = torch.zeros_like(w0)
    w1_stage = w1.clone().requires_grad_(True)

    if rank == 0:
        saved_x: list[torch.Tensor] = []
        for i in range(args.microbatches):
            x_mb = x_all[i * mb : (i + 1) * mb]
            h = x_mb @ w0
            saved_x.append(x_mb)
            dist.send(h.contiguous(), dst=1)

        for i in reversed(range(args.microbatches)):
            grad_h = torch.empty(mb, args.hidden, device=device)
            dist.recv(grad_h, src=1)
            grad_w0.add_(saved_x[i].transpose(0, 1) @ grad_h)
    else:
        saved_h: list[torch.Tensor] = []
        saved_loss: list[torch.Tensor] = []
        for i in range(args.microbatches):
            h = torch.empty(mb, args.hidden, device=device)
            dist.recv(h, src=0)
            h = h.detach().requires_grad_(True)
            target_mb = target_all[i * mb : (i + 1) * mb]
            y = h @ w1_stage
            loss = 0.5 * (y - target_mb).square().sum() / args.samples
            saved_h.append(h)
            saved_loss.append(loss)

        for i in reversed(range(args.microbatches)):
            saved_loss[i].backward()
            assert saved_h[i].grad is not None
            dist.send(saved_h[i].grad.contiguous(), dst=0)

    # Dense reference: the pipeline must equal one ordinary two-layer model.
    w0_ref = w0.clone().requires_grad_(True)
    w1_ref = w1.clone().requires_grad_(True)
    y_ref = (x_all @ w0_ref) @ w1_ref
    loss_ref = 0.5 * (y_ref - target_all).square().sum() / args.samples
    loss_ref.backward()

    errs = torch.zeros(2, device=device)
    if rank == 0:
        errs[0] = (grad_w0 - w0_ref.grad).abs().max()
    else:
        assert w1_stage.grad is not None
        errs[1] = (w1_stage.grad - w1_ref.grad).abs().max()
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)

    if rank == 0:
        print("=== Mini Pipeline Parallel ===")
        print(f"backend={dist.get_backend()} stages=2 microbatches={args.microbatches}")
        print("stage 0 owns: W0 and input-side forward/backward")
        print("stage 1 owns: W1, loss, and output-side backward")
        print("P2P boundary : activation -> ; <- activation gradient")
        print("schedule     : teaching GPipe-style F0 F1 F2 -> B2 B1 B0")
        print("note         : Megatron production schedules add warmup/1F1B/cooldown to reduce bubbles")
        print(f"max dW0 error: {errs[0].item():.3e}")
        print(f"max dW1 error: {errs[1].item():.3e}")
        print("PASS" if errs.max().item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
