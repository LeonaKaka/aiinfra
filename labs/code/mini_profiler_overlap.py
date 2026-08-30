#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function


def parse_args():
    p = argparse.ArgumentParser(description='Profiler-ready async reduce-scatter overlap lab.')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--ffn', type=int, default=128)
    p.add_argument('--out', type=int, default=64)
    p.add_argument('--trace-dir', type=str, default='profiler_traces')
    return p.parse_args()


def init_dist():
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        raise RuntimeError('Launch with torchrun --standalone --nproc-per-node=2 ...')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world
    dist.init_process_group('nccl' if use_cuda else 'gloo')
    if use_cuda:
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    return rank, world, device


def rs_async(flat_grad: torch.Tensor, world: int):
    if flat_grad.numel() % world:
        raise ValueError('gradient bucket numel must be divisible by WORLD_SIZE')
    out = torch.empty(flat_grad.numel() // world, device=flat_grad.device, dtype=flat_grad.dtype)
    work = dist.reduce_scatter_tensor(out, flat_grad.contiguous(), op=dist.ReduceOp.SUM, async_op=True)
    return out, work


def main():
    args = parse_args()
    rank, world, device = init_dist()
    if world != 2:
        raise ValueError('This lab is documented for WORLD_SIZE=2')
    if (args.ffn * args.out) % world or (args.hidden * args.ffn) % world:
        raise ValueError('W1/W2 gradient buckets must be divisible by WORLD_SIZE')

    torch.manual_seed(3101)
    global_batch = args.batch * world
    x_all = torch.randn(global_batch, args.hidden, device=device)
    t_all = torch.randn(global_batch, args.out, device=device)
    x = x_all[rank * args.batch : (rank + 1) * args.batch]
    target = t_all[rank * args.batch : (rank + 1) * args.batch]

    torch.manual_seed(3102)
    w1 = torch.randn(args.hidden, args.ffn, device=device) / args.hidden**0.5
    w2 = torch.randn(args.ffn, args.out, device=device) / args.ffn**0.5

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, acc_events=True) as prof:
        with record_function('forward'):
            pre = x @ w1
            h = torch.relu(pre)
            y = h @ w2
            dy = (y - target) / args.batch

        with record_function('backward_w2_ready'):
            grad_w2 = h.transpose(0, 1) @ dy
        with record_function('launch_reduce_scatter_w2'):
            shard_w2, work_w2 = rs_async(grad_w2.reshape(-1), world)

        # This compute is independent of the reduced W2 shard. On GPU, this is
        # where overlap may happen if the backend/hardware can make progress.
        with record_function('backward_hidden_and_w1'):
            dh = dy @ w2.transpose(0, 1)
            dpre = dh * (pre > 0)
            grad_w1 = x.transpose(0, 1) @ dpre

        with record_function('launch_reduce_scatter_w1'):
            shard_w1, work_w1 = rs_async(grad_w1.reshape(-1), world)

        with record_function('wait_for_gradient_shards'):
            work_w2.wait()
            work_w1.wait()
            shard_w2.div_(world)
            shard_w1.div_(world)

    trace_path = trace_dir / f'overlap_rank{rank}_{device.type}.json'
    prof.export_chrome_trace(str(trace_path))

    # Dense global-batch reference.
    w1r = w1.detach().clone().requires_grad_(True)
    w2r = w2.detach().clone().requires_grad_(True)
    yr = torch.relu(x_all @ w1r) @ w2r
    loss = 0.5 * (yr - t_all).square().sum() / global_batch
    loss.backward()

    full_w2 = w2r.grad.reshape(-1)
    full_w1 = w1r.grad.reshape(-1)
    n2 = full_w2.numel() // world
    n1 = full_w1.numel() // world
    ref_w2 = full_w2[rank * n2 : (rank + 1) * n2]
    ref_w1 = full_w1[rank * n1 : (rank + 1) * n1]
    errs = torch.tensor([
        (shard_w2 - ref_w2).abs().max(),
        (shard_w1 - ref_w1).abs().max(),
    ], device=device)
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)

    if rank == 0:
        print('=== Profiler-ready Overlap Lab ===')
        print(f'backend={dist.get_backend()} device={device.type} world_size={world}')
        print('trace markers: forward -> W2 ready -> launch RS -> backward W1 -> launch RS -> wait')
        print(f'max W2 shard error : {errs[0].item():.3e}')
        print(f'max W1 shard error : {errs[1].item():.3e}')
        print(f'trace files        : {trace_dir}/overlap_rank*_{{cpu|cuda}}.json')
        if device.type == 'cpu':
            print('NOTE: CPU/Gloo trace validates ordering only; it does not prove GPU communication-compute overlap.')
        else:
            print('CHECK: inspect the trace and verify NCCL kernels overlap compute before claiming overlap.')
        print('PASS' if errs.max().item() < 1e-5 else 'FAIL')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
