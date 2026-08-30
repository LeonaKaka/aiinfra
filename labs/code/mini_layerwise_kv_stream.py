#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Layer-by-layer KV streaming: prefill sends each layer as it is produced; decode waits per layer."
    )
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=8)
    return p.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=2 ...")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("This lab intentionally uses 2 ranks: prefill and decode")
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, device


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("hd,thd->ht", q, k) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", probs, v)


def make_layer_kv(prompt, wk, wv):
    k = torch.einsum("thd,ldf->lthf", prompt, wk)
    v = torch.einsum("thd,ldf->lthf", prompt, wv)
    return k, v


def decode_with_layers(q0, k_layers, v_layers):
    q = q0
    for l in range(k_layers.shape[0]):
        q = torch.tanh(q + attention(q, k_layers[l], v_layers[l]))
    return q


def main():
    args = parse_args()
    rank, device = init_dist()

    torch.manual_seed(1201)
    prompt = torch.randn(args.tokens, args.heads, args.head_dim, device=device)
    wk = torch.randn(args.layers, args.head_dim, args.head_dim, device=device) / args.head_dim**0.5
    wv = torch.randn(args.layers, args.head_dim, args.head_dim, device=device) / args.head_dim**0.5
    q0 = torch.randn(args.heads, args.head_dim, device=device)

    # Reference is local only for verification. The streamed path will not use it.
    k_ref, v_ref = make_layer_kv(prompt, wk, wv)
    out_ref = decode_with_layers(q0, k_ref, v_ref)

    recv_k = [torch.empty(args.tokens, args.heads, args.head_dim, device=device) for _ in range(args.layers)]
    recv_v = [torch.empty_like(recv_k[0]) for _ in range(args.layers)]

    dist.barrier()
    t0 = time.perf_counter()

    if rank == 1:
        # Post receives for every layer. Decode will only wait on layer L
        # immediately before it needs that layer's KV.
        works = []
        for l in range(args.layers):
            works.append((dist.irecv(recv_k[l], src=0), dist.irecv(recv_v[l], src=0)))

        q = q0.clone()
        wait_ms = []
        compute_ms = []
        for l, (wk_work, wv_work) in enumerate(works):
            tw = time.perf_counter()
            wk_work.wait()
            wv_work.wait()
            wait_ms.append((time.perf_counter() - tw) * 1000)

            tc = time.perf_counter()
            q = torch.tanh(q + attention(q, recv_k[l], recv_v[l]))
            compute_ms.append((time.perf_counter() - tc) * 1000)

        err = (q - out_ref).abs().max().detach()
        stat = torch.tensor(
            [sum(wait_ms), sum(compute_ms), (time.perf_counter() - t0) * 1000],
            dtype=torch.float64,
            device=device,
        )
    else:
        sends = []
        prefill_compute_ms = []
        for l in range(args.layers):
            tc = time.perf_counter()
            k_l = torch.einsum("thd,df->thf", prompt, wk[l]).contiguous()
            v_l = torch.einsum("thd,df->thf", prompt, wv[l]).contiguous()
            prefill_compute_ms.append((time.perf_counter() - tc) * 1000)
            sends.append(dist.isend(k_l, dst=1))
            sends.append(dist.isend(v_l, dst=1))

        for work in sends:
            work.wait()
        err = torch.tensor(0.0, device=device)
        stat = torch.tensor(
            [sum(prefill_compute_ms), 0.0, (time.perf_counter() - t0) * 1000],
            dtype=torch.float64,
            device=device,
        )

    dist.broadcast(err, src=1)
    gathered = [torch.empty_like(stat) for _ in range(2)] if rank == 0 else None
    dist.gather(stat, gather_list=gathered, dst=0)

    if rank == 0:
        print("=== Mini Layer-by-Layer KV Streaming ===")
        print(f"backend={dist.get_backend()} device={device.type} layers={args.layers}")
        print("prefill behavior        : compute layer L KV -> isend immediately")
        print("decode behavior         : wait(layer L) -> consume -> next layer")
        print(f"max final decode error  : {err.item():.3e}")
        if gathered is not None:
            print(f"prefill layer-compute sum ms : {gathered[0][0].item():.3f}")
            print(f"decode wait sum ms            : {gathered[1][0].item():.3f}")
            print(f"decode compute sum ms         : {gathered[1][1].item():.3f}")
        print("NOTE: timings show dependency structure only; use a GPU profiler to prove overlap.")
        print("PASS" if err.item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
